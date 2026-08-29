import os
import time

import numpy as np
import ufl
from mpi4py import MPI

import dolfinx
from dolfinx import fem, io
from dolfinx import mesh as dmesh

import dolfinx_mpc #boundary conditions

import finite_element_functions as fe_module

comm = MPI.COMM_WORLD
rank = comm.rank

# SETTINGS
n_list = np.array([2, 4, 8, 16, 32])
degrees = [1, 2]
z = np.linspace(0.0, 1.0, 201)
points = np.array([[0.5, 0.5, zz] for zz in z], dtype=np.float64)

results_rows = []
p_num_line_all = []


# Helper: parallel-safe point evaluation
def evaluate_function_at_points(u, eval_points, msh):
    eval_points = np.asarray(eval_points, dtype=np.float64)
    value_shape = u.function_space.value_shape
    value_size = int(np.prod(value_shape)) if len(value_shape) > 0 else 1
    tdim = msh.topology.dim
    try:
        bb_tree = dolfinx.geometry.bb_tree(msh, tdim)
    except AttributeError:  
        bb_tree = dolfinx.geometry.BoundingBoxTree(msh, tdim)

    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, eval_points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(msh, cell_candidates, eval_points)

    local_values = np.full((len(eval_points), value_size), np.nan)
    for i in range(len(eval_points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_values[i, :] = u.eval(eval_points[i:i + 1], links[0:1])

    gathered = comm.gather(local_values, root=0)
    if rank == 0:
        combined = gathered[0].copy()
        for arr in gathered[1:]:
            mask = np.isnan(combined) & ~np.isnan(arr)
            combined[mask] = arr[mask]
    else:
        combined = None
    combined = comm.bcast(combined, root=0)
    return combined


# Manufactured pressures
def p1_sol_np(x):
    return (16*x[0]**2 - 32*x[0]**3 + 16*x[0]**4) * x[2]**2 * (16*x[1]**2 - 32*x[1]**3 + 16*x[1]**4)

def p2_sol_np(x):
    return (16*x[0]**2 - 32*x[0]**3 + 16*x[0]**4) * (16*x[1]**2 - 32*x[1]**3 + 16*x[1]**4) * (16*x[2]**2 - 32*x[2]**3 + 16*x[2]**4)

def p3_sol_np(x):
    return (-16*x[0]**2 + 32*x[0]**3 - 16*x[0]**4) * x[2]**2 * (16*x[1]**2 - 32*x[1]**3 + 16*x[1]**4)

p_sol_np = [p1_sol_np, p2_sol_np, p3_sol_np]
p_ana_line = np.column_stack([f(points.T) for f in p_sol_np])


# Geometric definition for periodicity 
# On X : right face (x=1) is the slave of left face (x=0)
def periodic_boundary_x(x):
    return np.isclose(x[0], 1.0)

def periodic_relation_x(x):
    out_x = np.copy(x)
    out_x[0] = 0.0
    return out_x

# On Y : upper face (y=1) is the slave of lower face (y=0)
def periodic_boundary_y(x):
    return np.isclose(x[1], 1.0)

def periodic_relation_y(x):
    out_x = np.copy(x)
    out_x[1] = 0.0
    return out_x


# MAIN LOOP
for fe_degr in degrees:
    for n in n_list:

        if rank == 0:
            print(f"--- P{fe_degr}, n = {n}")

        msh = dmesh.create_box(
            comm,
            [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
            [int(n), int(n), int(n)],
            cell_type=dmesh.CellType.tetrahedron,
        )

        x = ufl.SpatialCoordinate(msh)

        # UFL expressions
        p1_sol = (16*x[0]**2 - 32*x[0]**3 + 16*x[0]**4) * x[2]**2 * (16*x[1]**2 - 32*x[1]**3 + 16*x[1]**4)
        p2_sol = (16*x[0]**2 - 32*x[0]**3 + 16*x[0]**4) * (16*x[1]**2 - 32*x[1]**3 + 16*x[1]**4) * (16*x[2]**2 - 32*x[2]**3 + 16*x[2]**4)
        p3_sol = (-16*x[0]**2 + 32*x[0]**3 - 16*x[0]**4) * x[2]**2 * (16*x[1]**2 - 32*x[1]**3 + 16*x[1]**4)

        beta12 = 1.5 + 0.5 * ufl.tanh(10 * x[2] - 5)
        beta23 = 3.0 + ufl.tanh(10 * x[2] - 5)
        beta21 = beta12
        beta32 = beta23

        K1 = ufl.as_matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1 * (2 - x[2])]])
        K2 = ufl.as_matrix([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
        K3 = ufl.as_matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.2 * (2 - x[2])]])

        sigma1 = beta12 * (p1_sol - p2_sol) - ufl.div(ufl.dot(K1, ufl.grad(p1_sol)))
        sigma2 = (beta21 * (p2_sol - p1_sol) + beta23 * (p2_sol - p3_sol) - ufl.div(ufl.dot(K2, ufl.grad(p2_sol))))
        sigma3 = beta32 * (p3_sol - p2_sol) - ufl.div(ufl.dot(K3, ufl.grad(p3_sol)))

        # Function space
        V, Vvel, v1, v2, v3, p, p1, p2, p3 = fe_module.allocation_functions_space(msh, fe_degr)

        # Variational problem
        LHS = (
            ufl.inner(K1 * ufl.grad(p1), ufl.grad(v1)) * ufl.dx + beta12 * (p1 - p2) * v1 * ufl.dx
            + ufl.inner(K2 * ufl.grad(p2), ufl.grad(v2)) * ufl.dx + beta12 * (p2 - p1) * v2 * ufl.dx + beta23 * (p2 - p3) * v2 * ufl.dx
            + ufl.inner(K3 * ufl.grad(p3), ufl.grad(v3)) * ufl.dx + beta23 * (p3 - p2) * v3 * ufl.dx
        )
        RHS = sigma1 * v1 * ufl.dx + sigma2 * v2 * ufl.dx + sigma3 * v3 * ufl.dx

        # 1. Define boundary z=1
        tdim = msh.topology.dim
        fdim = tdim - 1
        msh.topology.create_connectivity(fdim, tdim)

        boundary_facets_z1 = dmesh.locate_entities_boundary(
            msh, fdim, lambda coords: np.isclose(coords[2], 1.0)
        )

        # 2. Dirichlet z=1 only on capillary and venule
        bcs = []

        V1_sub, _ = V.sub(0).collapse()
        uD1 = fem.Function(V1_sub)
        uD1.interpolate(p1_sol_np) 
        bdofs1 = fem.locate_dofs_topological((V.sub(0), V1_sub), fdim, boundary_facets_z1)
        bcs.append(fem.dirichletbc(uD1, bdofs1, V.sub(0)))

        V3_sub, _ = V.sub(2).collapse()
        uD3 = fem.Function(V3_sub)
        uD3.interpolate(p3_sol_np) 
        bdofs3 = fem.locate_dofs_topological((V.sub(2), V3_sub), fdim, boundary_facets_z1)
        bcs.append(fem.dirichletbc(uD3, bdofs3, V.sub(2)))

        # 3. Periodicite on all compartments
        mpc = dolfinx_mpc.MultiPointConstraint(V)

        for i in range(V.num_sub_spaces):
            mpc.create_periodic_constraint_geometrical(V.sub(i), periodic_boundary_x, periodic_relation_x, bcs)
            mpc.create_periodic_constraint_geometrical(V.sub(i), periodic_boundary_y, periodic_relation_y, bcs)

        mpc.finalize()

        # 4.  Solving through mpc
        t_solve_start = time.time()

        problem = dolfinx_mpc.LinearProblem(
            LHS, RHS, mpc, bcs=bcs,
            petsc_options={
                "ksp_type": "bcgs",
                "pc_type": "hypre",
                "ksp_rtol": 1e-12
            }
        )
        p_num = problem.solve()

        t_solve_end = time.time()

        if rank == 0:
            print(f"\t MPC linear solver time = {t_solve_end - t_solve_start:.4f} [s]")

        # L2 error computation 
        qdeg = 2 * (fe_degr + 2)
        dx_q = ufl.dx(metadata={"quadrature_degree": qdeg})
        p1h, p2h, p3h = ufl.split(p_num)
        err_sq = ((p1h - p1_sol)**2 + (p2h - p2_sol)**2 + (p3h - p3_sol)**2) * dx_q
        err_local = fem.assemble_scalar(fem.form(err_sq))
        L2_norm = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))

        # Sampling 
        p_num_collapsed = [p_num.sub(i).collapse() for i in range(3)]
        p_num_line = np.column_stack(
            [evaluate_function_at_points(p_num_collapsed[i], points, msh)[:, 0] for i in range(3)]
        )
        p_num_line_all.append(p_num_line)

        grad1 = (p_num_line[-1, 0] - p_num_line[-2, 0]) / (z[-1] - z[-2])
        num_cells_global = msh.topology.index_map(tdim).size_global
        idx = int(np.where(n_list == n)[0][0])
        NGS = (2.0 ** (len(n_list) - 1)) / (2.0 ** idx)

        results_rows.append([n, num_cells_global, NGS, fe_degr, L2_norm, grad1, t_solve_end - t_solve_start])

        if rank == 0:
            print(f"\t L2-norm = {L2_norm:.6e}, dp1/dz = {grad1:.6f}")

# SAVE RESULTS
if rank == 0:
    out_dir = "./mms_dbc0_periodic_results/"
    os.makedirs(out_dir, exist_ok=True)
 
    data1 = np.array(results_rows)
    header = "n,num. ele.,NGS,ele. ord.,L2-norm,grad1(0.5;0.5;1),t_solve [s]"
    np.savetxt(out_dir + "grid_conv.csv", data1, delimiter=",", header=header)
    np.savetxt(out_dir + "z.csv", z, delimiter=",")
    np.savetxt(out_dir + "p_ana.csv", p_ana_line, delimiter=",")
 
    p_num_rearranged = np.hstack(p_num_line_all)
    np.savetxt(out_dir + "p_num.csv", p_num_rearranged, delimiter=",")
 
    print("\nDone. Results saved in:", out_dir)
    print(header)
    for row in results_rows:
        print(row)
    
        # Visualization through paraview
    p1_visu = p_num.sub(0).collapse()
    p1_visu.name = "Pression_Compartiment_1"

    p2_visu = p_num.sub(1).collapse()
    p2_visu.name = "Pression_Compartiment_2"

    p3_visu = p_num.sub(2).collapse()
    p3_visu.name = "Pression_Compartiment_3"

    with io.XDMFFile(msh.comm, "results_complete.xdmf", "w") as xdmf:
      xdmf.write_mesh(msh)
      xdmf.write_function(p1_visu)
      xdmf.write_function(p2_visu)
      xdmf.write_function(p3_visu)