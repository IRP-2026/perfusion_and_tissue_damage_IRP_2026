"""
Verification of a multi-compartment Darcy flow model with mixed Dirichlet and
Neumann boundary conditions using the method of manufactured solutions (MMS)

@author: Tamas Istvan Jozsa
"""

def evaluate_function(u, x):
    comm = u.function_space().mesh().mpi_comm()
    if comm.size == 1:
        return u(*x)

    # Find whether the point lies on the partition of the mesh local
    # to this process, and evaulate u(x)
    cell, distance = mesh.bounding_box_tree().compute_closest_entity(Point(*x))
    u_eval = u(*x) if distance < DOLFIN_EPS else None

    # Gather the results on process 0
    comm = mesh.mpi_comm()
    computed_u = comm.gather(u_eval, root=0)

    # Verify the results on process 0 to ensure we see the same value
    # on a process boundary
    if comm.rank == 0:
        global_u_evals = np.array([y for y in computed_u if y is not None], dtype=np.double)
        assert np.all(np.abs(global_u_evals[0] - global_u_evals) < 1e-9)
    
        computed_u = global_u_evals[0]
    else:
        computed_u = None

    # Broadcast the verified result to all processes
    computed_u = comm.bcast(computed_u, root=0)

    return computed_u


#%% IMPORT INSTALLED MODULES
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time

comm=MPI.comm_world
rank = comm.Get_rank()

# solver runs is "silent" mode
set_log_level(50)

start1 = time.time()


#%% SETTINGS
n = np.array([8,16,32])
DBC = 0
expr_degre = 4
z = np.linspace(0, 1, 201)

error_L2 = np.zeros(len(n)*2)
grad1_array = np.zeros(len(n)*2)
num_ele = np.zeros(len(n)*2)
order_ele = np.zeros(len(n)*2)
t_solve = np.zeros(len(n)*2)

#%% PARAMETERS
beta12 = Expression('1.0+0.5*(1.0+tanh(10*(x[2]-0.5)))',degree=expr_degre)
beta23 = Expression('2.0*(1.0+0.5*(1.0+tanh(10*(x[2]-0.5))))',degree=expr_degre)

beta21 = beta12
beta32 = beta23

k1   = Expression((('0.0','0.0','0.0'),
                   ('0.0','0.0','0.0'),
                   ('0.0','0.0','0.1*(2-x[2])')),degree=expr_degre)
K2   = as_matrix( ( (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1), ) )
k3   = Expression((('0.0','0.0','0.0'),
                   ('0.0','0.0','0.0'),
                   ('0.0','0.0','0.2*(2-x[2])')),degree=expr_degre)

sigma1expr = Expression(\
'-0.2*(16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4))*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*(2 - x[2]) + \
 0.2*(16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4))*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*x[2] + \
((16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4))*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))* \
pow(x[2],2) - (16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4))*(16*pow(x[1],2) - 32*pow(x[1],3) + \
16*pow(x[1],4))*(16*pow(x[2],2) - 32*pow(x[2],3) + 16*pow(x[2],4)))*(1 + 0.5*(1 + tanh(10*(-0.5 + x[2]))))' \
,degree=expr_degre)

sigma2expr = Expression(
'-0.1*(16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4))*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*(32 - \
192*x[2] + 192*pow(x[2],2)) - 0.1*(16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4))*(32 - 192*x[1] + \
192*pow(x[1],2))*(16*pow(x[2],2) - 32*pow(x[2],3) + 16*pow(x[2],4)) - 0.1*(32 - 192*x[0] + 192*pow(x[0],2))* \
(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*(16*pow(x[2],2) - 32*pow(x[2],3) + 16*pow(x[2],4)) + \
2*((-(-16*pow(x[0],2) + 32*pow(x[0],3) - 16*pow(x[0],4)))*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*pow(x[2],2) +\
(16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4))*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*(16*pow(x[2],2) - 32*pow(x[2],3) + 16*pow(x[2],4)))*\
(1 + 0.5*(1 + tanh(10*(-0.5 + x[2])))) + ((-(16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4)))*\
(16* pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*pow(x[2],2) + (16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4))*\
(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*(16*pow(x[2],2) - 32*pow(x[2],3) + 16*pow(x[2],4)))* \
(1 + 0.5*(1 + tanh(10*(-0.5 + x[2]))))' \
,degree=expr_degre)

sigma3expr = Expression(
'-0.4*(-16*pow(x[0],2) + 32*pow(x[0],3) - 16*pow(x[0],4))*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*(2 - x[2]) + \
0.4*(-16*pow(x[0],2) + 32*pow(x[0],3) - 16*pow(x[0],4))*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*x[2] + \
2*((-16*pow(x[0],2) + 32*pow(x[0],3) - 16*pow(x[0],4))*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*pow(x[2],2) \
- (16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4))*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))* \
(16*pow(x[2],2) - 32*pow(x[2],3) + 16*pow(x[2],4)))*(1 + 0.5*(1 + tanh(10*(-0.5 + x[2]))))' \
,degree=expr_degre)

p1_sol = '(16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4))*pow(x[2],2)*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))'
p2_sol = '(16*pow(x[0],2) - 32*pow(x[0],3) + 16*pow(x[0],4))*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))*(16*pow(x[2],2) - 32*pow(x[2],3) + 16*pow(x[2],4))'
p3_sol = '(-16*pow(x[0],2) + 32*pow(x[0],3) - 16*pow(x[0],4))*pow(x[2],2)*(16*pow(x[1],2) - 32*pow(x[1],3) + 16*pow(x[1],4))'
p_sol = Expression((p1_sol,p2_sol,p3_sol),degree=expr_degre)

K1 = k1
K3 = k3
sigma1 = sigma1expr
sigma2 = sigma2expr
sigma3 = sigma3expr

points = [(0.5, 0.5, z_) for z_ in z] # 1D points
p_ana_line = np.array([p_sol(point) for point in points])

#%% periodic BC
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool((near(x[0], 0) or near(x[1], 0)) and 
            (not ((near(x[0], 1) and near(x[1], 0)) or 
                  (near(x[0], 0) and near(x[1], 1)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1
            y[1] = x[1] - 1 
            y[2] = x[2]
        elif near(x[0], 1):
            y[0] = x[0] - 1
            y[1] = x[1]
            y[2] = x[2]
        elif near(x[1], 1):
            y[0] = x[0]
            y[1] = x[1] - 1
            y[2] = x[2]
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000


#%% CREATE BOX MESH
p1 = Point(0,0,0)
p2 = Point(1,1,1)

p_num_line = []

for j in range(1):
    for i in range(len(n)):
        if rank ==0: print(i,j)
        fe_degr = j+1
        res_fldr = 'res_o'+str(fe_degr)+'_n'+'{:02d}'.format(n[i])+'/'
        mesh = BoxMesh(p1, p2, n[i], n[i], n[i])
        
        # Define function space for system of pressures
        P1 = FiniteElement('P', tetrahedron, fe_degr)
        element = MixedElement([P1, P1, P1])
        
        if fe_degr == 1:
            Vvel = VectorFunctionSpace(mesh, "DG", 0)
        else:
            Vvel = VectorFunctionSpace(mesh, "Lagrange", fe_degr-1)
        
        if DBC == 0:
            V = FunctionSpace(mesh, element, constrained_domain=PeriodicBoundary())
            K_space = TensorFunctionSpace(mesh, "DG", 0, constrained_domain=PeriodicBoundary())
            sigma_space = FunctionSpace(mesh, "DG", 0, constrained_domain=PeriodicBoundary())
        else:
            V = FunctionSpace(mesh, element)
            K_space = TensorFunctionSpace(mesh, "DG", 0)
            sigma_space = FunctionSpace(mesh, "DG", 0)   
        
        # Define test functions
        v_1, v_2, v_3 = TestFunctions(V)
        
        # Define functions for pressures
        p = TrialFunction(V)
        
        # Split system functions to access components
        p_1, p_2, p_3 = split(p)
        
        

        
        
        #%% FE SOLVER
        # Define Dirichlet boundary
        if DBC == 1:
            def boundary(x):
                return x[0] < 0+DOLFIN_EPS or x[0] > 1.0-DOLFIN_EPS or x[1] < 0+DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS or x[2] < 0+DOLFIN_EPS or x[2] > 1.0 - DOLFIN_EPS
        else:
            def boundary(x):
                return x[2] > 1.0-DOLFIN_EPS
        # Define boundary condition
        p_bc1 = Expression(p1_sol, degree = expr_degre)
        bc1 = DirichletBC(V.sub(0), p_bc1, boundary)
        if DBC == 1:
            p_bc2 = Expression(p2_sol, degree = 2)
            bc2 = DirichletBC(V.sub(1), p_bc2, boundary)
        p_bc3 = Expression(p3_sol, degree = expr_degre)
        bc3 = DirichletBC(V.sub(2), p_bc3, boundary)
        
        # Define variational problem
        a = \
            -inner(K1*grad(p_1), grad(v_1))*dx - beta12*(p_1-p_2)*v_1*dx \
            -inner(K2*grad(p_2), grad(v_2))*dx - beta21*(p_2-p_1)*v_2*dx - beta23*(p_2-p_3)*v_2*dx \
            -inner(K3*grad(p_3), grad(v_3))*dx - beta32*(p_3-p_2)*v_3*dx
        L = - sigma1*v_1*dx - sigma2*v_2*dx - sigma3*v_3*dx
        
        
        # Define functions for pressures
        p = Function(V)        
        
        # Compute solution
        if DBC == 1:
            problem = LinearVariationalProblem(a, L, p, [bc1,bc2,bc3])
        else:
            problem = LinearVariationalProblem(a, L, p, [bc1,bc3])
        
        # solver settings
        solver = LinearVariationalSolver(problem)
        prm = solver.parameters
        prm['linear_solver'] = 'bicgstab'
        prm['preconditioner'] = 'amg'
        
        start2 = time.time()
        solver.solve()
        end2 = time.time()
        if rank ==0: print ('linear solver time= ', end2 - start2, '[s]')
        
        error_L2[j*len(n)+i] = errornorm(p_sol,p,degree_rise=2)
        # p_num_line.append( np.array([p(point) for point in points]) ) # serial implementation
        p_num_line.append( np.array([evaluate_function(p, point) for point in points]) ) # parallel implementation
        
        grad1_array[j*len(n)+i] = (p_num_line[-1][-1,0]-p_num_line[-1][-2,0])/(z[-1]-z[-2])
        num_ele[j*len(n)+i] = MPI.sum(MPI.comm_world, mesh.num_cells())
        order_ele[j*len(n)+i] = fe_degr
        t_solve[j*len(n)+i] = end2 - start2

vtkfile = File(res_fldr+'press.pvd')
vtkfile << p

end1 = time.time()

if rank ==0: print ('execution time = ', end1 - start1, '[s]')

NGS = pow(2,len(n)-1)*1/2**np.arange(len(n))

#%% save results
data1_header = 'n,num. ele.,NGS,ele. ord.,L2-norm,grad1(0.5;0.5;1)'
data1 = np.array([np.concatenate((n,n),axis=0),num_ele,np.concatenate((NGS,NGS),axis=0),order_ele,error_L2,grad1_array,t_solve])
data1 = data1.transpose()


p_num_rearranged = np.zeros([len(z),3*len(n)*2])
for i in range(len(n)):
    p_num_rearranged[:,i*3:i*3+3] = p_num_line[i]

if rank ==0: np.savetxt('grid_conv.csv',data1,delimiter=',',header=data1_header)
if rank ==0: np.savetxt('z.csv',z,delimiter=',')
if rank ==0: np.savetxt('p_ana.csv',np.array(p_ana_line),delimiter=',')
if rank ==0: np.savetxt('p_num.csv',p_num_rearranged,delimiter=',')
