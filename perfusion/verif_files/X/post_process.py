import os
import matplotlib.pyplot as plt
import numpy as np

# Data configuration
data_dir = "./mms_dbc0_periodic_results/"
conv_file = os.path.join(data_dir, "grid_conv.csv")

if not os.path.exists(conv_file):
    raise FileNotFoundError(f"Le fichier {conv_file} est introuvable. As-tu lancé la simulation ?")

# Loading the CSV file
# Column 0 = n, Column 3 = fe_degr, Column 4 = L2_norm
data = np.loadtxt(conv_file, delimiter=",")

# Filter by finite element order
data_P1 = data[data[:, 3] == 1]
data_P2 = data[data[:, 3] == 2]

# Extract columns n (column 0) and L2 norm (column 4)
n_P1 = data_P1[:, 0]
l2_P1  = data_P1[:, 4]

n_P2 = data_P2[:, 0]
l2_P2  = data_P2[:, 4]

# Graph generation
plt.figure(figsize=(7, 6))

# Plotting the simulation points for P1 and P2
plt.loglog(n_P1, l2_P1, 'ko', markersize=6, label=r"$1^{\text{st}}$ ord. FE (P1)")
plt.loglog(n_P2, l2_P2, 'ks', markersize=6, label=r"$2^{\text{nd}}$ ord. FE (P2)")

# Construction of reference slopes for P1 and P2
n_ref = np.sort(np.unique(data[:, 0])) # [2, 4, 8, 16, 32]


y_slope1 = l2_P1[0] * (n_P1[0] / n_ref)**1 
plt.loglog(n_ref, y_slope1, 'k:', label="slope -1")


y_slope2 = l2_P2[0] * (n_P2[0] / n_ref)**2
plt.loglog(n_ref, y_slope2, 'k--', label="slope -2")


plt.xlabel("$n$", fontsize=12)
plt.ylabel("$L^2-\t{norm}$", fontsize=12)


plt.xticks(n_ref, labels=[str(int(val)) for val in n_ref])

plt.xlim(1.5, 40)
plt.ylim(1e-4, 5e-1)

plt.grid(False) 

plt.legend(loc='upper right', frameon=False, fontsize=10)

plt.tight_layout()

output_path = os.path.join(data_dir, "convergence_plot_with_n.png")
plt.savefig(output_path, dpi=300)
print(f"Graphique de convergence (axe n) sauvegardé dans : {output_path}")
plt.show()