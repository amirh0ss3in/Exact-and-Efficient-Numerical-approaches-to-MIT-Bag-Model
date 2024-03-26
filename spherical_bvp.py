import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


R = 1
m = 0
k = -1


# Define the function for the differential equation system
def fun(x, y):
    dfdr = y[0]*(k-1)/x + (m - E)*y[1]
    dgdr = (-k-1)/x * y[1] + (m+E) * y[0]
    return np.vstack((dfdr, dgdr))

# Define the boundary condition residuals
def bc(ya, yb):
    return np.array([yb[0]-1, yb[1]+1])

# Create a mesh in the interval [0, 1]
x = np.linspace(1e-6, R, 1000)

# Initial guess
y_init_1 = np.full((2, x.size), 3)

# Define the objective function for minimization
def objective(E_):
    global E
    E = E_[0]
    # Solve the boundary value problem
    res = solve_bvp(fun, bc, x, y_init_1, tol = 1e-5, max_nodes=1800)
    f = res.sol(x)[0]
    g = res.sol(x)[1]
    rho = np.dot(f**2+g**2, x**2)

    return rho


for pos in [True, False]:
    if pos:
        bounds_list = [(1, 4), (4, 8), (8, 10), (10, 12), (12, 15)]  # Add more bounds as needed
    else:
        bounds_list = [(-4, -1), (-8, -4), (-12, -8), (-16, -12), (-17, -15)]  # Add more bounds as needed

    optimal_E_vals = []

    plt.figure(figsize=(8, 6))

    for energy_level, bounds in enumerate(bounds_list):
        # Perform the differential evolution
        result = differential_evolution(objective, [bounds], tol = 1e-4)

        # Solve the boundary value problem with optimal E
        E = result.x[0]
        res = solve_bvp(fun, bc, x, y_init_1, tol = 1e-5)
        optimal_f = res.sol(x)[0]
        optimal_g = res.sol(x)[1]

        print(f"The optimal E for bounds {bounds} is {E} with a rho of {result.fun}")
        optimal_E_vals.append(E)

        # Plot the solution
        plt.plot(x, optimal_f, linewidth=2, label=r'$f$'+f', {energy_level+1}'+ r'$S_\dfrac{1}{2}$')
        plt.plot(x, optimal_g, linewidth=2, label=r'$g$'+f', {energy_level+1}'+ r'$S_\dfrac{1}{2}$')

    plt.legend(bbox_to_anchor=(1.2, 1.0), loc='upper right')

    plt.xlabel(r'$r$', fontsize=14)
    plt.grid(True)

    if pos:
        plt.title(r'Positive Energy level Solutions of the BVP ($\psi_+)$')
        plt.tight_layout()
        plt.savefig('Results/BVP_positive.pdf')
        np.savetxt('Results/BVP_positive_energies.txt', optimal_E_vals, fmt= '%.15f')
    else:
        plt.title(r'Negative Energy level Solutions of the BVP ($\psi_-)$')
        plt.tight_layout()
        plt.savefig('Results/BVP_negative.pdf')
        np.savetxt('Results/BVP_negative_energies.txt', optimal_E_vals, fmt= '%.15f')


    plt.show()
