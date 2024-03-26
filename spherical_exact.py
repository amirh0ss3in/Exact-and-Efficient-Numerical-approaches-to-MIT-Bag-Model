import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brenth
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def const_q(E,m,U0):
    q = np.sqrt((m+U0)**2 - E**2)
    return q

def const_p(E,m):
    p = np.sqrt(E**2-m**2)
    return p

def const_N(R,E,m,U0):
    p, q = const_p(E, m), const_q(E, m, U0)
    return (R/(2*p**2) + R/(2*(m+E)**2) + (np.sin(2*p*R)/(4 * p**3)) * (p**2/((m+E)**2) - 1) - np.sin(p*R)**2 / (p**2 * R * (m+E)**2) + (np.sin(p*R)**2 / (2*p**2)) * (1/q + (q+2/R)/((m+E+U0)**2)))**-0.5

def const_M(R,E,m,U0):
    p, q = const_p(E, m), const_q(E,m, U0)
    return const_N(R,E,m,U0) * q * np.exp(q*R) * np.sin(p*R) / p


def solution(r, R, E, m, U0):
    p, q = const_p(E, m), const_q(E,m, U0)
    M, N = const_M(R,E,m,U0), const_N(R,E,m,U0)

    f = np.zeros_like(r)
    g = np.zeros_like(r)

    mask1 = r <= R
    mask2 = r > R

    r1 = r[mask1]
    r2 = r[mask2]

    g[mask1] = N * np.sin(p*r1)/ (p*r1)
    f[mask1] = -N * p * (np.sin(p*r1)/ ((p*r1)**2) - np.cos(p*r1)/ (p*r1)) / (m+E)

    g[mask2] = M*np.exp(-q*r2)/(q*r2)
    f[mask2] = -M*q*np.exp(-q*r2) * (1/(q*r2) + 1/(q*r2)**2) / (m+E+U0)

    return f, g

def solve_E(m, R, U0, E_range):
    def f(E, inf_limit = True):
        try:
            if not inf_limit:
                if m + U0 - E > 0:
                    term0 = np.tan(R * np.sqrt(E**2-m**2))
                    
                    if np.isclose(term0, 0):
                        return np.inf
                    
                    term1 = np.sqrt((E-m)/(E + m)) / term0
                    term2 = - 1/(R * (E + m))
                    term3 = np.sqrt((m + U0 - E)/(m + U0 + E))
                    term4 = 1/ (R * (m + U0 + E))
                    return term1 + term2 + term3 + term4
                
                
                else:
                    return np.nan
            
            elif inf_limit:
                return 1/(np.tan(const_p(E,m)*R) * ((m+E)*R)) - 1/((m+E)*const_p(E,m)*R**2) + 1/(const_p(E, m)*R)

        except ValueError:
            return np.inf

    roots = []

    for i in range(len(E_range) - 1):
        try:
            root = brenth(f, E_range[i], E_range[i + 1])
            roots.append(root)
        except ValueError:
            pass

    return np.array(roots)


def compute_E_values(m, R, U0_values, E_range, N = 10):
    E_values = [solve_E(m, R, U0, E_range) for U0 in tqdm(U0_values)]
    E_values_all = []
    start = 1 if np.all(E_range < 0) else 0

    for i in range(start, 2*N, 2):
        E_values_all.append([E[i] if len(E) > i else np.nan for E in E_values])
    return np.array(E_values_all)
    

def plot_energy_levels(U0_values, E_values_all, title, save_name):
    plt.figure(figsize=(8, 6))
    colormap = plt.cm.get_cmap('viridis', len(E_values_all))
    for i in range(len(E_values_all)):
        plt.plot(U0_values, E_values_all[i], label=f'Energy level {i+1}', linewidth=2, color=colormap(i))
    plt.legend()

    plt.xlabel(r'$U_0$', fontsize=14)
    plt.ylabel(r'$E$', fontsize=14)
    plt.grid(True)
    plt.title(title, fontsize=16)  
    plt.tight_layout()
    plt.savefig(save_name)  

    plt.show()

def plot_wavefunctions(r, U0_values, E_values_all, title, save_name):
    plt.figure(figsize=(8, 6))
    for energy_level in range(len(E_values_all)):
        f, g = solution(r, R, E_values_all[energy_level, -1], m, U0=U0_values[-1])
        plt.plot(r, f, linewidth=2, label=r'$f$'+f', {energy_level+1}'+ r'$S_\dfrac{1}{2}$')
        plt.plot(r, g, linewidth=2, label=r'$g$'+f', {energy_level+1}'+ r'$S_\dfrac{1}{2}$')
    plt.title(title)

    plt.legend(bbox_to_anchor=(1.2, 1.0), loc='upper right')
    plt.grid(True)
    plt.xlabel(r'$r$', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_name)

    plt.show()


m = 0
R = 1

U0_values = np.linspace(0, 5000, 100) 
r = np.linspace(1e-5, 5, 1000)

E_range_negative = np.linspace(-1, -30, 100)
E_range_positive = np.linspace(1, 30, 100)

E_values_all = compute_E_values(m, R, U0_values, E_range_positive, N = 5)
np.savetxt('Results/exact_positive_energies.txt', E_values_all[:,-1], fmt= '%.15f')
plot_energy_levels(U0_values, E_values_all, title=r'Positive Energy levels ($E_+$) as a function of $U_0$', save_name = 'Results/Positive_Energy_levels.pdf')
plot_wavefunctions(r, U0_values, E_values_all, title = r'Matter Wavefunction $\psi_+$', save_name = 'Results/Matter_Wavefunction.pdf')

E_values_all = compute_E_values(m, R, U0_values, E_range_negative, N = 5)
np.savetxt('Results/exact_negative_energies.txt', E_values_all[:,-1], fmt= '%.15f')
plot_energy_levels(U0_values, E_values_all, title=r'Negative Energy levels ($E_-$) as a function of $U_0$', save_name = 'Results/Negative_Energy_levels.pdf')
plot_wavefunctions(r, U0_values, E_values_all, title = r'Anti-Matter Wavefunction $\psi_-$', save_name = 'Results/Anti_Matter_Wavefunction.pdf')

