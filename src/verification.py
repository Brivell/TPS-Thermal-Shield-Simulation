"""
Tests de convergence pour validation du code
GCH2545 - Projet TPS
"""

import numpy as np
import matplotlib.pyplot as plt
from tps_fct import flux_plasma, simulation_complete


def test_convergence_temporelle(L, q_max, prm, nx, ny, t_final, t_entree, sigma, T_initiale, T_struc, dt_values):
    """
    Test convergence en temps
    """
    print("\n" + "="*60)
    print("TEST CONVERGENCE TEMPORELLE")
    print("="*60)
    
    T_max_list = []
    for dt_test in dt_values:
        print(f"\nSimulation dt = {dt_test} s...")
        T_max, temps, T_evol = simulation_complete(nx, ny, dt_test, T_initiale, T_struc, t_entree, sigma, t_final, L, q_max, prm)
        T_max_list.append(T_max)
        print(f"  T_max = {T_max:.4f} C")

    # Calcul erreurs
    erreurs = []
    ordres = []
    
    print("\nAnalyse convergence:")
    print("-"*50)
    
    for i in range(1, len(T_max_list)):
        err = abs(T_max_list[i] - T_max_list[i-1])
        erreurs.append(err)

        # Ordre de convergence
        if i >= 2:
            ordre = np.log(erreurs[i-2] / erreurs[i-1]) / np.log(dt_values[i-2] / dt_values[i-1])
            ordres.append(ordre)
            print(f"dt={dt_values[i-1]:.2f} -> dt={dt_values[i]:.2f} : err={err:.4f}C, ordre p={ordre:.3f}")
        else:
            print(f"dt={dt_values[i-1]:.2f} -> dt={dt_values[i]:.2f} : err={err:.4f}C")
    
    # Ordre moyen
    if len(ordres) > 0:
        ordre_moy = np.mean(ordres)
        print(f"\nOrdre moyen: p = {ordre_moy:.2f}")
        
        if 0.8 <= ordre_moy <= 1.2:
            print("OK - Ordre 1 (Euler implicite)")
        elif 1.8 <= ordre_moy <= 2.2:
            print("OK - Ordre 2")
        else:
            print(f"Attention - ordre inattendu")
    
    return {
        "dt_values": dt_values,
        "T_max_list": T_max_list,
        "erreurs": erreurs,
        "ordre": ordres
    }


def test_convergence_spatiale(L, q_max, prm, dt, t_final, t_entree, sigma, T_initiale, T_struc, nx_values):
    """
    Test convergence en espace avec régression
    """
    print("\n" + "="*60)
    print("TEST CONVERGENCE SPATIALE")
    print("="*60)
    
    T_max_list = []
    h_values = []
    
    for nx_test in nx_values:
        ny_test = nx_test
        h = L / (nx_test - 1)
        h_values.append(h)
        
        print(f"\nMaillage {nx_test}x{ny_test} (h={h:.4f} m)")
        T_max, temps, T_evol = simulation_complete(nx_test, ny_test, dt, T_initiale, T_struc, t_entree, sigma, t_final, L, q_max, prm)
        T_max_list.append(T_max)
        print(f"  T_max = {T_max:.6f} C")
    
    # Erreurs
    erreurs = []
    ordres_spatial = []
    
    print("\nAnalyse convergence:")
    print("-"*50)
    
    for i in range(1, len(T_max_list)):
        erreur = abs(T_max_list[i] - T_max_list[i-1]) / abs(T_max_list[i]) * 100
        erreurs.append(erreur)
        
        if i >= 2:
            ordre = np.log(erreurs[i-2] / erreurs[i-1]) / np.log(h_values[i-1] / h_values[i])
            ordres_spatial.append(ordre)
            print(f"nx={nx_values[i-1]} -> nx={nx_values[i]} : err={erreur:.4f}%, ordre p={ordre:.3f}")
        else:
            print(f"nx={nx_values[i-1]} -> nx={nx_values[i]} : err={erreur:.4f}%")
    
    # Régression log-log
    if len(h_values) >= 3 and len(erreurs) >= 2:
        log_h = np.log(h_values[1:])
        log_err = np.log(erreurs)
        
        coeffs = np.polyfit(log_h, log_err, 1)
        p_reg = coeffs[0]
        
        print(f"\nRegression log-log: ordre p = {p_reg:.2f}")
        
        if 1.8 <= p_reg <= 2.2:
            print("OK - Ordre 2")
        else:
            print(f"Note: ordre different de 2")
    else:
        coeffs = None
        p_reg = None
    
    return {
        "nx_values": nx_values,
        "h_values": h_values,
        "T_max_list": np.array(T_max_list),
        "erreurs": erreurs,
        "ordre_spatial": ordres_spatial,
        "regression_coeffs": coeffs
    }


def test_cas_limite(nx=21, ny=21):
    """
    Validation: diffusion pure sans flux
    """
    print("\n" + "="*60)
    print("VALIDATION CAS LIMITE")
    print("="*60)
    
    from tps_fct import simulation_complete
    
    # Params
    class prm_test:
        rho = 1800.0
        cp = 800.0
        k_therm = 0.5
        epsilon = 0.85
    
    L = 0.1
    T_initiale = 20.0 + 273.15
    T_stru = 20.0 + 273.15
    q_max = 0.0  # Pas de flux
    t_entree = 1200.0
    t_final = 1800.0
    dt = 10.0
    sigma = 5.67e-8
    
    print(f"q_max = {q_max} W/m2")
    print(f"Maillage {nx}x{ny}")
    
    T_max, temps, T_evol = simulation_complete(nx, ny, dt, T_initiale, T_stru, t_entree, sigma, t_final, L, q_max, prm_test)
    
    # Vérif
    T_attendue = T_initiale - 273.15
    erreur = abs(T_max - T_attendue)
    
    print(f"\nT initiale: {T_attendue:.2f} C")
    print(f"T finale: {T_max:.2f} C")
    print(f"Erreur: {erreur:.4f} C")
    
    tol = 0.1
    ok = erreur < tol
    
    if ok:
        print(f"VALIDATION OK (err < {tol} C)")
    else:
        print(f"VALIDATION FAILED")
    
    return {"erreur_max": erreur, "validation_ok": ok}


def gen_tableau_latex_spatial(nx_values, h_values, T_max_list, erreurs, filename='tab_conv_spatial.tex'):
    """
    Génère tableau LaTeX convergence spatiale
    """
    print(f"\nGeneration tableau: {filename}")
    
    tableau = r"""\begin{table}[h]
\centering
\caption{Convergence spatiale}
\begin{tabular}{|c|c|c|c|}
\hline
Maillage & h [m] & $T_{max}$ [\textdegree C] & Erreur [\%] \\
\hline
"""
    
    for i, nx in enumerate(nx_values):
        h = h_values[i]
        T_max = T_max_list[i]
        
        if i == 0:
            err_str = r"---"
        else:
            err_str = f"{erreurs[i-1]:.4f}"
        
        tableau += f"{nx}$\\times${nx} & {h:.4f} & {T_max:.6f} & {err_str} \\\\\n"
    
    tableau += r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(filename, 'w') as f:
        f.write(tableau)
    
    print(f"Tableau sauvegarde: {filename}")


def gen_tableau_latex_temporel(dt_values, T_max_list, erreurs, ordres, filename='tab_conv_temporel.tex'):
    """
    Génère tableau LaTeX convergence temporelle
    """
    print(f"\nGeneration tableau: {filename}")
    
    tableau = r"""\begin{table}[h]
\centering
\caption{Convergence temporelle}
\begin{tabular}{|c|c|c|c|}
\hline
$\Delta t$ [s] & $T_{max}$ [\textdegree C] & Erreur [\textdegree C] & Ordre $p$ \\
\hline
"""
    
    for i, dt in enumerate(dt_values):
        T_max = T_max_list[i]
        
        if i == 0:
            err_str = r"---"
            ord_str = r"---"
        elif i == 1:
            err_str = f"{erreurs[i-1]:.4f}"
            ord_str = r"---"
        else:
            err_str = f"{erreurs[i-1]:.4f}"
            ord_str = f"{ordres[i-2]:.3f}"
        
        tableau += f"{dt:.2f} & {T_max:.6f} & {err_str} & {ord_str} \\\\\n"
    
    tableau += r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(filename, 'w') as f:
        f.write(tableau)
    
    print(f"Tableau sauvegarde: {filename}")


def plot_conv_loglog(h_values, erreurs, ordre_th=2, filename='conv_loglog.png'):
    """
    Graphique log-log convergence
    """
    print(f"\nGeneration graphique: {filename}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Données
    ax.loglog(h_values[1:], erreurs, 'bo-', linewidth=2, markersize=8, label='Erreur mesuree')
    
    # Ligne théorique
    h_ref = h_values[1]
    err_ref = erreurs[0]
    h_theo = np.array(h_values[1:])
    err_theo = err_ref * (h_theo / h_ref)**ordre_th
    ax.loglog(h_theo, err_theo, 'r--', linewidth=2, label=f'Ordre {ordre_th}')
    
    ax.set_xlabel('h [m]', fontsize=12)
    ax.set_ylabel('Erreur [%]', fontsize=12)
    ax.set_title('Convergence Spatiale', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Figure sauvegardee: {filename}")
    
    return fig
