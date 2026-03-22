"""
GCH2545 - Projet TPS
Simulation bouclier thermique
Analyse complete avec validation et sensibilite
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tps_fct import simulation_complete, simulation_principale
from verification import (test_convergence_temporelle, test_convergence_spatiale,
                          test_cas_limite, gen_tableau_latex_spatial,
                          gen_tableau_latex_temporel, plot_conv_loglog)


print("\n" + "="*70)
print("  SIMULATION TPS - BOUCLIER THERMIQUE")
print("="*70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*70 + "\n")

#============================================================================
# PARAMETRES
#============================================================================

# Propriétés matériau
class prm():
    rho = 1800.0           # kg/m3
    cp = 800.0             # J/(kg.K)
    k_therm = 0.5          # W/(m.K)
    epsilon = 0.85         # sans dimension

# Constantes
sigma = 5.67e-8        # Stefan-Boltzmann W/(m2.K4)

# Géométrie
L = 0.1               # m
X = [0, L]
Y = [0, L]

# Conditions
T_initiale = 20.0 + 273.15     
T_stru = 20.0 + 273.15          
T_securite = 175.0 + 273.15     

# Flux
t_entree = 1200.0      
q_max = 50000.0        

# Simulation
t_final = 1800.0       

# Maillage
nx = 21                
ny = 21               
dt = 10.0              

n_steps = int(t_final / dt)

print("PARAMETRES")
print("-" * 70)
print(f"  L = {L} m")
print(f"  Maillage: {nx}x{ny}")
print(f"  dt = {dt} s")
print(f"  t_final = {t_final} s")
print(f"  q_max = {q_max/1000:.0f} kW/m2")
print(f"  T limite = {T_securite - 273.15:.0f} C")
print("-" * 70 + "\n")

#============================================================================
# VALIDATION
#============================================================================

print("\n" + "="*70)
print("VALIDATION")
print("="*70)

# Cas limite
result_valid = test_cas_limite(nx=21, ny=21)

# Convergence spatiale
print("\n" + "="*70)
print("CONVERGENCE SPATIALE")
print("="*70)

nx_values = [11, 21, 31, 41]

res_conv_spatial = test_convergence_spatiale(
    L, q_max, prm, dt, t_final, t_entree, sigma, T_initiale, T_stru, nx_values
)

# Tableau LaTeX
gen_tableau_latex_spatial(
    nx_values, res_conv_spatial['h_values'], 
    res_conv_spatial['T_max_list'], res_conv_spatial['erreurs'],
    filename='tab_conv_spatial.tex'
)

# Graphiques convergence
fig_conv, ax1 = plt.subplots(figsize=(10, 7))

ax1.plot(nx_values, res_conv_spatial['T_max_list'], 'bo-', linewidth=2, markersize=10)
ax1.set_xlabel('nx = ny', fontsize=12)
ax1.set_ylabel('T_max [C]', fontsize=12)
ax1.set_title('Convergence Maillage', fontsize=13)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('conv_maillage.png', dpi=300)
print("\nFigure: conv_maillage.png")

# Log-log
if len(res_conv_spatial['erreurs']) >= 2:
    fig_log = plot_conv_loglog(
        res_conv_spatial['h_values'], res_conv_spatial['erreurs'], 
        ordre_th=2, filename='conv_loglog.png'
    )

# Convergence temporelle
print("\n" + "="*70)
print("CONVERGENCE TEMPORELLE")
print("="*70)

dt_values = [20, 10, 5, 2.5, 1.25]

res_conv_temp = test_convergence_temporelle(
    L, q_max, prm, nx, ny, t_final, t_entree, sigma, T_initiale, T_stru,
    dt_values=dt_values
)

# Tableau
gen_tableau_latex_temporel(
    res_conv_temp['dt_values'],
    res_conv_temp['T_max_list'],
    res_conv_temp['erreurs'],
    res_conv_temp['ordre'],
    filename='tab_conv_temporel.tex'
)

#============================================================================
# SIMULATION PRINCIPALE
#============================================================================

print("\n" + "="*70)
print("SIMULATION PRINCIPALE")
print("="*70)
print(f"  Maillage: {nx}x{ny}")
print(f"  dt: {dt} s")
print(f"  Steps: {n_steps}")

# Temps pour sauvegarder profils
temps_snap = [0, 300, 600, 900, 1200, 1500, 1800]

# Run
Res = simulation_principale(
    X, Y, nx, ny, dt, t_final, temps_snap, T_initiale, T_stru, t_entree,
    q_max, sigma, prm, verbose=True
)

# Extract
Pos_x = Res["Position_x"]
Pos_y = Res["Position_y"]
temps = Res['temps']
T_max_evol = Res['T_max_evolution']
T_face = Res["T_face_interne"]
T_snaps = Res["T_instantanes"]
temps_snaps_reels = Res['temps_instantanes']

print("\nSimulation terminee")

#============================================================================
# FIGURES
#============================================================================

print("\n" + "="*70)
print("GENERATION FIGURES")
print("="*70)

# Profils 2D
print("\n1. Profils 2D")

fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (t_snap, T_snap) in enumerate(zip(temps_snaps_reels, T_snaps)):
    ax = axes[idx]
    T_plot = T_snap - 273.15

    levels = np.linspace(np.min(T_plot), np.max(T_plot), 20)
    contour = ax.contourf(Pos_x, Pos_y, T_plot, levels=levels, cmap='hot')
    
    ax.set_xlabel('x [m]', fontsize=10)
    ax.set_ylabel('y [m]', fontsize=10)
    ax.set_title(f't = {t_snap:.0f} s (T_max = {np.max(T_plot):.1f} C)', fontsize=11)
    ax.set_aspect('equal')
    
    fig1.colorbar(contour, ax=ax, label='T [C]')

fig1.suptitle('Profils Temperature 2D', fontsize=14)
plt.tight_layout()
plt.savefig('profils_2D.png', dpi=300)
print("  Figure: profils_2D.png")

# T max evolution
print("\n2. T max evolution")

fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(temps, T_max_evol, 'r-', linewidth=2, label='T_max(t)')
ax.axhline(y=T_securite - 273.15, color='k', linestyle='--', linewidth=2, 
           label=f'T_limite = {T_securite - 273.15:.0f} C')
ax.axvline(x=t_entree, color='b', linestyle='--', linewidth=1.5, 
           label=f't_entree = {t_entree:.0f} s')

ax.set_xlabel('Temps [s]', fontsize=12)
ax.set_ylabel('T max [C]', fontsize=12)
ax.set_title('Evolution Temperature Maximale', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('T_max_evolution.png', dpi=300)
print("  Figure: T_max_evolution.png")

# Profil ligne mediane
print("\n3. Profil ligne mediane")

fig3, ax = plt.subplots(figsize=(10, 6))

j_mid = nx // 2  
y_line = Pos_y[:, j_mid]

colors = plt.cm.viridis(np.linspace(0, 1, len(temps_snaps_reels)))

for idx, (t_snap, T_snap) in enumerate(zip(temps_snaps_reels, T_snaps)):
    T_line = T_snap[:, j_mid] - 273.15  
    ax.plot(y_line, T_line, marker='o', linewidth=2, markersize=5,
            color=colors[idx], label=f't = {t_snap:.0f} s')

ax.set_xlabel('y [m]', fontsize=12)
ax.set_ylabel('T [C]', fontsize=12)
ax.set_title(f'Profil T le long x = L/2', fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('profil_mediane.png', dpi=300)
print("  Figure: profil_mediane.png")

# T face interne
print("\n4. T face interne")

fig4, ax = plt.subplots(figsize=(10, 6))
ax.plot(temps, T_face, 'b-', linewidth=2, label='T face interne')
ax.axhline(y=T_securite - 273.15, color='k', linestyle='--', linewidth=2, 
           label=f'T_limite = {T_securite - 273.15:.0f} C')
ax.axvline(x=t_entree, color='r', linestyle='--', linewidth=1.5, 
           label=f't_entree = {t_entree:.0f} s')

ax.set_xlabel('Temps [s]', fontsize=12)
ax.set_ylabel('T [C]', fontsize=12)
ax.set_title('Temperature Face Interne (centre)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('T_face_interne.png', dpi=300)
print("  Figure: T_face_interne.png")

T_finale = T_face[-1]
print(f"\n  T finale face interne: {T_finale:.2f} C")

#============================================================================
# ANALYSES SENSIBILITE
#============================================================================

print("\n" + "="*70)
print("SENSIBILITE")
print("="*70)

# k
print("\n1. Sensibilite k")
print("-" * 70)

k_vals = np.linspace(0.2, 15.0, 30)  
T_max_k = np.zeros(len(k_vals))

k_orig = prm.k_therm

for idx, k in enumerate(k_vals):
    if idx % 5 == 0:
        print(f"  [{idx+1}/{len(k_vals)}] k = {k:.2f} W/(m.K)...", end='')
    
    prm.k_therm = k
    T_max, _, _ = simulation_complete(nx, ny, dt, T_initiale, T_stru, t_entree, sigma, t_final, L, q_max, prm)
    T_max_k[idx] = T_max
    
    if idx % 5 == 0:
        print(f" T_max = {T_max:.2f} C")

prm.k_therm = k_orig

# Figure
fig_k, ax = plt.subplots(figsize=(10, 6))

ax.plot(k_vals, T_max_k, 'bo-', linewidth=2, markersize=6)
ax.axhline(y=175, color='r', linestyle='--', linewidth=2, label='T_limite = 175 C')

ax.fill_between(k_vals, 0, 175, alpha=0.2, color='green', label='Zone OK')
ax.fill_between(k_vals, 175, 200, alpha=0.2, color='red', label='Zone danger')

ax.set_xlabel('k [W/(m.K)]', fontsize=12)
ax.set_ylabel('T_max structure [C]', fontsize=12)
ax.set_title('Sensibilite: Conductivite k', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sensibilite_k.png', dpi=300)
print("\n  Figure: sensibilite_k.png")

# L
print("\n2. Sensibilite L")
print("-" * 70)

L_vals = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
T_max_L = np.zeros(len(L_vals))

L_orig = L

for idx, L_test in enumerate(L_vals):
    print(f"  [{idx+1}/{len(L_vals)}] L = {L_test*1000:.0f} mm...", end='')
    L = L_test
    T_max, _, _ = simulation_complete(nx, ny, dt, T_initiale, T_stru, t_entree, sigma, t_final, L, q_max, prm)
    T_max_L[idx] = T_max
    print(f" T_max = {T_max:.2f} C")

L = L_orig

# Figure
fig_L, ax = plt.subplots(figsize=(10, 6))

ax.plot(L_vals * 1000, T_max_L, 'gs-', linewidth=2, markersize=8)
ax.axhline(y=175, color='r', linestyle='--', linewidth=2, label='T_limite = 175 C')

ax.fill_between(L_vals * 1000, 0, 175, alpha=0.2, color='green', label='Zone OK')
ax.fill_between(L_vals * 1000, 175, 200, alpha=0.2, color='red', label='Zone danger')

ax.set_xlabel('L [mm]', fontsize=12)
ax.set_ylabel('T_max structure [C]', fontsize=12)
ax.set_title('Sensibilite: Epaisseur L', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sensibilite_L.png', dpi=300)
print("\n  Figure: sensibilite_L.png")

# q_max
print("\n3. Sensibilite q_max")
print("-" * 70)

q_vals = np.linspace(30000, 80000, 10)
T_max_q = np.zeros(len(q_vals))

q_orig = q_max

for idx, q_test in enumerate(q_vals):
    print(f"  [{idx+1}/{len(q_vals)}] q_max = {q_test/1000:.0f} kW/m2...", end='')
    q_max = q_test
    T_max, _, _ = simulation_complete(nx, ny, dt, T_initiale, T_stru, t_entree, sigma, t_final, L, q_max, prm)
    T_max_q[idx] = T_max
    print(f" T_max = {T_max:.2f} C")

q_max = q_orig

# Figure
fig_q, ax = plt.subplots(figsize=(10, 6))

ax.plot(q_vals / 1000, T_max_q, 'r^-', linewidth=2, markersize=8)
ax.axhline(y=175, color='k', linestyle='--', linewidth=2, label='T_limite = 175 C')

ax.fill_between(q_vals / 1000, 0, 175, alpha=0.2, color='green', label='Zone OK')
ax.fill_between(q_vals / 1000, 175, 250, alpha=0.2, color='red', label='Zone danger')

ax.set_xlabel('q_max [kW/m2]', fontsize=12)
ax.set_ylabel('T_max structure [C]', fontsize=12)
ax.set_title('Sensibilite: Flux Maximal q_max', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sensibilite_qmax.png', dpi=300)
print("\n  Figure: sensibilite_qmax.png")

#============================================================================
# SAUVEGARDE JSON
#============================================================================

print("\n" + "="*70)
print("SAUVEGARDE RESULTATS")
print("="*70)

resultats = {
    "date": datetime.now().isoformat(),
    "parametres": {
        "L": L,
        "nx": nx,
        "ny": ny,
        "dt": dt,
        "k": prm.k_therm,
        "rho": prm.rho,
        "cp": prm.cp
    },
    "resultats": {
        "T_max_structure": float(np.max(T_face)),
        "T_finale": float(T_face[-1]),
        "facteur_securite": float(175 - np.max(T_face))
    },
    "sensibilite": {
        "k_values": [float(x) for x in k_vals],
        "T_max_k": [float(x) for x in T_max_k],
        "L_values_mm": [float(x*1000) for x in L_vals],
        "T_max_L": [float(x) for x in T_max_L],
        "q_values": [float(x/1000) for x in q_vals],
        "T_max_q": [float(x) for x in T_max_q]
    }
}

with open('resultats_TPS.json', 'w') as f:
    json.dump(resultats, f, indent=2)

print("\n  Fichier: resultats_TPS.json")

#============================================================================
# RESUME
#============================================================================

print("\n" + "="*70)
print("RESUME")
print("="*70)

T_max_struct = np.max(T_face)
facteur_secu = 175 - T_max_struct

print(f"\nT max structure: {T_max_struct:.2f} C")
print(f"T finale: {T_face[-1]:.2f} C")
print(f"Facteur securite: {facteur_secu:.2f} C")

if facteur_secu > 0:
    print(f"\nCONCLUSION: Structure PROTEGEE")
    print(f"  Marge: {facteur_secu:.1f} C")
else:
    print(f"\nATTENTION: Structure a risque")

print("\nFICHIERS GENERES:")
print("-" * 70)
print("  Figures:")
print("    - conv_maillage.png")
print("    - conv_loglog.png")
print("    - profils_2D.png")
print("    - T_max_evolution.png")
print("    - profil_mediane.png")
print("    - T_face_interne.png")
print("    - sensibilite_k.png")
print("    - sensibilite_L.png")
print("    - sensibilite_qmax.png")
print("\n  Tableaux LaTeX:")
print("    - tab_conv_spatial.tex")
print("    - tab_conv_temporel.tex")
print("\n  Donnees:")
print("    - resultats_TPS.json")

print("\n" + "="*70)
print("SIMULATION TERMINEE")
print("="*70 + "\n")

plt.show()
