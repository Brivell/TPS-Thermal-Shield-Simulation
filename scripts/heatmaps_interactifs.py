"""
Heatmaps interactifs avec Plotly
Visualisations interactives pour exploration résultats
"""
 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os
 
# Ajouter parent directory au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
from tps_fct import simulation_principale, simulation_complete
 
 
def create_interactive_heatmap(save_html=True):
    """
    Crée heatmap interactif avec slider temporel
    """
    print("\nCreation heatmap interactif...")
    
    # Simulation
    class prm:
        rho = 1800.0
        cp = 800.0
        k_therm = 0.5
        epsilon = 0.85
    
    L = 0.1
    X = [0, L]
    Y = [0, L]
    nx = ny = 21
    dt = 10.0
    t_final = 1800.0
    T_initiale = 20.0 + 273.15
    T_stru = 20.0 + 273.15
    t_entree = 1200.0
    q_max = 50000.0
    sigma = 5.67e-8
    
    # Snapshots tous les 100s
    temps_snap = list(range(0, 1801, 100))
    
    print("Simulation...")
    Res = simulation_principale(
        X, Y, nx, ny, dt, t_final, temps_snap,
        T_initiale, T_stru, t_entree, q_max, sigma, prm, verbose=False
    )
    
    Pos_x = Res["Position_x"]
    Pos_y = Res["Position_y"]
    T_snaps = Res["T_instantanes"]
    temps_reels = Res['temps_instantanes']
    
    # Créer figure avec slider
    fig = go.Figure()
    
    # Frames pour chaque instant
    frames = []
    for i, (t_snap, T_snap) in enumerate(zip(temps_reels, T_snaps)):
        T_plot = T_snap - 273.15
        
        frame = go.Frame(
            data=[go.Heatmap(
                x=Pos_x[0, :],
                y=Pos_y[:, 0],
                z=T_plot,
                colorscale='Hot',
                zmin=20,
                zmax=np.max([np.max(T - 273.15) for T in T_snaps]),
                colorbar=dict(title="T [°C]")
            )],
            name=str(i),
            layout=go.Layout(
                title=f"Temperature Distribution - t = {t_snap:.0f} s"
            )
        )
        frames.append(frame)
    
    # Premier frame
    T_plot = T_snaps[0] - 273.15
    fig.add_trace(go.Heatmap(
        x=Pos_x[0, :],
        y=Pos_y[:, 0],
        z=T_plot,
        colorscale='Hot',
        zmin=20,
        zmax=np.max([np.max(T - 273.15) for T in T_snaps]),
        colorbar=dict(title="T [°C]")
    ))
    
    fig.frames = frames
    
    # Slider
    fig.update_layout(
        title="Interactive Temperature Evolution",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 200, "redraw": True},
                                  "fromcurrent": True}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ],
            x=0.1,
            y=0
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name],
                             {"frame": {"duration": 0, "redraw": True},
                              "mode": "immediate"}],
                       method="animate",
                       label=f"{temps_reels[int(f.name)]:.0f}s")
                   for f in frames],
            active=0,
            y=0,
            len=0.9,
            x=0.1
        )]
    )
    
    if save_html:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'interactive')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "interactive_heatmap.html")
        fig.write_html(output_path)
        print(f"Heatmap sauvegardé: {output_path}")
    
    return fig
 
 
def create_3D_surface():
    """
    Crée surface 3D interactive de la température
    """
    print("\nCreation surface 3D...")
    
    # Simulation rapide
    class prm:
        rho = 1800.0
        cp = 800.0
        k_therm = 0.5
        epsilon = 0.85
    
    L = 0.1
    X = [0, L]
    Y = [0, L]
    nx = ny = 21
    dt = 10.0
    t_final = 1800.0
    T_initiale = 20.0 + 273.15
    T_stru = 20.0 + 273.15
    t_entree = 1200.0
    q_max = 50000.0
    sigma = 5.67e-8
    
    temps_snap = [0, 600, 1200, 1800]
    
    Res = simulation_principale(
        X, Y, nx, ny, dt, t_final, temps_snap,
        T_initiale, T_stru, t_entree, q_max, sigma, prm, verbose=False
    )
    
    Pos_x = Res["Position_x"]
    Pos_y = Res["Position_y"]
    T_snaps = Res["T_instantanes"]
    temps_reels = Res['temps_instantanes']
    
    # Créer subplots 3D
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=[f't = {t:.0f} s' for t in temps_reels]
    )
    
    for idx, (t_snap, T_snap) in enumerate(zip(temps_reels, T_snaps)):
        T_plot = T_snap - 273.15
        
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        fig.add_trace(
            go.Surface(
                x=Pos_x[0, :],
                y=Pos_y[:, 0],
                z=T_plot,
                colorscale='Hot',
                showscale=(idx == 0),
                colorbar=dict(title="T [°C]", x=1.1) if idx == 0 else None
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title="3D Temperature Distribution",
        height=800,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(1, 5):
        fig.update_scenes(
            dict(
                xaxis_title="x [m]",
                yaxis_title="y [m]",
                zaxis_title="T [°C]"
            ),
            row=(i-1)//2 + 1,
            col=(i-1)%2 + 1
        )
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'interactive')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "3D_surfaces.html")
    fig.write_html(output_path)
    print(f"Surface 3D sauvegardée: {output_path}")
    
    return fig
 
 
def create_sensitivity_heatmap():
    """
    Heatmap 2D interactif pour analyse sensibilité
    Montre T_max en fonction de (k, L)
    """
    print("\nCreation heatmap sensibilite...")
    
    class prm:
        rho = 1800.0
        cp = 800.0
        k_therm = 0.5
        epsilon = 0.85
    
    L_orig = 0.1
    nx = ny = 21
    dt = 10.0
    t_final = 1800.0
    T_initiale = 20.0 + 273.15
    T_stru = 20.0 + 273.15
    t_entree = 1200.0
    q_max = 50000.0
    sigma = 5.67e-8
    
    # Grille k × L
    k_vals = np.linspace(0.2, 3.0, 15)
    L_vals = np.linspace(0.04, 0.12, 15)
    
    T_max_grid = np.zeros((len(L_vals), len(k_vals)))
    
    print("Calcul grille sensibilité...")
    total = len(k_vals) * len(L_vals)
    count = 0
    
    for i, L in enumerate(L_vals):
        for j, k in enumerate(k_vals):
            count += 1
            print(f"  Progress: {count}/{total}", end='\r')
            
            prm.k_therm = k
            T_max, _, _ = simulation_complete(
                nx, ny, dt, T_initiale, T_stru, t_entree, 
                sigma, t_final, L, q_max, prm
            )
            T_max_grid[i, j] = T_max
    
    print("\nGeneration heatmap...")
    
    # Créer heatmap avec échelle améliorée
    # Clipper valeurs extrêmes pour meilleure visualisation
    T_max_display = np.clip(T_max_grid, 0, 250)
    
    # Échelle de couleurs personnalisée centrée sur 175°C
    colorscale = [
        [0.0, 'darkgreen'],   # 0°C - vert foncé
        [0.4, 'lightgreen'],  # ~100°C - vert clair
        [0.7, 'yellow'],      # 175°C - jaune (limite)
        [0.85, 'orange'],     # ~210°C - orange
        [1.0, 'darkred']      # 250°C+ - rouge foncé
    ]
    
    fig = go.Figure(data=go.Heatmap(
        x=k_vals,
        y=L_vals * 1000,  # Convert to mm
        z=T_max_display,
        colorscale=colorscale,
        colorbar=dict(
            title="T_max [°C]",
            tickvals=[0, 50, 100, 175, 200, 250],
            ticktext=['0', '50', '100', '175 (Limite)', '200', '250+']
        ),
        hovertemplate='k: %{x:.2f} W/(m·K)<br>L: %{y:.1f} mm<br>T_max: %{customdata:.2f}°C<extra></extra>',
        customdata=T_max_grid  # Valeurs réelles pour hover
    ))
    
    # Ligne de contour à 175°C
    fig.add_trace(go.Contour(
        x=k_vals,
        y=L_vals * 1000,
        z=T_max_grid,
        contours=dict(
            start=175,
            end=175,
            size=1,
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        line=dict(width=3),
        name='Safety Limit (175°C)'
    ))
    
    fig.update_layout(
        title="Sensitivity Analysis: T_max vs (k, L)",
        xaxis_title="Thermal Conductivity k [W/(m·K)]",
        yaxis_title="Thickness L [mm]",
        height=600
    )
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'interactive')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sensitivity_heatmap.html")
    fig.write_html(output_path)
    print(f"Heatmap sensibilité sauvegardé: {output_path}")
    
    return fig
 
 
if __name__ == "__main__":
    print("="*70)
    print("GENERATION VISUALISATIONS INTERACTIVES")
    print("="*70)
    
    # Heatmap temporel
    create_interactive_heatmap()
    
    # Surface 3D
    create_3D_surface()
    
    # Sensibilité
    create_sensitivity_heatmap()
    
    print("\n" + "="*70)
    print("VISUALISATIONS TERMINEES")
    print("="*70)
    print("\nOuvrir les fichiers HTML dans navigateur:")
    print("  - results/interactive/interactive_heatmap.html")
    print("  - results/interactive/3D_surfaces.html")
    print("  - results/interactive/sensitivity_heatmap.html")