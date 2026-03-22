"""
Script pour générer animations temporelles
Crée des animations de l'évolution de température
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from tps_fct import simulation_principale

def create_2D_animation(filename='temperature_evolution.gif', fps=10):
    """
    Crée animation de l'évolution température 2D
    
    Args:
        filename: nom fichier sortie (gif ou mp4)
        fps: images par seconde
    """
    print("\nCreation animation 2D...")
    
    # Params simulation
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
    
    # Sauvegarder tous les 30 pas (300s)
    temps_snap = list(range(0, 1801, 300))
    
    print("Simulation en cours...")
    Res = simulation_principale(
        X, Y, nx, ny, dt, t_final, temps_snap,
        T_initiale, T_stru, t_entree, q_max, sigma, prm, verbose=True
    )
    
    Pos_x = Res["Position_x"]
    Pos_y = Res["Position_y"]
    T_snaps = Res["T_instantanes"]
    temps_reels = Res['temps_instantanes']
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Limites temperature pour echelle fixe
    T_min = 20
    T_max = np.max([np.max(T - 273.15) for T in T_snaps])
    
    # Premier frame
    T_plot = T_snaps[0] - 273.15
    contour = ax.contourf(Pos_x, Pos_y, T_plot, levels=20, 
                          cmap='hot', vmin=T_min, vmax=T_max)
    
    cbar = plt.colorbar(contour, ax=ax, label='Temperature [°C]')
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_aspect('equal')
    
    title = ax.set_title(f't = {temps_reels[0]:.0f} s | T_max = {T_max:.1f} °C', 
                         fontsize=13)
    
    def update(frame):
        """Update function pour animation"""
        ax.clear()
        
        T_plot = T_snaps[frame] - 273.15
        T_current_max = np.max(T_plot)
        
        contour = ax.contourf(Pos_x, Pos_y, T_plot, levels=20,
                              cmap='hot', vmin=T_min, vmax=T_max)
        
        ax.set_xlabel('x [m]', fontsize=12)
        ax.set_ylabel('y [m]', fontsize=12)
        ax.set_title(f't = {temps_reels[frame]:.0f} s | T_max = {T_current_max:.1f} °C',
                     fontsize=13)
        ax.set_aspect('equal')
        
        return contour,
    
    # Créer animation
    print("Génération animation...")
    anim = FuncAnimation(fig, update, frames=len(T_snaps),
                        interval=1000/fps, blit=False, repeat=True)
    
    # Sauvegarder
    if filename.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print(f"Animation sauvegardée: {filename}")
    elif filename.endswith('.mp4'):
        # Requires ffmpeg
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(filename, writer=writer)
        print(f"Animation sauvegardée: {filename}")
    
    plt.close()
    return anim


def create_split_view_animation(filename='split_view.gif', fps=5):
    """
    Animation avec vue coupée: température + gradient
    """
    print("\nCreation animation split view...")
    
    # Setup similaire
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
    
    temps_snap = list(range(0, 1801, 300))
    
    print("Simulation...")
    Res = simulation_principale(
        X, Y, nx, ny, dt, t_final, temps_snap,
        T_initiale, T_stru, t_entree, q_max, sigma, prm, verbose=False
    )
    
    Pos_x = Res["Position_x"]
    Pos_y = Res["Position_y"]
    T_snaps = Res["T_instantanes"]
    temps_reels = Res['temps_instantanes']
    
    # Figure avec 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    T_min = 20
    T_max = np.max([np.max(T - 273.15) for T in T_snaps])
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        T_plot = T_snaps[frame] - 273.15
        
        # Left: Temperature
        c1 = ax1.contourf(Pos_x, Pos_y, T_plot, levels=20,
                         cmap='hot', vmin=T_min, vmax=T_max)
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.set_title(f'Temperature [°C] - t = {temps_reels[frame]:.0f} s')
        ax1.set_aspect('equal')
        
        # Right: Gradient magnitude
        dy = (Y[1] - Y[0]) / (ny - 1)
        dx = (X[1] - X[0]) / (nx - 1)
        grad_y, grad_x = np.gradient(T_plot, dy, dx)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        c2 = ax2.contourf(Pos_x, Pos_y, grad_mag, levels=20, cmap='viridis')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('y [m]')
        ax2.set_title(f'Temperature Gradient [°C/m]')
        ax2.set_aspect('equal')
        
        return c1, c2
    
    anim = FuncAnimation(fig, update, frames=len(T_snaps),
                        interval=1000/fps, blit=False)
    
    if filename.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print(f"Animation sauvegardée: {filename}")
    
    plt.close()
    return anim


if __name__ == "__main__":
    print("="*70)
    print("GENERATION ANIMATIONS")
    print("="*70)
    
    # Animation simple
    create_2D_animation('temp_evolution.gif', fps=5)
    
    # Animation split view
    create_split_view_animation('temp_gradient.gif', fps=5)
    
    print("\n" + "="*70)
    print("ANIMATIONS TERMINEES")
    print("="*70)
