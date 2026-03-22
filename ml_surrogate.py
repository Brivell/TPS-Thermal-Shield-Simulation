"""
ML Surrogate Model pour TPS
Prédit T_max instantanément sans simulation FEM complète

Input: (k, L, q_max)
Output: T_max structure
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import json
from tps_fct import simulation_complete


def generate_dataset(n_samples=500, save_file='dataset_TPS.npz'):
    """
    Génère dataset en variant k, L, q_max
    
    Args:
        n_samples: nombre échantillons
        save_file: fichier sauvegarde
        
    Returns:
        X: inputs (n_samples, 3)
        y: outputs (n_samples,)
    """
    print(f"\nGeneration dataset ({n_samples} samples)...")
    
    # Params simulation
    class prm:
        rho = 1800.0
        cp = 800.0
        k_therm = 0.5
        epsilon = 0.85
    
    nx = ny = 21
    dt = 10.0
    t_final = 1800.0
    T_initiale = 20.0 + 273.15
    T_stru = 20.0 + 273.15
    t_entree = 1200.0
    sigma = 5.67e-8
    
    # Ranges paramètres
    k_range = (0.2, 5.0)      # W/(m.K)
    L_range = (0.04, 0.12)    # m
    q_range = (30000, 80000)  # W/m2
    
    # Génération aléatoire
    np.random.seed(42)
    
    k_samples = np.random.uniform(*k_range, n_samples)
    L_samples = np.random.uniform(*L_range, n_samples)
    q_samples = np.random.uniform(*q_range, n_samples)
    
    X = np.column_stack([k_samples, L_samples, q_samples])
    y = np.zeros(n_samples)
    
    # Simulation pour chaque sample
    for i in range(n_samples):
        if (i+1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
        
        k, L, q_max = X[i]
        prm.k_therm = k
        
        T_max, _, _ = simulation_complete(
            nx, ny, dt, T_initiale, T_stru, t_entree,
            sigma, t_final, L, q_max, prm
        )
        
        y[i] = T_max
    
    # Sauvegarder
    np.savez(save_file, X=X, y=y)
    print(f"\nDataset sauvegarde: {save_file}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}] C")
    
    return X, y


def train_surrogate(X, y, save_model='surrogate_model.pkl'):
    """
    Entraîne modèle MLP
    
    Args:
        X: inputs (n, 3)
        y: outputs (n,)
        save_model: fichier sauvegarde modèle
        
    Returns:
        model: modèle entraîné
        scaler_X, scaler_y: scalers
        metrics: dict métriques
    """
    print("\nEntrainement surrogate model...")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalisation
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # Modèle MLP
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    print("  Entrainement en cours...")
    model.fit(X_train_scaled, y_train_scaled)
    
    # Prédictions
    y_train_pred_scaled = model.predict(X_train_scaled)
    y_test_pred_scaled = model.predict(X_test_scaled)
    
    # Denormalisation
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    
    # Métriques
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    metrics = {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae_train': mae_train,
        'mae_test': mae_test
    }
    
    print(f"\n  Resultats:")
    print(f"    R² train: {r2_train:.4f}")
    print(f"    R² test:  {r2_test:.4f}")
    print(f"    MAE train: {mae_train:.3f} C")
    print(f"    MAE test:  {mae_test:.3f} C")
    
    # Sauvegarder
    with open(save_model, 'wb') as f:
        pickle.dump((model, scaler_X, scaler_y), f)
    print(f"\n  Modele sauvegarde: {save_model}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train
    ax1.scatter(y_train, y_train_pred, alpha=0.5, s=20)
    ax1.plot([y_train.min(), y_train.max()],
             [y_train.min(), y_train.max()], 'r--', lw=2)
    ax1.set_xlabel('T_max FEM [C]', fontsize=12)
    ax1.set_ylabel('T_max ML [C]', fontsize=12)
    ax1.set_title(f'Train Set (R²={r2_train:.4f})', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Test
    ax2.scatter(y_test, y_test_pred, alpha=0.5, s=20, color='orange')
    ax2.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('T_max FEM [C]', fontsize=12)
    ax2.set_ylabel('T_max ML [C]', fontsize=12)
    ax2.set_title(f'Test Set (R²={r2_test:.4f})', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ML_predictions.png', dpi=300)
    print("  Figure: ML_predictions.png")
    
    return model, scaler_X, scaler_y, metrics


def predict_Tmax(k, L, q_max, model_file='surrogate_model.pkl'):
    """
    Prédit T_max avec modèle entraîné
    
    Args:
        k: conductivité [W/(m.K)]
        L: épaisseur [m]
        q_max: flux max [W/m2]
        model_file: fichier modèle
        
    Returns:
        T_max: température max prédite [C]
    """
    # Charger modèle
    with open(model_file, 'rb') as f:
        model, scaler_X, scaler_y = pickle.load(f)
    
    # Préparer input
    X_new = np.array([[k, L, q_max]])
    X_scaled = scaler_X.transform(X_new)
    
    # Prédire
    y_scaled = model.predict(X_scaled)
    T_max = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[0, 0]
    
    return T_max


def benchmark_speed():
    """
    Compare vitesse FEM vs ML
    """
    print("\nBenchmark vitesse FEM vs ML...")
    
    import time
    from tps_fct import simulation_complete
    
    # Params test
    class prm:
        rho = 1800.0
        cp = 800.0
        k_therm = 0.5
        epsilon = 0.85
    
    k, L, q_max = 0.5, 0.1, 50000
    
    # FEM
    start = time.time()
    T_max_fem, _, _ = simulation_complete(
        21, 21, 10.0, 20.0 + 273.15, 20.0 + 273.15,
        1200.0, 5.67e-8, 1800.0, L, q_max, prm
    )
    time_fem = time.time() - start
    
    # ML
    start = time.time()
    T_max_ml = predict_Tmax(k, L, q_max)
    time_ml = time.time() - start
    
    print(f"\n  FEM:")
    print(f"    T_max: {T_max_fem:.2f} C")
    print(f"    Temps: {time_fem:.3f} s")
    
    print(f"\n  ML:")
    print(f"    T_max: {T_max_ml:.2f} C")
    print(f"    Temps: {time_ml:.6f} s")
    
    speedup = time_fem / time_ml
    error = abs(T_max_ml - T_max_fem)
    
    print(f"\n  Speedup: {speedup:.0f}x")
    print(f"  Erreur: {error:.3f} C ({error/T_max_fem*100:.2f}%)")


if __name__ == "__main__":
    print("="*70)
    print("ML SURROGATE MODEL POUR TPS")
    print("="*70)
    
    # Étape 1: Générer dataset
    print("\n[1/3] Generation dataset...")
    X, y = generate_dataset(n_samples=500)
    
    # Étape 2: Entraîner modèle
    print("\n[2/3] Entrainement modele...")
    model, scaler_X, scaler_y, metrics = train_surrogate(X, y)
    
    # Étape 3: Benchmark
    print("\n[3/3] Benchmark vitesse...")
    benchmark_speed()
    
    print("\n" + "="*70)
    print("ML SURROGATE MODEL TERMINE")
    print("="*70)
    print("\nUtilisation:")
    print("  from ml_surrogate import predict_Tmax")
    print("  T_max = predict_Tmax(k=0.5, L=0.1, q_max=50000)")
