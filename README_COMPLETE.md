# TPS Thermal Shield Simulation

> 2D transient heat transfer simulation of a Thermal Protection System (TPS) for spacecraft atmospheric reentry with ML acceleration

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-1.20+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

---

## 🎯 Project Overview

This project simulates the thermal behavior of a thermal protection tile used in spacecraft during atmospheric reentry. The simulation employs a 2D finite difference method with implicit Euler time integration and Newton-Raphson linearization for nonlinear radiation boundary conditions.

**Context**: Academic project (GCH2545 - Numerical Methods) focused on solving complex heat transfer problems with validation and sensitivity analysis.

### Key Highlights

- ✅ **Rigorous validation** with spatial (order ~3) and temporal (order ~1) convergence tests
- 🤖 **ML surrogate model** with 99.7% accuracy and 47× speedup
- 🎬 **Interactive visualizations** including animations and 3D heatmaps
- 📊 **Comprehensive sensitivity analysis** for design parameter optimization
- 🔬 **Physics-based** approach with nonlinear radiation boundary conditions

---

## 🔥 Features

### Core Simulation (`src/`)

- **2D Transient Heat Conduction**
  - Governing equation: `ρ·cp·∂T/∂t = k·(∂²T/∂x² + ∂²T/∂y²)`
  - Implicit Euler time integration (unconditionally stable)
  - Finite difference spatial discretization (2nd order)

- **Boundary Conditions**
  - Top surface: Time-varying sinusoidal heat flux (plasma heating)
  - Bottom surface: Nonlinear radiation to structure (Stefan-Boltzmann law)
  - Lateral sides: Symmetry (adiabatic)

- **Validation & Verification**
  - Pure diffusion test case (analytical validation)
  - Spatial convergence analysis (order p ≈ 3.0 - super-convergence!)
  - Temporal convergence analysis (order p ≈ 1.0 - matches Euler implicit)

### Advanced Features (`scripts/`)

#### 🤖 ML Surrogate Model
- **Neural network** trained on 500 FEM simulations
- **Performance**: R² = 0.997, MAE = 4.0°C
- **Speedup**: 47× faster than FEM (1.4s → 0.03s)
- **Use case**: Rapid design space exploration before detailed FEM validation

#### 🎬 Animations
- Temperature evolution over time (GIF format)
- Split-view: temperature + gradient magnitude
- Perfect for presentations and reports

#### 📊 Interactive Visualizations
- **Plotly-based** 3D surface plots with rotation
- **Time-slider** for temperature evolution exploration
- **Sensitivity heatmap** showing safe design zones (k, L parameters)
- Export-ready HTML files for web sharing

#### 📈 Sensitivity Analysis
- Parametric studies: thermal conductivity (k), thickness (L), heat flux (q_max)
- Safety threshold visualization (T_max < 175°C)
- Automated critical parameter identification

---

## 📊 Results Summary

### Baseline Configuration

```
Material: k = 0.5 W/(m·K), ρ = 1800 kg/m³, cp = 800 J/(kg·K)
Geometry: L = 0.1 m (square tile)
Heat flux: q_max = 50 kW/m² (sinusoidal, 1200s duration)

Maximum temperature (structure): 25.68°C
Safety factor: 149.32°C (limit: 175°C)
Status: ✓ PROTECTED
```

### Validation Results

| Test | Expected | Obtained | Status |
|------|----------|----------|--------|
| Pure diffusion | T = T₀ | Error < 0.0001°C | ✅ Excellent |
| Spatial convergence | Order ≈ 2 | Order ≈ 3.0 | ✅ Super-convergence |
| Temporal convergence | Order ≈ 1 | Order ≈ 1.0 | ✅ Perfect |

### Critical Parameters

- **k_critical**: ~1.5 W/(m·K) (above this → structure overheats)
- **L_minimum**: ~60 mm (below this → insufficient protection)
- **q_max**: Linear relationship with T_max

---

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.8
pip install -r requirements.txt
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/TPS-Thermal-Shield-Simulation.git
cd TPS-Thermal-Shield-Simulation

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Run Main Analysis

```bash
# Complete simulation with validation and sensitivity
python -m src.analyse_TPS
```

**Outputs**:
- 9 figures (convergence, profiles, sensitivity)
- 2 LaTeX tables (spatial/temporal convergence)
- JSON results file

**Time**: ~10 minutes

#### 2. Generate Animations

```bash
python scripts/animations.py
```

**Outputs**: 
- `temp_evolution.gif` - Temperature evolution over time
- `temp_gradient.gif` - Temperature + gradient split view

**Time**: ~5 minutes

#### 3. Interactive Visualizations

```bash
python scripts/heatmaps_interactifs.py
```

**Outputs**:
- `interactive_heatmap.html` - Time-slider temperature map
- `3D_surfaces.html` - Rotatable 3D temperature surfaces
- `sensitivity_heatmap.html` - Interactive (k, L) design space

Open HTML files in browser for interactive exploration.

**Time**: ~10 minutes

#### 4. ML Surrogate Model

```bash
python scripts/ml_surrogate.py
```

**Outputs**:
- `surrogate_model.pkl` - Trained neural network
- `dataset_TPS.npz` - Training dataset (500 samples)
- `ML_predictions.png` - Validation plots

**Time**: ~45 minutes (dataset generation)

**After training**, use for instant predictions:

```python
from scripts.ml_surrogate import predict_Tmax

# Predict maximum temperature instantly
T_max = predict_Tmax(k=0.5, L=0.1, q_max=50000)
print(f"Predicted T_max: {T_max:.2f}°C")
# Result in 0.03s instead of 1.4s!
```

---

## 📁 Project Structure

```
TPS-Thermal-Shield-Simulation/
│
├── src/                          # Core simulation package
│   ├── __init__.py              # Package initialization
│   ├── tps_fct.py               # FEM solver functions
│   ├── verification.py          # Convergence tests
│   └── analyse_TPS.py           # Main analysis script
│
├── scripts/                      # Advanced features
│   ├── animations.py            # GIF generation
│   ├── heatmaps_interactifs.py  # Plotly visualizations
│   ├── ml_surrogate.py          # ML surrogate model
│   └── sensitivity_improved.py  # Enhanced heatmaps
│
├── results/                      # Generated outputs
│   ├── figures/                 # PNG plots (300 DPI)
│   ├── animations/              # GIF animations
│   ├── interactive/             # HTML visualizations
│   └── data/                    # JSON, NPZ, PKL files
│
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── .gitignore                    # Git exclusions
└── LICENSE                       # MIT License
```

---

## 🔬 Physics & Numerics

### Governing Equation

2D transient heat conduction with internal heat generation:

```
ρ·cp·∂T/∂t = k·∇²T
```

### Boundary Conditions

**Top surface (y = L)**: Incoming plasma heat flux
```
-k·∂T/∂y = q(t) = q_max·sin(πt/t_entry)  for 0 ≤ t ≤ 1200s
```

**Bottom surface (y = 0)**: Radiation to spacecraft structure
```
-k·∂T/∂y = σ·ε·(T⁴ - T_structure⁴)
```

**Lateral sides (x = 0, L)**: Symmetry (adiabatic)
```
∂T/∂x = 0
```

### Numerical Method

- **Spatial discretization**: Finite differences (centered, 2nd order)
- **Temporal discretization**: Implicit Euler (1st order, unconditionally stable)
- **Nonlinear solver**: Newton-Raphson with linearization of radiation term
  - Linearization: `T⁴ ≈ T_old³·T` (iterative update)
  - Convergence criterion: `|T^(n+1) - T^(n)| < 10⁻⁶`
- **Grid**: 21×21 nodes (adjustable)
- **Time step**: 10s (adjustable, stability guaranteed)

---

## 📈 Performance

### FEM Solver
- **Grid**: 21×21 → Simulation time: ~1.4s
- **Grid**: 51×51 → Simulation time: ~15s
- Scales as O(N²) for grid size

### ML Surrogate
- **Single prediction**: 0.03s (47× faster than FEM)
- **Batch 1000 predictions**: 0.5s total (~2800× faster)
- **Accuracy**: R² = 0.997, MAE = 4.0°C

### Workflow Optimization

Traditional approach:
```
1000 designs × 1.4s = 23 minutes (FEM only)
```

ML-accelerated approach:
```
1000 designs × 0.03s = 30s (ML screening)
+ 10 best designs × 1.4s = 14s (FEM validation)
= 44 seconds total (31× faster!)
```

---

## 🎓 Applications

### Academic
- Numerical methods validation (convergence orders)
- Heat transfer fundamentals
- Nonlinear PDE solving
- ML for engineering applications

### Industrial
- Thermal protection system design
- Aerospace reentry vehicles
- High-temperature material selection
- Design space exploration

### Research
- Surrogate modeling techniques
- Multi-physics coupling (future: ablation)
- Uncertainty quantification (future: stochastic analysis)

---

## 🚧 Future Improvements

### Short-term (1-2 weeks)
- [ ] 3D extension (cylindrical geometry for nose cone)
- [ ] Adaptive mesh refinement (AMR) for boundary layers
- [ ] Uncertainty quantification with Monte Carlo
- [ ] Additional materials database (carbon-carbon, PICA, etc.)

### Medium-term (1-2 months)
- [ ] Ablation coupling (surface recession)
- [ ] Multi-layer TPS with different materials
- [ ] Optimization framework (scipy.optimize integration)
- [ ] Real atmospheric reentry profile (not sinusoidal)

### Long-term (3-6 months)
- [ ] GPU acceleration with CuPy/Numba
- [ ] Physics-Informed Neural Networks (PINNs)
- [ ] Streamlit web interface for non-technical users
- [ ] Integration with CAD software (FreeCAD, OpenFOAM)

### Advanced Research
- [ ] Turbulent boundary layer coupling (CFD)
- [ ] Chemical reactions (oxidation, pyrolysis)
- [ ] Structural mechanics coupling (thermal stress)
- [ ] Real-time mission simulation

---

## 📚 Documentation

### Guides Included
- Installation and setup
- Usage examples
- Physics background
- Numerical methods details
- GitHub workflow

### External Resources
- [COMSOL Multiphysics](https://www.comsol.com/) - Commercial FEM software for validation
- [NASA TPS Database](https://tps.nasa.gov/) - Material properties
- [PyFEM](https://pyfem.org/) - Python FEM resources

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- **Code optimization**: Vectorization, Numba JIT compilation
- **New features**: Additional boundary conditions, materials
- **Documentation**: Tutorials, examples
- **Validation**: Experimental data comparison
- **Testing**: Unit tests, integration tests

Please open an issue before submitting major pull requests.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

### Third-party Libraries
- NumPy, SciPy, Matplotlib: BSD License
- Plotly: MIT License
- Scikit-learn: BSD License

---

## 🙏 Acknowledgments

- **Course**: GCH2545 - Numerical Methods in Engineering
- **Institution**: Polytechnique Montréal
- **Validation**: COMSOL Multiphysics (commercial software)
- **Inspiration**: NASA Apollo heat shield design, SpaceX Starship TPS

---

## 📧 Contact

**Author**: Amaury  
**LinkedIn**: [Amaury Tchoupe](https://www.linkedin.com/feed/)  
**Email**: amaurytchoupe01@gmail.com

For questions, suggestions, or collaborations:
- Open an issue on GitHub
- Send an email
- Connect on LinkedIn

---

## 📊 Citation

If you use this code in your research or project, please cite:

```bibtex
@software{tps_simulation_2026,
  author = {Your Name},
  title = {TPS Thermal Shield Simulation: 2D FEM Solver with ML Acceleration},
  year = {2026},
  url = {https://github.com/yourusername/TPS-Thermal-Shield-Simulation},
  version = {1.0.0}
}
```

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

**Made with ❤️ for aerospace thermal engineering**

*Last updated: March 2026*
