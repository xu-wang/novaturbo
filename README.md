<p align="center">
  <img src="repo_thumbnail.png" alt="NovaTurbo AI — Micro Turbojet Engine Viewer" width="800"/>
</p>

<h1 align="center">NovaTurbo AI</h1>

<p align="center">
  <b>AI-Powered Micro Turbojet Engine Design System</b><br/>
  Inspired by <a href="https://leap71.com">LEAP 71's Noyron</a> — computational engineering for autonomous engine design
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#3d-viewer">3D Viewer</a> •
  <a href="#contributing">Contributing</a> •
  <a href="https://discord.gg/SQDBr8Mt8">Discord</a> •
  <a href="LICENSE">License</a>
</p>

---

## What is NovaTurbo?

NovaTurbo AI is an open-source computational engineering prototype for **micro turbojet engine design**. Given target specifications such as thrust, size, and weight, it explores parameterized engine designs and generates geometries for visualization and further validation.

The system combines **parametric geometry generation**, **Brayton-cycle thermodynamics**, a **PyTorch neural-network surrogate model**, **NSGA-II multi-objective optimization**, and a rich **Three.js web UI** with real-time simulation visualizations.

### V3 22D development status

V3 implementation is paused at code/acceptance baseline `ea1cfc5` while its documentation is consolidated. The explicit development path connects a 22D physics backend, a 55-class target gate, local conditional CFM, D-Flow, continuous refinement, independent recomputation, and web target jobs. New V3 web jobs use `v3_guided_cfm_development`; entering V3 clears the old targets and waits for new input and an explicit run action. The CLI default remains V2.1; selecting V3 without a profile uses the original fixture.

The latest 98-target development check found 95 C hits and 3 natural D activations. D, supplementary E evaluation, and matched sequential evaluation rescued none of those three `NO_FLOW_MATCH` cases. E has no evolutionary search, the 3D view is a 19D projection, and no whole-engine feasible delivery or independent physics validation has been demonstrated in these checks. Formal use and deployment remain disabled.

See [V3 current status, usage, and document index](docs/physics/V3_CURRENT_STATUS.md) and [the expanded check](docs/physics/V3_REFINEMENT_EXPANDED_CHECK_20260927.md) for commands, model/data assets, evidence, and remaining work.

### Default V2.1 inverse-design delivery policy

The current Geometry19 target-directed pipeline uses the bounded 19D CFM-4 and the three-output `performance3_log` surrogate by default:

```text
target thrust / TSFC / mass
→ bounded Geometry19 CFM-4 generates 256 candidates
→ performance3_log guides D-Flow and ranks candidates by predicted hit-aligned rho
→ nearest-integer projection and deterministic Geometry19 validation
→ V2.1 evaluates the retained power-of-two candidate set
→ surrogate-gradient refinement when no true hit exists
→ warm-start NSGA-II when refinement still has no hit
→ V2.1 recalculation, preview mesh, and UI delivery
```

The performance surrogate predicts only thrust, TSFC, and mass; V2.1 remains the authority for legality, constraints, and final hit status. The previous shared 48-output S0 surrogate remains available through `--forward-model-tag s0`. The public Geometry19 entry is `scripts/run_wp19_23_pipeline.py`; see `docs/physics/GEOMETRY19_CURRENT_PIPELINE.md` for the current system overview, `docs/physics/WP19_23_PIPELINE_ENTRYPOINT.md` for implementation rules, and `docs/physics/WP19_23_PERFORMANCE3_FULL_PIPELINE_COMPARISON_20260908.md` for the surrogate comparison.

The earlier active-10D **CFM-4 + Diag-CFM** workflow remains reproducible through `scripts/run_inverse_design.py` and is retained as a historical/parallel baseline. Its evidence and cost accounting remain in `完整高精度逆设计默认策略说明.md`; it is not the Geometry19 default described above.

> 🚀 **We're looking for contributors!** Especially in aerospace propulsion physics, CFD, and additive manufacturing. See [Contributing](#contributing).
>
> 💬 **Join our Discord:** [discord.gg/SQDBr8Mt8](https://discord.gg/SQDBr8Mt8)

## Features

| Feature | Description |
|---|---|
| 🔧 **Parametric Geometry** | 5 engine components (inlet, compressor, combustor, turbine, nozzle) with full dimensional control |
| 🌡️ **Brayton Cycle Solver** | Station-by-station thermodynamic analysis (ambient → nozzle exit) |
| 🧠 **Neural Surrogate** | MLP (144K params) trained on 10K+ design variants for instant performance prediction |
| 📊 **NSGA-II Optimizer** | Multi-objective optimization (thrust vs. weight vs. TSFC) with Pareto front |
| 🏗️ **TPMS Lattice** | Internal gyroid/Schwarz-P/diamond lattice structures via slab-warp technique |
| 🛡️ **Thermal Barrier Coatings** | Bio-inspired + conventional TBC analysis with 1-D heat transfer model (7 coatings) |
| ✈️ **Professional Blade Geometry** | Lofted NACA airfoil turbine blades & curved centrifugal compressor impellers |
| 🔥 **Flame Simulation** | FumeFX-style combustion particle system (3500 particles, spiral turbulence) |
| 🌈 **Thermal/Airflow/Stress** | Color-mapped simulation overlays on 3D engine geometry |
| 📐 **Engineering Dashboard** | Live parameter sliders, Brayton cycle charts, Pareto front, TBC analysis |
| 🔄 **Closed-Loop Training** | Save design variants from UI → retrain surrogate model in real-time |
| 🎯 **Inverse Design** | Specify target thrust/TSFC → AI suggests optimal geometry parameters |
| 🔬 **CFD Calibration** | Optional OpenFOAM/SU2 integration for physics-calibrated training labels |
| 📦 **STL Export** | Multi-variant STL export for Fusion 360 / 3D printing workflows |

## Quick Start

### Prerequisites
- Python 3.10+
- ~2 GB disk (for generated datasets)
- GPU optional (CPU works for inference & small training runs)

### Installation

```bash
git clone https://github.com/bxf1001g/novaturbo.git
cd novaturbo
pip install -r requirements.txt
```

### Generate Design Dataset

```bash
python app.py --generate 10000    # Generate 10K design variants (CPU: ~30 min)
```

### Train the Surrogate Model

```bash
python app.py --train data/generated/dataset_10000.csv
```

### Launch the 3D Viewer

```bash
python app.py --ui
# Open http://localhost:5000 in your browser
```

In the Geometry19 workspace, switch **Physics model** to **V2.2 new design**.
The existing model and metrics clear. Enter target thrust, TSFC, and mass, then
click **Generate V2.2 design**. The UI starts a new background GPU pipeline
run in an isolated directory and displays only that run's verified result.
Use Preview or Export to choose which delivered mesh to display.

The historical read-only completed-result endpoint remains available for
inspection; it does not control the new-design UI mode. To configure that
endpoint separately, set:

```bash
export NOVATURBO_GEOMETRY19_PIPELINE_RESULT_ROOT="$PWD/data/validation/v2_2_pipeline_mesh_delivery_acceptance_v1"
python app.py --ui
```

Both the new-job result and the historical read-only endpoint verify the
pipeline manifest, mesh manifest, and every STL hash before serving components.

### Run Physics Analysis

```bash
python app.py --brayton     # Brayton cycle analysis + material validation
python app.py --geometry    # Engine geometry summary
```

### Run the Geometry19 Inverse-Design Pipeline

The public entry accepts thrust, TSFC, and mass. V2.1 remains the default model:

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n inverse_design_nova python \
  scripts/run_wp19_23_pipeline.py \
  --thrust-N 60.0 --tsfc-kg-N-s 0.000067 --mass-kg 1.77 \
  --device cuda:0
```

Select the development-only V2.2 local flow explicitly. Add `--export-mesh` to
publish both preview and high-resolution component STL files; use
`--storage-root` to select their hash-isolated design store:

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n inverse_design_nova python \
  scripts/run_wp19_23_pipeline.py --model "model V2.2" \
  --thrust-N 60.0 --tsfc-kg-N-s 0.000067 --mass-kg 1.77 \
  --device cuda:0 --export-mesh
```

V2.2 remains `formal_use_allowed=false`. Mesh export does not change the
physics result or authorize manufacturing use. The earlier active-10D entry
`scripts/run_inverse_design.py` remains available for frozen historical
reproduction but is not the Geometry19 public entry.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NovaTurbo AI Pipeline                    │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Parametric  │   Physics    │   AI Engine  │  Manufacturing │
│   Geometry   │   Solver     │              │    Output      │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ • Inlet      │ • Brayton    │ • Surrogate  │ • STL Export   │
│ • Compressor │   Cycle      │   (PyTorch)  │ • CFD Case     │
│ • Combustor  │ • Station    │ • NSGA-II    │ • Lattice      │
│ • Turbine    │   Analysis   │   Optimizer  │   Infill       │
│ • Nozzle     │ • CFD Bridge │ • Inverse    │ • Build Prep   │
│              │              │   Design     │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
                              ↕
                    ┌──────────────────┐
                    │   Web UI (Three.js)   │
                    │ • 3D Viewer      │
                    │ • Simulations    │
                    │ • Dashboard      │
                    │ • Flame FX       │
                    └──────────────────┘
```

## Project Structure

```
novaturbo/
├── app.py                    # Main entry point (CLI)
├── requirements.txt          # Python dependencies
├── config/                   # Engine parameters, materials, constraints
├── src/
│   ├── geometry/             # Parametric engine component generators
│   │   └── lattice.py        # TPMS lattice (gyroid, Schwarz-P, diamond)
│   ├── physics/              # Thermodynamic & fluid dynamics solvers
│   │   ├── brayton.py        # Brayton cycle station analysis
│   │   ├── materials.py      # Material database & TBC coating analysis
│   │   ├── blade_analysis.py # XFLR5-style blade profile analysis
│   │   ├── simulation.py     # Thermal/flow/stress simulation engine
│   │   ├── validation.py     # Brayton validation vs real engines
│   │   └── cfd_calibration.py # OpenFOAM/SU2 calibration bridge
│   ├── ai/                   # Neural network surrogate & optimizer
│   │   ├── surrogate.py      # MLP surrogate model (PyTorch)
│   │   └── optimizer.py      # NSGA-II multi-objective optimizer
│   ├── export/               # STL export & manufacturing prep
│   └── visualization/        # Matplotlib plots & performance charts
├── ui/
│   ├── server.py             # Flask backend with REST API
│   ├── templates/viewer.html # Main 3D viewer page
│   └── static/
│       ├── js/viewer.js      # Three.js 3D engine (simulations, flame FX)
│       ├── js/dashboard.js   # Engineering dashboard logic
│       └── css/viewer.css    # UI styling
├── data/                     # Generated datasets & trained models
├── exports/                  # Output STL files
├── tests/                    # Test suite
└── notebooks/                # Jupyter exploration notebooks
```

## 3D Viewer

The web-based viewer provides:

- **Component inspector** — Click to inspect individual parts (inlet, compressor, combustor, turbine, nozzle, shaft, casing)
- **Thermal simulation** — Temperature-mapped heatmap overlay
- **Airflow simulation** — Velocity streamlines with color coding
- **Stress simulation** — Von Mises stress distribution
- **Flame simulation** — FumeFX-style combustion particles with real physics temps
- **Lattice view** — Toggle TPMS internal structure (gyroid/Schwarz-P/diamond variants)
- **Dashboard** — Adjust parameters live, view Brayton cycle charts, run inverse design
- **TBC Analysis** — Compare bio-inspired & conventional thermal barrier coatings side-by-side
- **Section plane** — Adjustable cross-section slider
- **STL export** — Screenshot & export current design

## CFD Calibration (Optional)

NovaTurbo can optionally calibrate its fast surrogate model against high-fidelity CFD results:

```bash
# Set up OpenFOAM command template
export NOVATURBO_OPENFOAM_CMD_TEMPLATE="simpleFoam -case /tmp/novaturbo_{case_id}"

# Or SU2
export NOVATURBO_SU2_CMD_TEMPLATE="SU2_CFD /tmp/novaturbo_{case_id}.cfg"
```

Then use the **Dashboard → Run CFD Calibration** button, or enable "Use CFD labels" when training.

## Contributing

**We actively welcome contributions!** This is an ambitious project and we need help from people with expertise in:

💬 **Join the discussion on Discord:** [discord.gg/SQDBr8Mt8](https://discord.gg/SQDBr8Mt8)

- 🚀 **Aerospace Propulsion** — Combustion physics, turbomachinery aerodynamics, nozzle design
- 🌊 **CFD / Fluid Dynamics** — OpenFOAM/SU2 case setup, mesh generation, validation
- 🔬 **Materials Science** — High-temp alloys, thermal barrier coatings, bio-inspired materials, additive manufacturing
- 🧠 **Machine Learning** — Physics-informed neural networks, surrogate model improvements
- 🏗️ **CAD/CAM** — STEP export, build orientation optimization, support structure generation
- 🎨 **3D Visualization** — Three.js, WebGL, advanced rendering techniques

### How to Contribute

1. **Fork** the repository
2. **Create a branch** (`git checkout -b feature/your-feature`)
3. **Make your changes** and add tests
4. **Submit a Pull Request** with a clear description

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Ideas for First Contributions

- [ ] Integrate TBC-adjusted wall temps into 3D thermal heatmap
- [ ] Push TIT to 1400K with TBC and show thrust/efficiency gains
- [ ] Add physics-informed loss function to surrogate training
- [ ] Implement ensemble model with uncertainty quantification
- [ ] Add active learning (auto-sample where model uncertainty is highest)
- [ ] Cross-section slicer with thermal/stress overlay
- [ ] Design comparison mode (side-by-side)
- [ ] Map CFD results back to 3D viewer as ground-truth heatmaps
- [ ] STEP file export for manufacturing
- [ ] Make flame field CFD-driven (temperature/species/velocity per voxel)
- [ ] Add more engine topologies (axial compressor, afterburner)
- [ ] Real material property databases (temp-dependent Cp, k, σ_yield)

## Thermal Barrier Coatings (TBC)

NovaTurbo includes a **1-D steady-state heat transfer model** for thermal barrier coatings with 7 built-in materials:

| Coating | Type | k (W/mK) | Mass | Temp Drop |
|---|---|---|---|---|
| 🦎 Diatomite-Silica | Bio-inspired | 0.06 | +5.7g | ~403K avg |
| 🔬 Prismatic Chitin | Bio-inspired | 0.03 | +4.0g | ~407K avg |
| 🐚 Nacre-Layered | Bio-inspired | 0.80 | +7.6g | ~273K avg |
| 🌿 Ceramic Aerogel | Bio-inspired | 0.015 | +1.2g | ~410K avg |
| 🏭 YSZ Standard | Conventional | 2.00 | +50g | ~365K avg |
| 🏭 YSZ EB-PVD | Conventional | 1.50 | +38g | ~378K avg |
| 🏭 Gadolinium Zirconate | Conventional | 1.60 | +45g | ~374K avg |

Bio-inspired coatings are **10× lighter** than conventional YSZ while providing **better insulation** — enabling higher turbine inlet temperatures (TIT) for more thrust, or dramatically extended blade life at current temps.

Use the **Dashboard → 🛡️ TBC Analysis** panel to analyze and compare coatings interactively.

## Blade Geometry

Engine blades use **professional lofted airfoil geometry**:

- **Turbine NGV & Rotor** — NACA-profile airfoils lofted across 7 span sections, wrapped onto cylindrical surfaces with proper twist and taper
- **Compressor Impeller** — Curved radial blades with thickness distribution perpendicular to camberline, plus splitter blades at half-pitch offset
- **Triangle-strip lofting** with end caps for watertight STL meshes (96K vertices, 185K faces)

## Engine Specifications (Default)

| Parameter | Value |
|---|---|
| Type | Single-spool micro turbojet |
| Compressor | Centrifugal, single-stage |
| Combustor | Annular |
| Turbine | Single-stage axial |
| Nozzle | Convergent |
| Thrust | ~76 N (7.8 kgf) |
| Max Diameter | 120 mm |
| Total Length | ~232 mm |
| Weight | ~1.87 kg |
| RPM | 100,000 |
| Pressure Ratio | 3.5 |
| TIT | 1100 K |
| TSFC | ~37 g/kN·s |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [LEAP 71's Noyron](https://leap71.com) — autonomous computational engineering for rocket engines
- Built with [Three.js](https://threejs.org/), [PyTorch](https://pytorch.org/), [Flask](https://flask.palletsprojects.com/), [Chart.js](https://www.chartjs.org/)
- TPMS lattice algorithms based on triply periodic minimal surface research
