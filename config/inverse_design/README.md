# Stable inverse-design configuration

The public inverse-design entry is assembled from two explicit inputs:

- `default.yaml`: CFM-4/Diag-CFM identities, candidate protocol, runtime, and optional post-processing settings;
- `targets/**/*.yaml`: one target requirement per file.

Create a target by copying `targets/examples/user_request.yaml`. The executable
target contract currently contains only `thrust_N`, `tsfc_kg_N_s`, optional
`total_mass_kg`, `tolerance_profile`, and the required audit fields. Size,
material, lifetime, noise, and manufacturing constraints are not executable
target conditions yet and must not be presented as enforced requirements.

Supported `target_type` values are `ID`, `Interpolation`,
`OOD-feasible-unknown`, and `Infeasible`. Use `OOD-feasible-unknown` for a new
user request unless its relationship to the training domain is established.

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n inverse_design_nova \
  python scripts/run_inverse_design.py \
  --target-config config/inverse_design/targets/examples/user_request.yaml \
  --output-dir data/inverse_design/runs/example_user_request
```

Every completed run copies the default and target source files and their SHA-256 hashes into
`<output>/configuration/`. The generated `build_config.yaml` is the exact flat
configuration passed to the independent high-precision strategy.

`gradient.enabled` and `nsga2.enabled` are independent switches. The CLI also
provides `--disable-surrogate-gradient` and `--disable-nsga2`. Disabling both
keeps CFM-4 generation, Diag-CFM prefiltering, V2.1 filtering, and real-0D
acceptance while removing learned-gradient and optimizer post-processing.

The frozen former CFM-4 default is declared in
`baselines/active10d_cfm4_v1.yaml`; the completed immutable snapshot is indexed
by `baselines/active10d_cfm4_v1_freeze.yaml`. Create it only from a clean commit
on physical GPU 6:

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n inverse_design_nova \
  python scripts/freeze_inverse_design_baseline.py
```

Verify the completed immutable snapshot without running inference again:

```bash
conda run -n inverse_design_nova \
  python scripts/freeze_inverse_design_baseline.py \
  --verify data/inverse_design/baselines/active10d_cfm4_v1
```

Files under `config/stage2/`, `config/experiments/`, and the older
`config/inverse_design/runs/` hierarchy remain reproduction/experiment inputs.
The former public facade is retained as
`src/ai/inverse_design_delivery_cfm4_legacy.py`, driven by
`legacy/cfm4_default_v1.yaml`.
