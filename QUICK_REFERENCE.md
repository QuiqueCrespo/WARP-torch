# Quick Reference: run_dynamics.py

## Most Common Commands

```bash
# List all available configurations
python run_dynamics.py --list

# Run a single experiment (most common usage)
python run_dynamics.py --dataset sine --config tiny

# Run without performance logging (faster)
python run_dynamics.py --dataset sine --config tiny --no-performance-logging

# Dry run (see what would execute)
python run_dynamics.py --dataset sine --config tiny --dry-run

# Run all configs for a dataset
python run_dynamics.py --dataset sine --all-configs

# Run all datasets
python run_dynamics.py --all-datasets

# Generate data only
python run_dynamics.py --generate-data sine

# Use custom config
python run_dynamics.py --custom-config my_config.yaml
```

## Dataset Quick Reference

| Dataset | Configs Available | Data Location |
|---------|------------------|---------------|
| sine | tiny, small, medium, large, huge, physics_tiny | data/dynamics/sine/ |
| mass-spring-damper | original, original_physics, zero, zero_physics | data/dynamics/mass-spring-damper/ |
| lorentz | default | data/dynamics/lorentz-63/ |
| lotka | default | data/dynamics/lotka/ |

## Performance Logging

**Enabled by default** - Creates logs in `experiment_logs/[config]_[timestamp]/`:
- `performance_metrics.json` - Detailed metrics
- `memory_usage.png` - Memory timeline plot
- `performance_summary.png` - Summary dashboard

**Metrics tracked:**
- Execution time (minutes)
- CPU memory (peak/mean/min)
- GPU memory (if available)
- System information

**Disable with:** `--no-performance-logging`

## Output Locations

```
WARP-torch/
├── experiment_logs/           # Performance metrics (from run_dynamics.py)
│   └── [config]_[timestamp]/
│       ├── performance_metrics.json
│       ├── memory_usage.png
│       └── performance_summary.png
│
└── runs/                      # Training outputs (from main_torch.py)
    └── [dataset]/
        └── [timestamp]/
            ├── checkpoints/   # Model checkpoints
            ├── plots/         # Training plots
            ├── artefacts/     # Model artifacts
            └── log.txt        # Training log
```

## Flags Quick Reference

| Flag | Description |
|------|-------------|
| `--dataset` | Choose dataset: sine, mass-spring-damper, lorentz, lotka |
| `--config` | Choose configuration variant |
| `--all-configs` | Run all configs for selected dataset |
| `--all-datasets` | Run default config for all datasets |
| `--custom-config PATH` | Use custom YAML config |
| `--list` | List all available configurations |
| `--generate-data DATASET` | Generate data for dataset |
| `--dry-run` | Show command without executing |
| `--no-performance-logging` | Disable monitoring (faster) |
| `--skip-data-check` | Don't check if data exists |

## Typical Workflow

```bash
# 1. Check what's available
python run_dynamics.py --list

# 2. Dry run to verify command
python run_dynamics.py --dataset sine --config tiny --dry-run

# 3. Run actual experiment
python run_dynamics.py --dataset sine --config tiny

# 4. Check results
ls -la experiment_logs/sine_tiny_*/
ls -la runs/sine/

# 5. View visualizations
open experiment_logs/sine_tiny_*/memory_usage.png
open experiment_logs/sine_tiny_*/performance_summary.png
```

## Troubleshooting

```bash
# Data not found? Generate it:
python run_dynamics.py --generate-data sine

# Want to see what's available?
python run_dynamics.py --list

# Check if dependencies are installed:
pip list | grep -E "psutil|matplotlib|torch"

# Out of memory? Disable performance logging:
python run_dynamics.py --dataset sine --config tiny --no-performance-logging
```

## Example Use Cases

### Quick Test
```bash
python run_dynamics.py --dataset sine --config tiny
```

### Production Run
```bash
python run_dynamics.py --dataset mass-spring-damper --config original
```

### Batch Testing
```bash
python run_dynamics.py --dataset sine --all-configs
```

### Overnight Experiments
```bash
nohup python run_dynamics.py --all-datasets > experiment.log 2>&1 &
```

### Custom Experiment
```bash
cp cfgs/wsm/sine/sine_tiny.yaml my_experiment.yaml
# Edit my_experiment.yaml
python run_dynamics.py --custom-config my_experiment.yaml
```
