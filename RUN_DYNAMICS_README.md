# WARP-Torch Dynamics Runner

A comprehensive script for running WARP-torch experiments on dynamical systems datasets with automatic performance monitoring, memory tracking, and visualization generation.

## Features

- **Automated Experiment Execution**: Run single or multiple experiments with ease
- **Performance Monitoring**:
  - Real-time memory usage tracking (CPU and GPU)
  - Execution time measurement
  - System information logging
- **Automatic Visualizations**:
  - Memory usage plots over time
  - Performance summary dashboards
  - Training metrics visualization
- **Data Management**: Automatic data generation if needed
- **Flexible Configuration**: Support for all dynamical systems datasets

## Installation

### Required Dependencies

```bash
pip install psutil matplotlib numpy
```

The script also requires the existing WARP-torch dependencies (PyTorch, JAX, etc.).

## Quick Start

### 1. List Available Configurations

```bash
python run_dynamics.py --list
```

This shows all available datasets and their configurations.

### 2. Run a Single Experiment

```bash
# Run sine dataset with tiny configuration
python run_dynamics.py --dataset sine --config tiny

# Run mass-spring-damper with original configuration
python run_dynamics.py --dataset mass-spring-damper --config original

# Run Lorentz attractor (uses default config)
python run_dynamics.py --dataset lorentz
```

### 3. Run Multiple Experiments

```bash
# Run all configurations for sine dataset
python run_dynamics.py --dataset sine --all-configs

# Run default config for all datasets
python run_dynamics.py --all-datasets
```

### 4. Use Custom Configuration

```bash
python run_dynamics.py --custom-config path/to/my_config.yaml
```

## Available Datasets and Configurations

### Sine Wave
- `tiny`: 1 sample (for quick testing)
- `small`: Small dataset
- `medium`: Medium dataset
- `large`: Large dataset
- `huge`: Huge dataset
- `physics_tiny`: Physics-informed variant

### Mass-Spring-Damper
- `original`: Standard configuration
- `original_physics`: Physics-informed variant
- `zero`: Zero initial conditions
- `zero_physics`: Zero initial conditions with physics

### Lorentz-63
- `default`: Default configuration

### Lotka-Volterra
- `default`: Default configuration

## Command Line Options

### Dataset Selection
- `--dataset DATASET`: Choose dataset (sine, mass-spring-damper, lorentz, lotka)
- `--config CONFIG`: Choose specific configuration
- `--all-configs`: Run all configs for selected dataset
- `--all-datasets`: Run default config for all datasets
- `--custom-config PATH`: Use custom config file

### Data Management
- `--generate-data DATASET`: Generate data for specified dataset
- `--skip-data-check`: Skip checking if data files exist

### Execution Options
- `--dry-run`: Show what would be executed without running
- `--no-performance-logging`: Disable performance monitoring (faster execution)

### Information
- `--list`: List all available configurations

## Performance Logging

By default, the script creates detailed performance logs for each experiment:

### Log Directory Structure

```
experiment_logs/
└── sine_tiny_20250101_123456/
    ├── performance_metrics.json      # Detailed metrics in JSON
    ├── memory_usage.png              # Memory over time plot
    └── performance_summary.png       # Overall summary dashboard
```

### Metrics Collected

1. **Execution Time**: Total runtime in seconds/minutes
2. **Memory Usage**:
   - CPU Memory (RSS): Peak, mean, minimum
   - GPU Memory (if available): Peak allocated, max reserved
3. **System Information**:
   - CPU count
   - Total RAM
   - CUDA availability and device info
4. **Timestamps**: Start and end times

### Example Output

```
Performance logging enabled
Log directory: experiment_logs/sine_tiny_20250101_123456

Executing: python main_torch.py cfgs/wsm/sine/sine_tiny.yaml

✓ Experiment completed successfully
Execution time: 5.23 minutes

Collecting performance metrics...

Memory Usage Summary:
  CPU Memory (RSS):
    Peak: 2.45 GB
    Mean: 1.87 GB
  GPU Memory:
    Peak Allocated: 1.23 GB
    Max Reserved: 1.50 GB

Generating visualizations...
   Memory plot saved to: experiment_logs/sine_tiny_20250101_123456/memory_usage.png
   Summary plot saved to: experiment_logs/sine_tiny_20250101_123456/performance_summary.png
   Performance metrics saved to: experiment_logs/sine_tiny_20250101_123456/performance_metrics.json

Performance logs saved to: experiment_logs/sine_tiny_20250101_123456
```

## Usage Examples

### Example 1: Quick Test Run

```bash
# Dry run to see what would be executed
python run_dynamics.py --dataset sine --config tiny --dry-run

# Actual run
python run_dynamics.py --dataset sine --config tiny
```

### Example 2: Generate Data and Run

```bash
# Generate data for sine dataset
python run_dynamics.py --generate-data sine

# Run experiment
python run_dynamics.py --dataset sine --config medium
```

### Example 3: Batch Processing

```bash
# Run all sine configurations
python run_dynamics.py --dataset sine --all-configs

# Run all datasets with default configs
python run_dynamics.py --all-datasets
```

### Example 4: Without Performance Logging (Faster)

```bash
# Skip performance monitoring for faster execution
python run_dynamics.py --dataset sine --config tiny --no-performance-logging
```

### Example 5: Custom Experiment

```bash
# Create your own config file
cp cfgs/wsm/sine/sine_tiny.yaml my_experiment.yaml
# Edit my_experiment.yaml with your parameters

# Run it
python run_dynamics.py --custom-config my_experiment.yaml
```

## Data Generation

The script can automatically generate data for datasets that have `datagen.py` scripts:

```bash
# Generate data for specific dataset
python run_dynamics.py --generate-data sine

# The script will prompt to generate data if it's missing
python run_dynamics.py --dataset sine --config tiny
# Output: ⚠️  Data files not found for sine
#         Attempt to generate data? [Y/n]:
```

## Visualizations

### Memory Usage Plot

Shows CPU and GPU memory usage over time during experiment execution:
- Line plot of memory consumption
- Helps identify memory leaks or spikes
- Separate plots for CPU (RSS) and GPU memory

### Performance Summary Dashboard

Multi-panel visualization showing:
1. **Execution Time**: Bar chart of total runtime
2. **Memory Statistics**: Peak, mean, and minimum memory usage
3. **System Information**: Hardware and CUDA details
4. **Training Progress**: Loss curves (if available from logs)

## Troubleshooting

### Missing Dependencies

```bash
# If psutil is not installed
pip install psutil

# If matplotlib is not installed
pip install matplotlib
```

### Data Not Found

```bash
# Generate data manually
cd data/dynamics/sine/
python datagen.py
cd ../../..

# Or use the script
python run_dynamics.py --generate-data sine
```

### CUDA Out of Memory

If you encounter GPU memory issues, try:
1. Use `--no-performance-logging` to reduce overhead
2. Reduce batch size in config file
3. Use a smaller dataset variant

### Config File Not Found

```bash
# List available configs
python run_dynamics.py --list

# Check if config exists
ls -la cfgs/wsm/sine/
```

## Performance Tips

1. **Use `--dry-run`** first to verify your command
2. **Disable logging** (`--no-performance-logging`) for faster execution if you don't need metrics
3. **Run batch experiments** overnight with `--all-configs` or `--all-datasets`
4. **Monitor logs** in real-time: `tail -f experiment_logs/*/performance_metrics.json`

## Advanced Usage

### Integrating with Existing Workflows

```bash
#!/bin/bash
# Example: Run multiple experiments and collect results

DATASETS=("sine" "mass-spring-damper" "lorentz")

for dataset in "${DATASETS[@]}"; do
    echo "Running $dataset..."
    python run_dynamics.py --dataset $dataset
done

# Aggregate results
python analyze_results.py experiment_logs/
```

### Monitoring Long-Running Experiments

```bash
# In one terminal: Run experiment
python run_dynamics.py --dataset lorentz

# In another terminal: Monitor progress
watch -n 5 'tail -n 20 runs/lorentz-63/*/log.txt'
```

## Output Files

### From WARP-torch (in `runs/` directory)
- Training checkpoints
- Training logs
- Generated plots
- Model artifacts

### From Performance Logger (in `experiment_logs/` directory)
- `performance_metrics.json`: Detailed metrics
- `memory_usage.png`: Memory timeline
- `performance_summary.png`: Summary dashboard

## FAQ

**Q: How do I disable performance logging?**
A: Add `--no-performance-logging` flag

**Q: Can I run experiments in parallel?**
A: This script runs sequentially. For parallel execution, use multiple terminal sessions or a job scheduler.

**Q: Where are the trained models saved?**
A: In `runs/[dataset]/[timestamp]/checkpoints/`

**Q: How do I customize experiment parameters?**
A: Edit the YAML config file or create a custom config

**Q: Can I add my own dataset?**
A: Yes! Add it to `DATASET_CONFIGS` and `DATASET_PATHS` in the script

## Contributing

To add support for new datasets:

1. Add dataset to `DATASET_CONFIGS` dictionary
2. Add data path to `DATASET_PATHS` dictionary
3. Create config YAML file in `cfgs/wsm/[dataset_name]/`
4. Optionally create `datagen.py` in `data/dynamics/[dataset_name]/`

## License

Same as WARP-torch project.

## Support

For issues specific to this runner script, check:
1. Data files exist: `ls data/dynamics/[dataset]/`
2. Config files exist: `ls cfgs/wsm/[dataset]/`
3. Dependencies installed: `pip list | grep -E "psutil|matplotlib|torch"`

For WARP-torch issues, refer to the main project documentation.
