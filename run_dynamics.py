#!/usr/bin/env python3
"""
Script to run WARP-torch experiments on dynamical systems datasets.

Usage:
    python run_dynamics.py --dataset sine --config tiny
    python run_dynamics.py --dataset mass-spring-damper --config original
    python run_dynamics.py --all-datasets
    python run_dynamics.py --custom-config path/to/config.yaml
"""

import argparse
import os
import sys
import subprocess
import time
import json
import psutil
import threading
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


DATASET_CONFIGS = {
    "sine": {
        "tiny": "cfgs/wsm/sine/sine_tiny.yaml",
        "small": "cfgs/wsm/sine/sine_small.yaml",
        "medium": "cfgs/wsm/sine/sine_medium.yaml",
        "large": "cfgs/wsm/sine/sine_large.yaml",
        "huge": "cfgs/wsm/sine/sine_huge.yaml",
        "physics_tiny": "cfgs/wsm/sine/physics_sine_tiny.yaml",
    },
    "mass-spring-damper": {
        "original": "cfgs/wsm/msd/original.yaml",
        "original_physics": "cfgs/wsm/msd/original_physics.yaml",
        "zero": "cfgs/wsm/msd/zero.yaml",
        "zero_physics": "cfgs/wsm/msd/zero_physics.yaml",
    },
    "lorentz": {
        "default": "cfgs/wsm/lorentz/lorentz.yaml",
    },
    "lotka": {
        "default": "cfgs/wsm/lotka/lotka.yaml",
    },
}

# Data paths for reference
DATASET_PATHS = {
    "sine": "data/dynamics/sine/",
    "mass-spring-damper": "data/dynamics/mass-spring-damper/",
    "lorentz": "data/dynamics/lorentz-63/",
    "lotka": "data/dynamics/lotka/",
    "cheetah": "data/dynamics/cheetah/",
    "electricity": "data/dynamics/electricity/",
}


class MemoryMonitor:
    """Monitor memory usage during experiment execution."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.thread = None
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.process = psutil.Process()

    def _monitor(self):
        """Background monitoring function."""
        while self.running:
            # CPU memory
            mem_info = self.process.memory_info()
            self.memory_samples.append({
                'timestamp': time.time(),
                'rss': mem_info.rss / (1024 ** 3),  # GB
                'vms': mem_info.vms / (1024 ** 3),  # GB
            })

            # GPU memory (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    gpu_max = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
                    self.gpu_memory_samples.append({
                        'timestamp': time.time(),
                        'allocated': gpu_mem,
                        'max_allocated': gpu_max,
                    })
            except ImportError:
                pass

            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        self.running = True
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def get_stats(self):
        """Get memory statistics."""
        if not self.memory_samples:
            return {}

        rss_values = [s['rss'] for s in self.memory_samples]
        stats = {
            'cpu_memory': {
                'peak_rss_gb': max(rss_values),
                'mean_rss_gb': np.mean(rss_values),
                'min_rss_gb': min(rss_values),
            }
        }

        if self.gpu_memory_samples:
            allocated_values = [s['allocated'] for s in self.gpu_memory_samples]
            max_values = [s['max_allocated'] for s in self.gpu_memory_samples]
            stats['gpu_memory'] = {
                'peak_allocated_gb': max(allocated_values),
                'mean_allocated_gb': np.mean(allocated_values),
                'max_reserved_gb': max(max_values),
            }

        return stats

    def plot(self, output_path):
        """Generate memory usage plots."""
        if not self.memory_samples:
            return

        fig, axes = plt.subplots(2 if self.gpu_memory_samples else 1, 1, figsize=(12, 8))
        if not self.gpu_memory_samples:
            axes = [axes]

        # CPU memory plot
        times = [(s['timestamp'] - self.memory_samples[0]['timestamp']) / 60 for s in self.memory_samples]
        rss = [s['rss'] for s in self.memory_samples]

        axes[0].plot(times, rss, label='RSS', linewidth=2)
        axes[0].set_xlabel('Time (minutes)')
        axes[0].set_ylabel('Memory (GB)')
        axes[0].set_title('CPU Memory Usage Over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # GPU memory plot
        if self.gpu_memory_samples:
            gpu_times = [(s['timestamp'] - self.gpu_memory_samples[0]['timestamp']) / 60
                        for s in self.gpu_memory_samples]
            allocated = [s['allocated'] for s in self.gpu_memory_samples]
            max_allocated = [s['max_allocated'] for s in self.gpu_memory_samples]

            axes[1].plot(gpu_times, allocated, label='Allocated', linewidth=2)
            axes[1].plot(gpu_times, max_allocated, label='Max Allocated', linewidth=2, linestyle='--')
            axes[1].set_xlabel('Time (minutes)')
            axes[1].set_ylabel('Memory (GB)')
            axes[1].set_title('GPU Memory Usage Over Time')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Memory plot saved to: {output_path}")


class PerformanceLogger:
    """Log and visualize performance metrics."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            'config': None,
            'start_time': None,
            'end_time': None,
            'duration_seconds': None,
            'memory_stats': None,
            'system_info': self._get_system_info(),
        }

    def _get_system_info(self):
        """Collect system information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024 ** 3),
            'platform': sys.platform,
        }

        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_device'] = torch.cuda.get_device_name(0)
                info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except ImportError:
            info['cuda_available'] = False

        return info

    def start_experiment(self, config_path):
        """Mark experiment start."""
        self.metrics['config'] = str(config_path)
        self.metrics['start_time'] = datetime.now().isoformat()

    def end_experiment(self, memory_stats):
        """Mark experiment end and save metrics."""
        self.metrics['end_time'] = datetime.now().isoformat()
        start = datetime.fromisoformat(self.metrics['start_time'])
        end = datetime.fromisoformat(self.metrics['end_time'])
        self.metrics['duration_seconds'] = (end - start).total_seconds()
        self.metrics['memory_stats'] = memory_stats

        # Save metrics to JSON
        metrics_file = self.log_dir / 'performance_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"   Performance metrics saved to: {metrics_file}")

    def parse_training_logs(self, log_file):
        """Parse training logs to extract performance metrics."""
        if not Path(log_file).exists():
            return None

        metrics = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
        }

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # This is a placeholder - adapt to actual log format
                    if 'epoch' in line.lower() and 'loss' in line.lower():
                        # Parse epoch and loss values
                        # Example: "Epoch 100: train_loss=0.0123, val_loss=0.0145"
                        pass
        except Exception as e:
            print(f"   Warning: Could not parse training logs: {e}")
            return None

        return metrics if metrics['epochs'] else None

    def generate_summary_plot(self, output_path, training_metrics=None):
        """Generate a summary visualization."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Execution time
        ax1 = fig.add_subplot(gs[0, 0])
        duration_min = self.metrics['duration_seconds'] / 60
        ax1.barh(['Execution Time'], [duration_min], color='steelblue')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_title('Total Execution Time')
        ax1.grid(True, alpha=0.3, axis='x')

        # Memory usage
        ax2 = fig.add_subplot(gs[0, 1])
        if self.metrics.get('memory_stats'):
            mem_stats = self.metrics['memory_stats']
            if 'cpu_memory' in mem_stats:
                cpu_mem = mem_stats['cpu_memory']
                labels = ['Peak', 'Mean', 'Min']
                values = [cpu_mem['peak_rss_gb'], cpu_mem['mean_rss_gb'], cpu_mem['min_rss_gb']]
                ax2.bar(labels, values, color=['red', 'orange', 'green'])
                ax2.set_ylabel('Memory (GB)')
                ax2.set_title('CPU Memory Usage')
                ax2.grid(True, alpha=0.3, axis='y')

        # System info
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        sys_info = self.metrics['system_info']
        info_text = f"""
System Information:
  CPUs: {sys_info['cpu_count']}
  Total RAM: {sys_info['total_memory_gb']:.1f} GB
  CUDA Available: {sys_info.get('cuda_available', 'N/A')}
  CUDA Device: {sys_info.get('cuda_device', 'N/A')}
  CUDA Memory: {sys_info.get('cuda_memory_gb', 'N/A')} GB

Experiment:
  Config: {Path(self.metrics['config']).name}
  Start Time: {self.metrics['start_time']}
  Duration: {duration_min:.2f} minutes
        """
        ax3.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                verticalalignment='center')

        # Training metrics (if available)
        if training_metrics and training_metrics.get('epochs'):
            ax4 = fig.add_subplot(gs[2, :])
            ax4.plot(training_metrics['epochs'], training_metrics['train_loss'],
                    label='Train Loss', linewidth=2)
            if training_metrics.get('val_loss'):
                ax4.plot(training_metrics['epochs'], training_metrics['val_loss'],
                        label='Val Loss', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Training Progress')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Summary plot saved to: {output_path}")


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_available_configs():
    """Print all available dataset configurations."""
    print_header("Available Dynamical Systems Configurations")

    for dataset, configs in DATASET_CONFIGS.items():
        print(f"\n{dataset}:")
        for config_name, config_path in configs.items():
            exists = "✓" if Path(config_path).exists() else "✗"
            print(f"  {exists} {config_name:20s} -> {config_path}")

    print("\n")


def check_data_exists(dataset):
    """Check if dataset files exist."""
    if dataset not in DATASET_PATHS:
        return True  # Unknown dataset, assume it's okay

    data_path = Path(DATASET_PATHS[dataset])
    if not data_path.exists():
        return False

    # Check for common data files
    has_train = (data_path / "train.npy").exists() or (data_path / "train.npz").exists()
    has_test = (data_path / "test.npy").exists() or (data_path / "test.npz").exists()

    return has_train and has_test


def generate_data(dataset):
    """Generate data for a dataset if datagen.py exists."""
    if dataset not in DATASET_PATHS:
        print(f"⚠️  Unknown dataset: {dataset}")
        return False

    data_path = Path(DATASET_PATHS[dataset])
    datagen_script = data_path / "datagen.py"

    if not datagen_script.exists():
        print(f"⚠️  No data generator found at {datagen_script}")
        return False

    print(f"🔄 Generating data for {dataset}...")
    print(f"   Running: python {datagen_script}")

    try:
        result = subprocess.run(
            ["python", "datagen.py"],
            cwd=str(data_path),
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Data generation complete")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Data generation failed: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def run_experiment(config_path, dry_run=False, log_performance=True):
    """Run a single experiment with the given config."""
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False

    print_header(f"Running Experiment: {config_path.name}")
    print(f"Config: {config_path}")
    print(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cmd = ["python", "main_torch_converted.py", str(config_path)]

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    # Setup performance monitoring
    monitor = None
    logger = None
    if log_performance:
        # Create log directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path('experiment_logs') / f"{config_path.stem}_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Performance logging enabled")
        print(f"Log directory: {log_dir}\n")

        logger = PerformanceLogger(log_dir)
        logger.start_experiment(config_path)

        monitor = MemoryMonitor(interval=2.0)
        monitor.start()

    try:
        print(f"Executing: {' '.join(cmd)}\n")
        start_time = time.time()
        result = subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time

        print(f"\n✓ Experiment completed successfully")
        print(f"Execution time: {elapsed_time / 60:.2f} minutes\n")

        # Stop monitoring and save results
        if log_performance and monitor and logger:
            print("Collecting performance metrics...")
            monitor.stop()
            memory_stats = monitor.get_stats()

            # Print memory stats
            if memory_stats:
                print("\nMemory Usage Summary:")
                if 'cpu_memory' in memory_stats:
                    cpu = memory_stats['cpu_memory']
                    print(f"  CPU Memory (RSS):")
                    print(f"    Peak: {cpu['peak_rss_gb']:.2f} GB")
                    print(f"    Mean: {cpu['mean_rss_gb']:.2f} GB")
                if 'gpu_memory' in memory_stats:
                    gpu = memory_stats['gpu_memory']
                    print(f"  GPU Memory:")
                    print(f"    Peak Allocated: {gpu['peak_allocated_gb']:.2f} GB")
                    print(f"    Max Reserved: {gpu['max_reserved_gb']:.2f} GB")

            # Save performance data
            logger.end_experiment(memory_stats)

            # Generate visualizations
            print("\nGenerating visualizations...")
            monitor.plot(log_dir / 'memory_usage.png')
            logger.generate_summary_plot(log_dir / 'performance_summary.png')

            print(f"\nPerformance logs saved to: {log_dir}")

        return True

    except subprocess.CalledProcessError as e:
        if monitor:
            monitor.stop()
        print(f"\n✗ Experiment failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        if monitor:
            monitor.stop()
        print("\n⚠️  Experiment interrupted by user")
        return False


def run_multiple_experiments(configs, dry_run=False, log_performance=True):
    """Run multiple experiments sequentially."""
    print_header(f"Running {len(configs)} Experiments")

    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Starting experiment: {config}")
        success = run_experiment(config, dry_run=dry_run, log_performance=log_performance)
        results.append((config, success))

        if not success and not dry_run:
            response = input("\nContinue with remaining experiments? [y/N]: ")
            if response.lower() != 'y':
                break

    # Print summary
    print_header("Experiment Summary")
    for config, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {config}")

    successful = sum(1 for _, success in results if success)
    print(f"\nCompleted: {successful}/{len(results)} experiments successful")


def main():
    parser = argparse.ArgumentParser(
        description="Run WARP-torch experiments on dynamical systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sine dataset with tiny config
  python run_dynamics.py --dataset sine --config tiny

  # Run mass-spring-damper with original config
  python run_dynamics.py --dataset mass-spring-damper --config original

  # Run all configs for a dataset
  python run_dynamics.py --dataset sine --all-configs

  # Run with custom config file
  python run_dynamics.py --custom-config my_experiment.yaml

  # List available configurations
  python run_dynamics.py --list

  # Generate data for a dataset
  python run_dynamics.py --generate-data sine

  # Dry run (show what would be executed)
  python run_dynamics.py --dataset sine --config tiny --dry-run
        """
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dynamical system dataset to use"
    )
    parser.add_argument(
        "--config",
        help="Configuration name (e.g., tiny, small, original)"
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run all configurations for the selected dataset"
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run default config for all datasets"
    )
    parser.add_argument(
        "--custom-config",
        type=str,
        help="Path to custom configuration file"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available configurations"
    )
    parser.add_argument(
        "--generate-data",
        type=str,
        metavar="DATASET",
        help="Generate data for specified dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip checking if data files exist"
    )
    parser.add_argument(
        "--no-performance-logging",
        action="store_true",
        help="Disable performance logging and monitoring"
    )

    args = parser.parse_args()

    # Handle list command
    if args.list:
        print_available_configs()
        return 0

    # Handle data generation
    if args.generate_data:
        success = generate_data(args.generate_data)
        return 0 if success else 1

    # Determine which experiments to run
    configs_to_run = []

    if args.custom_config:
        configs_to_run = [args.custom_config]

    elif args.all_datasets:
        # Run default config for each dataset
        for dataset, configs in DATASET_CONFIGS.items():
            # Use first available config as default
            default_config = list(configs.values())[0]
            configs_to_run.append(default_config)

    elif args.dataset:
        if args.dataset not in DATASET_CONFIGS:
            print(f"✗ Unknown dataset: {args.dataset}")
            print_available_configs()
            return 1

        dataset_configs = DATASET_CONFIGS[args.dataset]

        if args.all_configs:
            # Run all configs for this dataset
            configs_to_run = list(dataset_configs.values())

        elif args.config:
            if args.config not in dataset_configs:
                print(f"✗ Unknown config '{args.config}' for dataset '{args.dataset}'")
                print(f"\nAvailable configs for {args.dataset}:")
                for name in dataset_configs.keys():
                    print(f"  - {name}")
                return 1
            configs_to_run = [dataset_configs[args.config]]

        else:
            # Use default (first) config
            configs_to_run = [list(dataset_configs.values())[0]]
            print(f"No config specified, using default: {Path(configs_to_run[0]).name}")

        # Check if data exists (skip for dry-run)
        if not args.skip_data_check and not args.dry_run and not check_data_exists(args.dataset):
            print(f"⚠️  Data files not found for {args.dataset}")
            print(f"   Expected location: {DATASET_PATHS[args.dataset]}")
            response = input(f"\nAttempt to generate data? [Y/n]: ")
            if response.lower() != 'n':
                if not generate_data(args.dataset):
                    print("✗ Failed to generate data. Aborting.")
                    return 1
            else:
                print("Aborting.")
                return 1

    else:
        print("Error: No experiment specified.")
        print("Use --help to see available options, or --list to see configurations.")
        return 1

    # Run experiments
    if not configs_to_run:
        print("No experiments to run.")
        return 1

    log_performance = not args.no_performance_logging

    if len(configs_to_run) == 1:
        success = run_experiment(configs_to_run[0], dry_run=args.dry_run, log_performance=log_performance)
        return 0 if success else 1
    else:
        run_multiple_experiments(configs_to_run, dry_run=args.dry_run, log_performance=log_performance)
        return 0


if __name__ == "__main__":
    sys.exit(main())
