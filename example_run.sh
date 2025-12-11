#!/bin/bash
# Example script demonstrating how to run experiments on dynamical systems

echo "========================================"
echo "  WARP-Torch Dynamics Runner Examples  "
echo "========================================"
echo ""

# Example 1: List available configurations
echo "Example 1: Listing available configurations..."
python run_dynamics.py --list

# Example 2: Run a quick test with dry-run
echo ""
echo "Example 2: Dry run of sine tiny experiment..."
python run_dynamics.py --dataset sine --config tiny --dry-run

# Example 3: Check if data exists, generate if needed, then run
echo ""
echo "Example 3: Running sine tiny experiment with performance logging..."
echo "This will:"
echo "  - Check if data exists (generate if needed)"
echo "  - Run the experiment"
echo "  - Monitor memory usage"
echo "  - Generate visualizations"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

python run_dynamics.py --dataset sine --config tiny

# Example 4: Run without performance logging (faster)
echo ""
echo "Example 4: Running without performance logging (faster)..."
read -p "Press Enter to continue or Ctrl+C to cancel..."

python run_dynamics.py --dataset sine --config tiny --no-performance-logging

echo ""
echo "========================================"
echo "Examples completed!"
echo "Check experiment_logs/ for performance metrics and visualizations"
echo "Check runs/ for model checkpoints and training logs"
echo "========================================"
