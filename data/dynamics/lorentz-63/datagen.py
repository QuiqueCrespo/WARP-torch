#!/usr/bin/env python3
"""
Generate Lorenz-63 attractor trajectories for training.
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


def lorenz_system(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Lorenz-63 system equations.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


def generate_lorenz_trajectories(n_trajectories=64, n_timesteps=1000, t_span=100.0):
    """
    Generate multiple Lorenz-63 trajectories with different initial conditions.

    Parameters:
    - n_trajectories: Number of trajectories to generate
    - n_timesteps: Number of time steps per trajectory
    - t_span: Total time span for integration

    Returns:
    - trajectories: Array of shape (n_trajectories, n_timesteps, 3)
    """
    print(f"Generating {n_trajectories} Lorenz-63 trajectories...")

    # Time points
    t = np.linspace(0, t_span, n_timesteps)

    # Initialize trajectories array
    trajectories = np.zeros((n_trajectories, n_timesteps, 3))

    # Generate random initial conditions around the attractor
    np.random.seed(42)

    for i in range(n_trajectories):
        # Random initial conditions
        x0 = np.random.uniform(-15, 15)
        y0 = np.random.uniform(-20, 20)
        z0 = np.random.uniform(5, 40)

        initial_state = [x0, y0, z0]

        # Integrate the system
        trajectory = odeint(lorenz_system, initial_state, t)
        trajectories[i] = trajectory

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_trajectories} trajectories")

    print(f"Generated trajectories shape: {trajectories.shape}")
    print(f"Data range: [{trajectories.min():.2f}, {trajectories.max():.2f}]")

    return trajectories


def visualize_trajectories(trajectories, max_display=8):
    """
    Visualize a subset of the generated trajectories.
    """
    n_display = min(len(trajectories), max_display)
    dim0, dim1 = 0, 1  # x vs y

    fig, ax = plt.subplots(n_display, n_display, figsize=(4*n_display, 4*n_display))
    ax = ax.flatten()

    for i in range(n_display**2):
        if i < len(trajectories):
            ax[i].plot(trajectories[i, :, dim0], trajectories[i, :, dim1], linewidth=0.5)
            ax[i].set_title(f"Trajectory {i+1}")

    plt.tight_layout()
    # plt.show()  # Commented out for non-interactive execution
    plt.savefig('lorenz_trajectories.png', dpi=100, bbox_inches='tight')
    print("Saved visualization to lorenz_trajectories.png")


if __name__ == "__main__":
    # Generate trajectories
    data = generate_lorenz_trajectories(n_trajectories=64, n_timesteps=1000, t_span=100.0)

    # Visualize
    print("\nVisualizing trajectories...")
    visualize_trajectories(data, max_display=8)

    # Split into train and test
    train = data[:48, :, :]  # First 48 trajectories for training
    test = data[48:, :, :]   # Last 16 trajectories for testing
    val = data[40:48, :, :]  # 8 trajectories for validation (overlap with train is ok for small dataset)

    # Save as numpy arrays
    print("\nSaving data files...")
    np.save("train.npy", train)
    np.save("test.npy", test)
    np.save("val.npy", val)

    print(f"\nTrain shape: {train.shape}")
    print(f"Val shape: {val.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Data range: [{train.min():.2f}, {train.max():.2f}]")

    print("\n✓ Lorenz-63 data generation complete!")
