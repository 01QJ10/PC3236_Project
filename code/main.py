# main.py
"""
Main entry point for the 2D Heat Equation Project.
This script runs the finite difference solver for time evolution (both simple and complex cases)
and then runs the PINN solver. It visualizes the results and computes error metrics.
"""

import numpy as np
from solver import solve_heat_2d_explicit
from pinn import init_network_params, train_pinn, predict_solution
from utils import make_grid_2d, boundary_data_2d, initial_data_2d, l2_error
from visualisation import plot_heatmap, plot_contour
import jax
import jax.numpy as jnp

# Set simulation parameters
Lx, Ly = 1.0, 1.0  # domain dimensions
nx, ny = 51, 51    # grid points
T = 0.1            # total simulation time
nt = 500           # time steps
alpha = 0.01       # thermal diffusivity

# ----------------------------
# Finite Difference Solver
# ----------------------------
print("Running Finite Difference Solver (Simple case)...")
u_fd_simple, x, y, t_vals = solve_heat_2d_explicit(nx, ny, nt, Lx, Ly, T, alpha, case="simple")
# Visualize final time step
plot_heatmap(u_fd_simple[-1,:,:], x, y, title="FD Solution (Simple) at t = T")
plot_contour(u_fd_simple[-1,:,:], x, y, title="FD Contour (Simple) at t = T")

print("Running Finite Difference Solver (Complex case with flux on left)...")
# For complex case, set a flux value (e.g., 1.0) on left boundary
flux = 1.0
u_fd_complex, x, y, t_vals = solve_heat_2d_explicit(nx, ny, nt, Lx, Ly, T, alpha, case="complex", flux=flux)
plot_heatmap(u_fd_complex[-1,:,:], x, y, title="FD Solution (Complex) at t = T")
plot_contour(u_fd_complex[-1,:,:], x, y, title="FD Contour (Complex) at t = T")

# ----------------------------
# PINN Solver
# ----------------------------
print("Preparing PINN training data...")

# Collocation points in the interior (random sampling)
num_coll = 1000
x_coll = np.random.uniform(0, Lx, num_coll)
y_coll = np.random.uniform(0, Ly, num_coll)
t_coll = np.random.uniform(0, T, num_coll)
colloc = (x_coll, y_coll, t_coll)

# Boundary data (using the same boundaries as in the complex case)
(bc_x, bc_y, bc_t), bc_val = boundary_data_2d(Lx, Ly, nx, ny, T, case="complex", flux=flux)

# Initial condition data
(ic_x, ic_y), ic_val = initial_data_2d(Lx, Ly, nx, ny)

# Define network architecture: input 3 -> 2 hidden layers (64 neurons) -> 1 output
layer_sizes = [3, 64, 64, 1]
key = jax.random.PRNGKey(0)
params = init_network_params(layer_sizes, key)

print("Training PINN...")
params = train_pinn(params, colloc, (bc_x, bc_y, bc_t), bc_val, (ic_x, ic_y), ic_val, alpha,
                    epochs=5000, lr=1e-3)

# Predict solution at final time T on a grid for comparison
xs = np.linspace(0, Lx, nx)
ys = np.linspace(0, Ly, ny)
ts = np.array([T])
u_pinn = predict_solution(params, xs, ys, ts)[0,:,:]

plot_heatmap(u_pinn, xs, ys, title="PINN Solution at t = T")
plot_contour(u_pinn, xs, ys, title="PINN Contour at t = T")

# Compute L2 error between FD (complex case) and PINN at t = T
error = l2_error(u_pinn, u_fd_complex[-1,:,:])
print(f"L2 error between FD (complex) and PINN at t = T: {error:.6f}")