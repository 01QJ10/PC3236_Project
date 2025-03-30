# utils.py
"""
Utility functions for grid generation, boundary data, and error calculations.
"""

import numpy as np

def make_grid_2d(Lx: float, Ly: float, nx: int, ny: int):
    """
    Create a 2D spatial grid over [0, Lx] x [0, Ly].
    Returns arrays for x and y coordinates.
    """
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    return x, y

def boundary_data_2d(Lx: float, Ly: float, nx: int, ny: int, T: float, case: str = "simple", flux: float = 0.0):
    """
    Generate boundary points and target values for the PINN.

    For the time-dependent case, we generate boundary points for all times in [0, T].
    Here we assume:
      - Bottom (y=0): u = 0
      - Right (x=Lx): u = 0
      - Top (y=Ly): u(x, Ly, t) = 10 + 2*sin(2*pi*x/Lx)
      - Left (x=0): For Dirichlet, u = 0; for Neumann, flux is prescribed.
         (For simplicity, we treat left as Dirichlet with u computed from flux: u = ?)
         In this example, we assume left is Dirichlet (u=0).
    
    Returns:
      bc: tuple (x_bc, y_bc, t_bc) for boundary points.
      g: array of target boundary values.
    """
    # We create boundary points at a set of times, e.g. 20 time levels
    nt_bc = 20
    t_bc = np.linspace(0, T, nt_bc)

    # Create arrays for each boundary:
    x_bc = []
    y_bc = []
    g = []
    
    # Bottom: y = 0, u=0, x in [0,Lx]
    x_vals = np.linspace(0, Lx, nx)
    for t in t_bc:
        for x in x_vals:
            x_bc.append(x)
            y_bc.append(0.0)
            g.append(0.0)
    
    # Top: y = Ly, u = 10+2*sin(2*pi*x/Lx)
    for t in t_bc:
        for x in x_vals:
            x_bc.append(x)
            y_bc.append(Ly)
            g.append(10 + 2 * np.sin(2 * np.pi * x / Lx))
    
    # Left: x = 0, u = 0 (for simplicity; flux condition can be added similarly)
    y_vals = np.linspace(0, Ly, ny)
    for t in t_bc:
        for y in y_vals:
            x_bc.append(0.0)
            y_bc.append(y)
            g.append(0.0)
    
    # Right: x = Lx, u = 0
    for t in t_bc:
        for y in y_vals:
            x_bc.append(Lx)
            y_bc.append(y)
            g.append(0.0)
    
    # Repeat time coordinate for each boundary point
    t_bc_full = np.repeat(t_bc, (nx + nx + ny + ny))
    
    return (np.array(x_bc), np.array(y_bc), t_bc_full), np.array(g)

def initial_data_2d(Lx: float, Ly: float, nx: int, ny: int):
    """
    Generate initial condition for u(x,y,0). For example, a Gaussian bump.
    Returns:
      ic: tuple (x_ic, y_ic) of grid points on t=0.
      u0: target initial values.
    """
    x, y = make_grid_2d(Lx, Ly, nx, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    u0 = np.exp(-50 * ((X - Lx/2)**2 + (Y - Ly/2)**2))
    # Flatten grid for PINN training:
    return (X.flatten(), Y.flatten()), u0.flatten()

def l2_error(u_approx: np.ndarray, u_ref: np.ndarray) -> float:
    """
    Compute the L2 norm error between two solutions.
    """
    return np.sqrt(np.mean((u_approx - u_ref)**2))