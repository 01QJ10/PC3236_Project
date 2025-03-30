# solver.py
"""
Finite Difference Time Evolution Solver for the 2D Heat Equation

We solve the transient heat equation:
    u_t = alpha * (u_xx + u_yy)
in the domain [0, Lx] x [0, Ly] for t in [0, T].

Two cases are provided:
  1. A simple case with Dirichlet BC on all boundaries.
  2. A more complex case where the top boundary is given by a sinusoidal profile,
     and the left boundary has a prescribed flux (Neumann BC).
"""

import numpy as np

def solve_heat_2d_explicit(nx: int, ny: int, nt: int, Lx: float, Ly: float, T: float,
                             alpha: float, case: str = "simple", flux: float = 0.0, tol: float = 1e-6):
    """
    Solve the time-dependent 2D heat equation using an explicit finite difference method.

    PDE: u_t = alpha*(u_xx + u_yy)
    Domain: [0, Lx] x [0, Ly], time in [0, T]
    
    Parameters:
      nx, ny: Number of grid points in x and y directions.
      nt: Number of time steps.
      Lx, Ly: Physical dimensions of the domain.
      T: Total simulation time.
      alpha: Thermal diffusivity constant.
      case: "simple" or "complex".
         - "simple": Dirichlet BC on all edges (u=0 except initial condition).
         - "complex": Top boundary: u(x, Ly, t)=10+2*sin(2*pi*x/Lx) (Dirichlet);
                      Left boundary: Neumann BC with specified flux;
                      Other boundaries: Dirichlet u=0.
      flux: The prescribed flux (du/dx) at the left boundary (only for case="complex").
      tol: (Unused here; explicit method uses fixed time step.)
    
    Returns:
      u: 3D array with shape (nt+1, ny, nx) representing u(x,y,t).
      x: 1D array of x coordinates.
      y: 1D array of y coordinates.
      t: 1D array of time levels.
    """
    # Grid spacing and time step
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = T / nt

    # Create spatial and time grids
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    t = np.linspace(0, T, nt+1)

    # Stability condition for explicit method (CFL condition)
    r_x = alpha * dt / dx**2
    r_y = alpha * dt / dy**2
    if r_x + r_y > 0.5:
        print("Warning: CFL condition may be violated (r_x + r_y > 0.5).")

    # Initialize solution: set initial condition u(x,y,0)
    # Example initial condition: a Gaussian bump in the center.
    X, Y = np.meshgrid(x, y)
    u = np.zeros((nt+1, ny, nx))
    u[0,:,:] = np.exp(-50 * ((X - Lx/2)**2 + (Y - Ly/2)**2))

    # Time stepping loop
    for n in range(nt):
        # Copy current time step for update
        u_n = u[n,:,:].copy()
        u_next = u_n.copy()

        # Update interior points using central differences
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                u_next[j, i] = u_n[j, i] + alpha * dt * (
                    (u_n[j, i+1] - 2*u_n[j, i] + u_n[j, i-1]) / dx**2 +
                    (u_n[j+1, i] - 2*u_n[j, i] + u_n[j-1, i]) / dy**2
                )
        # Apply boundary conditions based on case:
        if case == "simple":
            # Dirichlet BC: u=0 on all boundaries.
            u_next[0, :] = 0          # bottom (y=0)
            u_next[-1, :] = 0         # top (y=Ly)
            u_next[:, 0] = 0          # left (x=0)
            u_next[:, -1] = 0         # right (x=Lx)
        elif case == "complex":
            # Top boundary: Dirichlet: u(x, Ly, t) = 10 + 2*sin(2*pi*x/Lx)
            u_next[-1, :] = 10 + 2 * np.sin(2 * np.pi * x / Lx)
            # Right and bottom: Dirichlet u = 0
            u_next[0, :] = 0
            u_next[:, -1] = 0
            # Left boundary: Neumann BC (flux)
            # Using one-sided difference: (u[1] - u[0]) / dx = flux  => u[0] = u[1] - flux*dx
            u_next[:, 0] = u_next[:, 1] - flux * dx
        # Store the new time step
        u[n+1,:,:] = u_next
    return u, x, y, t

if __name__ == "__main__":
    # For testing: solve the simple case
    u, x, y, t_vals = solve_heat_2d_explicit(nx=51, ny=51, nt=500, Lx=1.0, Ly=1.0, T=0.1, alpha=0.01, case="simple")
    print("Finite difference simulation complete.")