# pinn.py
"""
PINN Solver for the Time-Dependent 2D Heat Equation

We solve the PDE:
    u_t = alpha * (u_xx + u_yy)
in the domain [0,Lx] x [0,Ly] for t in [0,T].
Boundary conditions (for the complex case):
    - Top boundary: u(x, Ly, t) = 10 + 2*sin(2*pi*x/Lx)
    - Left boundary (Neumann): u_x(0, y, t) = flux (prescribed flux)
    - Other boundaries: Dirichlet: u=0
Initial condition: given (e.g., a Gaussian bump).

We define a neural network u(x,y,t;theta) and minimize a loss function
that includes the PDE residual (computed via automatic differentiation) and the boundary conditions.
"""

import jax
import jax.numpy as jnp
import numpy as np

def init_network_params(layer_sizes, key):
    """
    Initialize network parameters for an MLP with given layer sizes.
    Uses Xavier initialization.
    """
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        in_size = layer_sizes[i]
        out_size = layer_sizes[i+1]
        W = jax.random.normal(subkey, (in_size, out_size)) * jnp.sqrt(2.0 / (in_size + out_size))
        b = jnp.zeros((out_size,))
        params.append((W, b))
    return params

def neural_net(params, x, y, t):
    """
    Forward pass for the MLP.
    Input: scalars x, y, t.
    Output: u, an approximation for the temperature.
    """
    z = jnp.stack([x, y, t])
    for (W, b) in params[:-1]:
        z = jnp.tanh(jnp.dot(W.T, z) + b)
    W_last, b_last = params[-1]
    u = jnp.dot(W_last.T, z) + b_last
    return u[0]

def pde_residual(params, x, y, t, alpha):
    """
    Compute the PDE residual at a single point (x,y,t):
      f(x,y,t) = u_t - alpha*(u_xx + u_yy)
    using automatic differentiation.
    """
    u_fun = lambda x, y, t: neural_net(params, x, y, t)
    # First derivatives
    u_t = jax.grad(u_fun, argnums=2)(x, y, t)
    u_x = jax.grad(u_fun, argnums=0)(x, y, t)
    u_y = jax.grad(u_fun, argnums=1)(x, y, t)
    # Second derivatives
    u_xx = jax.grad(lambda xx, y, t: jax.grad(u_fun, argnums=0)(xx, y, t))(x, y, t)
    u_yy = jax.grad(lambda x, yy, t: jax.grad(u_fun, argnums=1)(x, yy, t))(x, y, t)
    return u_t - alpha * (u_xx + u_yy)

# Vectorize the residual computation for batches of points.
pde_residual_v = jax.vmap(pde_residual, in_axes=(None, 0, 0, 0, None))

def loss_fn(params, colloc, bc, bc_val, ic, ic_val, alpha):
    """
    Total loss for PINN:
      - PDE residual loss on collocation points.
      - Boundary condition loss on boundary points.
      - Initial condition loss on initial points.
    
    Parameters:
      colloc: tuple of arrays (x_coll, y_coll, t_coll) for interior domain points.
      bc: tuple (x_bc, y_bc, t_bc) for boundary points.
      bc_val: target values at boundary points.
      ic: tuple (x_ic, y_ic) for initial condition (t=0).
      ic_val: target initial condition values.
      alpha: thermal diffusivity.
    """
    x_coll, y_coll, t_coll = colloc
    f = pde_residual_v(params, x_coll, y_coll, t_coll, alpha)
    loss_pde = jnp.mean(jnp.square(f))
    
    # Boundary loss: for Dirichlet boundaries, compare u(x,y,t) to target.
    x_bc, y_bc, t_bc = bc
    u_bc = jax.vmap(lambda x, y, t: neural_net(params, x, y, t))(x_bc, y_bc, t_bc)
    loss_bc = jnp.mean(jnp.square(u_bc - bc_val))
    
    # Initial condition loss: t=0
    x_ic, y_ic = ic
    u_ic = jax.vmap(lambda x, y: neural_net(params, x, y, 0.0))(x_ic, y_ic)
    loss_ic = jnp.mean(jnp.square(u_ic - ic_val))
    
    return loss_pde + loss_bc + loss_ic

def train_pinn(params, colloc, bc, bc_val, ic, ic_val, alpha, epochs=5000, lr=1e-3):
    """
    Train the PINN using gradient descent.
    
    Returns updated network parameters.
    """
    loss_and_grad = jax.value_and_grad(lambda p: loss_fn(p, colloc, bc, bc_val, ic, ic_val, alpha))
    for epoch in range(epochs):
        loss_val, grads = loss_and_grad(params)
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss_val:.6f}")
    return params

def predict_solution(params, xs, ys, ts):
    """
    Evaluate the PINN solution on a grid defined by arrays xs and ys for each time in ts.
    Returns a 3D numpy array with shape (len(ts), len(ys), len(xs)).
    """
    def predict_at_time(t):
        # Create a meshgrid for the current time t.
        X, Y = jnp.meshgrid(xs, ys, indexing='xy')  # X, Y shapes: (len(ys), len(xs))
        # Flatten the grid so that each element is a scalar.
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        # Evaluate the neural network for each pair (x, y) at the fixed time t.
        u_flat = jax.vmap(lambda x, y: neural_net(params, x, y, t))(X_flat, Y_flat)
        # Reshape the flat predictions back to the 2D grid.
        return u_flat.reshape(X.shape)
    
    # Map predict_at_time over all times in ts.
    u_all = jax.vmap(predict_at_time)(ts)
    return np.array(u_all)