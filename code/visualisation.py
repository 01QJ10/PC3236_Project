# visualization.py
"""
Visualization routines for the 2D heat equation.
Provides functions for plotting heatmaps, contours, and animations.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def plot_heatmap(u: np.ndarray, x: np.ndarray, y: np.ndarray, title="Solution"):
    """
    Plot a heatmap for a 2D solution u(x,y).
    u: 2D array of shape (ny, nx)
    """
    plt.figure(figsize=(6,5))
    plt.imshow(u, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(label="Temperature")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()

def plot_contour(u: np.ndarray, x: np.ndarray, y: np.ndarray, title="Contour Plot"):
    """
    Plot a contour plot for a 2D solution u(x,y).
    """
    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=(6,5))
    cp = plt.contourf(X, Y, u, cmap='hot')
    plt.colorbar(cp)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()

def animate_solution(u_seq: list, x: np.ndarray, y: np.ndarray, filename="heat_2d.gif"):
    """
    Animate a sequence of 2D solutions (time evolution).
    u_seq: list of 2D arrays (each shape (ny, nx)).
    """
    fig, ax = plt.subplots()
    cax = ax.imshow(u_seq[0], extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='hot', aspect='auto')
    fig.colorbar(cax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("t = 0.0")
    
    def update(frame):
        cax.set_data(u_seq[frame])
        title.set_text(f"t = {frame}")
        return [cax, title]
    
    ani = animation.FuncAnimation(fig, update, frames=len(u_seq), interval=100, blit=True)
    ani.save(filename, writer='imagemagick')
    plt.close()