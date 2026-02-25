"""
CST-305 Project 3
Green's Function and ODE with IVP
Author: John Berry
Packages: numpy, scipy, matplotlib
Description:
    - Solves two ODEs using Green's function
    - Solves homogeneous equations
    - Uses SciPy to solve the full ODE numerically
    - Plots all results
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

# ---------------------------------------------------------
# Green's Function for y'' + a y = f(t)
# ---------------------------------------------------------
def greens_function(t, tau, a):
    """Green's function G(t, tau) = (1/sqrt(a)) sin(sqrt(a)(t - tau)) H(t - tau)"""
    if t < tau:
        return 0
    w = np.sqrt(a)
    return (1 / w) * np.sin(w * (t - tau))


# ---------------------------------------------------------
# Compute Green's function solution
# ---------------------------------------------------------
def greens_solution(t, f, a):
    """Compute y(t) = ∫ G(t, τ) f(τ) dτ"""
    y_vals = []
    for ti in t:
        integrand = lambda tau: greens_function(ti, tau, a) * f(tau)
        val, _ = quad(integrand, 0, ti)
        y_vals.append(val)
    return np.array(y_vals)


# ---------------------------------------------------------
# ODE #1: y'' + y = 4
# ---------------------------------------------------------
def f1(t):
    return 4

def ode1(t, y):
    return [y[1], 4 - y[0]]


# ---------------------------------------------------------
# ODE #2: y'' + 4y = t^2
# ---------------------------------------------------------
def f2(t):
    return t**2

def ode2(t, y):
    return [y[1], t**2 - 4*y[0]]


# ---------------------------------------------------------
# Homogeneous solutions
# ---------------------------------------------------------
def homogeneous_solution(a, t):
    """Solve y'' + a y = 0 with y(0)=0, y'(0)=0 → trivial solution"""
    return np.zeros_like(t)

# ---------------------------------------------------------
# Homogeneous solutions (non-trivial basis functions)
# ---------------------------------------------------------
def homogeneous1(t):
    """Homogeneous solution for y'' + y = 0"""
    C1, C2 = 1, 1
    return C1 * np.cos(t) + C2 * np.sin(t)

def homogeneous2(t):
    """Homogeneous solution for y'' + 4y = 0"""
    C1, C2 = 1, 1
    return C1 * np.cos(2*t) + C2 * np.sin(2*t)

# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
if __name__ == "__main__":
    t = np.linspace(0, 10, 400)

    # Green's function solutions
    y1_green = greens_solution(t, f1, a=1)
    y2_green = greens_solution(t, f2, a=4)

    # Numerical SciPy solutions
    sol1 = solve_ivp(ode1, [0, 10], [0, 0], t_eval=t)
    sol2 = solve_ivp(ode2, [0, 10], [0, 0], t_eval=t)

    # -----------------------------------------------------
    # Plot Green's function + numerical solutions
    # -----------------------------------------------------
    plt.figure(figsize=(12, 8))

    # ODE 1
    plt.subplot(2, 1, 1)
    plt.plot(t, y1_green, label="Green's Function Solution", linewidth=2)
    plt.plot(t, sol1.y[0], '--', label="SciPy Numerical Solution")
    plt.title("ODE 1:  y'' + y = 4")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid(True)

    # ODE 2
    plt.subplot(2, 1, 2)
    plt.plot(t, y2_green, label="Green's Function Solution", linewidth=2)
    plt.plot(t, sol2.y[0], '--', label="SciPy Numerical Solution")
    plt.title("ODE 2:  y'' + 4y = t²")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # Plot homogeneous solutions
    # -----------------------------------------------------
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, homogeneous1(t), label="Homogeneous Solution y'' + y = 0", color='purple')
    plt.title("Homogeneous Solution for ODE 1")
    plt.xlabel("t")
    plt.ylabel("y_h(t)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, homogeneous2(t), label="Homogeneous Solution y'' + 4y = 0", color='green')
    plt.title("Homogeneous Solution for ODE 2")
    plt.xlabel("t")
    plt.ylabel("y_h(t)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
