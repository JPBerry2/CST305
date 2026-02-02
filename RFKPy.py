import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------
#   ODE: y' = -y + ln(x)
# ---------------------------------------------------------

def f(x, y):
    return -y + math.log(x)

# ---------------------------------------------------------
#   Classical RK4
# ---------------------------------------------------------

def rk4_step(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h/2, y + k1/2)
    k3 = h * f(x + h/2, y + k2/2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_rk4(x0, y0, h, steps):
    x = x0
    y = y0
    xs = [x]
    ys = [y]

    for _ in range(steps):
        y = rk4_step(x, y, h)
        x += h
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

# ---------------------------------------------------------
#   MAIN PROGRAM — 1000 POINTS + 3 FIGURES
# ---------------------------------------------------------

if __name__ == "__main__":
    x0, y0 = 2.0, 1.0
    h = 0.3
    steps = 1000
    x_end = x0 + steps * h

    # ---- RK4 ----
    xs_rk4, ys_rk4 = solve_rk4(x0, y0, h, steps)

    # ---- SciPy RK45 ----
    sol = solve_ivp(
        fun=lambda t, y: f(t, y),
        t_span=(x0, x_end),
        y0=[y0],
        method="RK45",
        max_step=h
    )

    xs_scipy = sol.t
    ys_scipy = sol.y[0]

    # Interpolate SciPy results to RK4 grid
    ys_scipy_interp = np.interp(xs_rk4, xs_scipy, ys_scipy)

    # ---- Error ----
    error = np.abs(ys_rk4 - ys_scipy_interp)

    # ---------------------------------------------------------
    #   FIGURE 1 — RK4 Curve
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(xs_rk4, ys_rk4, label="RK4 (manual)", color="blue")
    plt.title("RK4 Solution (1000 Points)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # ---------------------------------------------------------
    #   FIGURE 2 — SciPy RK45 Curve
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(xs_rk4, ys_scipy_interp, label="SciPy RK45 (interpolated)", color="green")
    plt.title("SciPy RK45 Solution (1000 Points)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # ---------------------------------------------------------
    #   FIGURE 3 — Comparison Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(xs_rk4, ys_rk4, label="RK4 (manual)", linewidth=2)
    plt.plot(xs_rk4, ys_scipy_interp, label="SciPy RK45", linestyle="--")
    plt.title("RK4 vs SciPy RK45 (1000 Points)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # OPTIONAL: Error plot (uncomment if needed)
    # plt.figure(figsize=(12, 6))
    # plt.plot(xs_rk4, error, color="red")
    # plt.title("Absolute Error: |RK4 - SciPy RK45|")
    # plt.xlabel("x")
    # plt.ylabel("Error")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show(block=False)
    # plt.pause(0.1)

    print("Comparison complete.")
    print("Max error:", np.max(error))
    print("Final values:")
    print("  RK4:   ", ys_rk4[-1])
    print("  SciPy: ", ys_scipy_interp[-1])
    plt.show()