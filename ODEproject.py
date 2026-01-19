"""
CST-305 Project 1 – Visualizing ODEs with SciPy

Programmers: John Berry
Course: CST-305 Operating Systems and System Programming
Packages Used: NumPy, SciPy, Matplotlib
Approach:
This program models CPU utilization as a first-order ordinary differential
equation (ODE). The ODE is solved numerically using SciPy, and the solution
is visualized to analyze system performance over time.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -----------------------------
# ODE Definition
# -----------------------------
def cpu_utilization(t, u, arrival_rate, service_rate):
    """
    Computes the rate of change of CPU utilization.

    Parameters:
    t : float
        Time (seconds)
    u : float
        CPU utilization (0 to 1)
    arrival_rate : float
        Incoming task rate (tasks per second)
    service_rate : float
        Task processing rate (tasks per second)

    Returns:
    du/dt : float
        Rate of change of CPU utilization
    """
    return arrival_rate - service_rate * u


# -----------------------------
# User Input
# -----------------------------
print("CPU Utilization ODE Model")
arrival_rate = float(input("Enter task arrival rate (tasks/sec): "))
service_rate = float(input("Enter CPU service rate (tasks/sec): "))
initial_utilization = float(input("Enter initial CPU utilization (0 to 1): "))

# Time interval for simulation
start_time = 0
end_time = 20
time_points = np.linspace(start_time, end_time, 300)

# -----------------------------
# Solve the ODE
# -----------------------------
solution = solve_ivp(
    cpu_utilization,
    (start_time, end_time),
    [initial_utilization],
    t_eval=time_points,
    args=(arrival_rate, service_rate)
)

# -----------------------------
# Visualization
# -----------------------------
plt.plot(solution.t, solution.y[0])
plt.xlabel("Time (seconds)")
plt.ylabel("CPU Utilization")
plt.title("CPU Utilization Over Time")
plt.grid(True)
plt.show()
