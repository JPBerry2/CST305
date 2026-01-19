# CST-305 Project 1 – Visualizing ODEs with SciPy

## Overview
This project models CPU utilization in a computer system using a first-order
ordinary differential equation (ODE). The solution is computed numerically
using SciPy and visualized to analyze system performance over time.

## System Context
CPU utilization represents the fraction of time a processor is actively
executing tasks. Modeling this metric using ODEs allows continuous analysis
of system behavior under varying workload conditions.

## Mathematical Model
The CPU utilization model is defined as:

du/dt = λ − μu

where:
- u(t) is CPU utilization
- λ is the task arrival rate
- μ is the CPU service rate

## Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib

## Installation
Install required packages using:

pip install numpy scipy matplotlib

## Running the Program
Execute the program from the command line:

python cpu_utilization_ode.py

You will be prompted to enter:
- Task arrival rate (tasks per second)
- CPU service rate (tasks per second)
- Initial CPU utilization (0 to 1)

## Output
The program displays a 2D plot of CPU utilization versus time, illustrating
transient and steady-state system behavior.

## Author
John Berry
