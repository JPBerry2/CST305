# CST-305 Project 4 – Data Integrity Degradation Simulator

## Overview

This project implements mathematical models for data degradation in digital storage systems using systems of first-order ordinary differential equations (ODEs). The project covers:

- Eigendecomposition and Stability Analysis
- Analytical Matrix Exponential ($e^{At}$)
- Numerical Integration (RK45)
- Conservation Laws in Closed Systems

The program computes and visualizes:

- Three-processor network data flow
- Two-processor closed-loop initial value problems (IVP)
- Phase portraits of data states
- Matrix exponential component grids

This repository contains the full Python implementation, documentation, and supporting materials for CST-305 Project 4.

---

## ODE Systems Solved

### Part 1: Three-Processor Network
Models data flowing between nodes A, B, and C with an exit to an external network.
- **Model**: $\mathbf{x}' = A\mathbf{x}$
- **Goal**: Identify eigenvalues to determine decay rates and system stability.

### Part 2: Two-Processor Closed Loop
Models a system where data is conserved and exchanged only between two processors.
- **Model**: $\mathbf{x}' = A\mathbf{x}$, $\mathbf{x}(0) = [1, -1]$
- **Analytical Solution**: Solved via the matrix method $\mathbf{x}(t) = e^{At}\mathbf{x}(0)$.

Each system is analyzed using:
- [cite_start]Input-Output rate balance equations [cite: 11]
- Matrix-based analytical solutions
- Numerical simulation (SciPy)
- Scientific visualization (Matplotlib)

---

## Repository Structure

- [cite_start]`data_integrity_ode_solver.py`: Main Python implementation containing the ODE solvers and plotting logic. [cite: 60, 61]
- [cite_start]`Documentation.pdf`: Project report covering the mathematical approach, algorithm flowcharts, and performance context. [cite: 50]
- [cite_start]`Results/`: Directory containing generated PNG plots (Part 1, Part 2, and Matrix Exponential components). [cite: 47, 57]
- [cite_start]`README.md`: This file, detailing project overview and installation. [cite: 59]

---

## Installation & Usage

1. **Install Dependencies**:
   ```bash
   pip install numpy scipy matplotlib
