# CST-305 Project 3 – Green’s Function and ODE with IVP

## Overview

This project implements analytical and numerical solutions for two second-order ODE initial value problems using:

- Green’s Function
- Undetermined Coefficients
- Variation of Parameters
- Numerical RKF (via SciPy’s solve_ivp)

The program computes and visualizes:

- Green’s function solutions
- Numerical solutions
- Homogeneous solutions
- Plots for both ODEs

This repository contains the full Python implementation, documentation, and supporting materials for CST-305 Project 3.

---

## ODEs Solved

### ODE 1
y'' + y = 4  
y(0) = 0, y'(0) = 0

### ODE 2
y'' + 4y = t^2  
y(0) = 0, y'(0) = 0

Each ODE is solved using:

- Green’s function convolution
- Undetermined coefficients
- Variation of parameters
- Numerical RKF (SciPy)

---

## Repository Structure

