# Optimization Project: AIT 203 (Bike Sharing Demand & Cloud Resource Allocation)

This repository contains the complete documentation, source code, and analysis for the AIT 203 Optimization course. This project applies advanced optimization techniques to address two distinct challenges: high-performance predictive regression modeling and constrained elastic cloud resource allocation.

### Project Overview
This AIT 203 project applies advanced optimization to two challenges: predictive demand modeling and resource allocation. We used the Normal Equation to model bike rentals, achieving a 40% $R^2$ boost via feature engineering. 

[Image of polynomial regression curve]
 Additionally, we used the Penalty Method to solve cloud VM allocation, balancing costs and energy efficiency. 

---

## Project 1: Bike Sharing Demand Prediction (Question 1)
* **Topic:** Predictive Regression Modeling
* **Period:** Nov 2025 – Dec 2025
* **Technologies:** Python, NumPy, Matplotlib, Pandas

### Overview
This question focused on predicting hourly bike rental counts by implementing regression models from scratch using the Normal Equation, ensuring a deep understanding of the underlying numerical optimization.

### Key Contributions & Results
* **Numerical Computation:** Modeled daily bike-sharing demand using large-scale Kaggle datasets and NumPy for high-performance vectorized numerical computation.
* **Feature Engineering:** Refined predictive baseline accuracy by integrating higher-degree polynomials and complex interaction terms (e.g., `hour` × `workingday`, `temperature` × `humidity`) to capture non-linear usage patterns.
* **Performance Gains:** Boosted model performance by 40% in $R^{2}$ score (from 0.7012 to 0.9134) and reduced Mean Squared Error (MSE) by 27%.

---

## Project 2: Energy-Aware Elastic Cloud Resource Allocation (Question 2)
* **Topic:** Constrained Resource Allocation
* **Technologies:** Python, NumPy, Matplotlib

### Problem Definition
This question required allocating heterogeneous Virtual Machine (VM) types to satisfy specific resource demands (CPU, memory, I/O) while minimizing operational costs and non-linear energy consumption.

### Methodology
We constructed a smooth, differentiable objective function:
* **Nonlinear Energy Consumption:** Power-law scaling (exponent 1.4) to reflect CPU power dynamics.
* **Integer-Promoting Regularizer:** Utilized $\lambda \sum (1 - \cos(2\pi x_j))$ to guide allocations toward integer values.
* **Constraint Penalty:** A quadratic penalty function $P \sum \max(0, \text{shortfall})^2$ was applied to ensure hard resource demands were met.

### Implementation & Results
* **Algorithm:** Employed the **Penalty Method** with Gradient Descent.
* **Convergence:** The solver reached a stable, feasible allocation where resource shortfalls were reduced to zero, successfully meeting the target demand vector $D = (50, 180, 15, 800, 40)$.

---

## Repository Contents
* `Optimization_Project_Q1_Notebook.ipynb`: Manual implementation of the normal equation and polynomial feature expansion.
* `Optimization_Project_Q1_Report.pdf`: In-depth discussion on why interaction terms outperformed pure polynomial models.
* `Optimization_Project_Q2_Notebook.ipynb`: Implementation of the Penalty Method for cloud resource allocation.
* `Optimization_Project_Q2_Report.pdf`: Mathematical formulation of the energy-aware optimization model and final results interpretation.
* `train.csv`: Original bike-sharing training dataset.

---
*Project team: Aryan Sharma, Sutaria Parth Anandkumar (AIT 203)*
