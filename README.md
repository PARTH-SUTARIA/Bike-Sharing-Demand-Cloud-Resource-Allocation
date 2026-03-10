# Optimization Project: AIT 203 (Bike Sharing Demand & Cloud Resource Allocation)

This repository contains the complete documentation, source code, and analysis for the AIT 203 Optimization course. This project applies advanced optimization techniques to address two distinct challenges: high-performance predictive regression modeling and constrained elastic cloud resource allocation.

---

## Project 1: Bike Sharing Demand Prediction
* **Period:** Nov 2025 – Dec 2025
* **Technologies:** Python, NumPy, Matplotlib, Pandas

### Overview
This project focused on predicting hourly bike rental counts by implementing regression models from scratch using the Normal Equation, avoiding off-the-shelf black-box libraries to ensure deep understanding of the underlying numerical optimization.

### Key Contributions & Results
* **Numerical Computation:** Modeled daily bike-sharing demand using large-scale Kaggle datasets and NumPy for high-performance vectorized numerical computation.
* **Feature Engineering:** Refined predictive baseline accuracy by integrating higher-degree polynomials and complex interaction terms (e.g., `hour` × `workingday`, `temperature` × `humidity`) to capture non-linear, real-world usage patterns.
* **Performance Gains:** Boosted model performance by 40% in $R^{2}$ score (improving from 0.7012 in the baseline to 0.9134) and reduced Mean Squared Error (MSE) by 27% through iterative feature engineering.
* **Rigorous Validation:** Implemented strict preprocessing, including one-hot encoding for categorical variables and feature standardization, to ensure no train/test leakage occurred during model training.



---

## Project 2: Energy-Aware Elastic Cloud Resource Allocation
* **Technologies:** Python, NumPy, Matplotlib

### Problem Definition
Cloud platforms must allocate heterogeneous Virtual Machine (VM) types to satisfy specific CPU, memory, I/O, storage, and network demands while simultaneously minimizing operational costs and non-linear energy consumption.

### Methodology
We constructed a smooth, differentiable objective function to balance multiple competing requirements:
* **Linear Cost:** Fixed monetary costs associated with individual VM units.
* **Nonlinear Energy Consumption:** Implemented power-law scaling (exponent 1.4) to accurately reflect real-world CPU power-to-usage dynamics.
* **Integer-Promoting Regularizer:** Utilized $\lambda \sum (1 - \cos(2\pi x_j))$ to nudge allocations toward integer values while maintaining the differentiability required for gradient-based solvers.
* **Constraint Penalty:** A quadratic penalty function $P \sum \max(0, \text{shortfall})^2$ was applied to ensure the system strictly meets hard resource demands.



### Implementation & Results
* **Algorithm:** Employed the **Penalty Method** with Gradient Descent, iteratively increasing the penalty parameter $P$ to drive the solution toward global feasibility.
* **Convergence:** The solver reached a stable, feasible allocation where resource shortfalls were effectively reduced to zero across all dimensions.
* **Analysis:** The model demonstrated scalability and stability, successfully selecting high-CPU VMs with optimal RAM-to-cost ratios to meet the demand vector $D = (50, 180, 15, 800, 40)$.

---

## Repository Contents
* `Optimization_Project_Q1_Notebook.ipynb`: Manual implementation of the normal equation and polynomial feature expansion.
* `Optimization_Project_Q1_Report.pdf`: In-depth discussion on why interaction terms outperformed pure polynomial models.
* `Optimization_Project_Q2_Notebook.ipynb`: Implementation of the Penalty Method for cloud resource allocation.
* `Optimization_Project_Q2_Report.pdf`: Mathematical formulation of the energy-aware optimization model and final results interpretation.
* `train.csv`: Original bike-sharing training dataset.

---
*Project team: Aryan Sharma, Sutaria Parth Anandkumar (AIT 203)*
