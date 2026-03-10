# Optimization Project: AIT 203 (Bike Sharing Demand & Cloud Resource Allocation)

This repository contains the complete documentation, source code, and analysis for the AIT 203 Optimization course. This project applies advanced optimization techniques to address two distinct challenges: high-performance predictive regression modeling and constrained elastic cloud resource allocation.

---

## Project 1: Bike Sharing Demand Prediction (Question 1)
* **Topic:** Predictive Regression Modeling
* **Period:** Nov 2025 – Dec 2025
* **Technologies:** Python, NumPy, Matplotlib, Pandas

### Overview
This question focused on predicting hourly bike rental counts by implementing regression models from scratch using the Normal Equation, ensuring a deep understanding of the underlying numerical optimization without reliance on high-level libraries.

### Methodology
We modeled the relationship between features $X$ and target $y$ using $y = X\theta + \epsilon$. To find optimal weights, we derived the **Normal Equation**: $\theta = (X^{T}X)^{-1}X^{T}y$. Polynomial expansion $f(x) = \beta_0 + \beta_1 x + \dots$ and interaction terms were used to capture non-linear patterns.

Models were evaluated using:
* **Mean Squared Error (MSE):** Measures the average squared difference between estimates and actual values: $MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$.
* **$R^{2}$ Score:** Explains the proportion of variance in the dependent variable: $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$.

### Implementation
We performed data cleaning, one-hot encoding, and standardization. The weight vector $\theta$ was computed manually using NumPy’s linear algebra module to solve the system of equations.

### Results
Through iterative feature engineering, we boosted model performance by 40% in $R^{2}$ score (from 0.7012 to 0.9134) and reduced MSE by 27% compared to the baseline.

---

## Project 2: Energy-Aware Elastic Cloud Resource Allocation (Question 2)
* **Topic:** Constrained Resource Allocation
* **Technologies:** Python, NumPy, Matplotlib

### Overview
This question required allocating heterogeneous Virtual Machine (VM) types to satisfy specific resource demands while minimizing operational costs and non-linear energy consumption.

### Methodology
We minimized a differentiable objective $F(x) = C(x) + E(x) + R(x) + P(x)$, where:
* **Cost:** $C(x) = \sum c_j x_j$ (Fixed costs).
* **Energy:** $E(x) = \sum a_j x_j^{1.4}$ (Power-law scaling).
* **Regularizer:** $R(x) = \lambda \sum (1 - \cos(2\pi x_j))$ (Integer-promoting).
* **Penalty:** $P(x) = \frac{1}{2} \mu \sum (\max(0, d_i - \sum A_{ij} x_j))^2$ (Constraint enforcement).


### Implementation
We employed the **Penalty Method** with Gradient Descent. The penalty parameter $\mu$ was iteratively increased to drive the solution toward global feasibility while maintaining differentiability.

### Results
The solver reached a stable, feasible allocation where resource shortfalls were reduced to zero, successfully meeting the target demand vector $D = (50, 180, 15, 800, 40)$.

---

## Repository Contents
* `Optimization_Project_Q1_Notebook.ipynb`: Manual implementation of the normal equation and polynomial feature expansion.
* `Optimization_Project_Q1_Report.pdf`: In-depth discussion on why interaction terms outperformed pure polynomial models.
* `Optimization_Project_Q2_Notebook.ipynb`: Implementation of the Penalty Method for cloud resource allocation.
* `Optimization_Project_Q2_Report.pdf`: Mathematical formulation of the energy-aware optimization model and results interpretation.
* `train.csv`: Original bike-sharing training dataset.

---
*Project team: Aryan Sharma, Sutaria Parth Anandkumar (AIT 203)*
