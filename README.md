# Error Surface Analysis

A custom Python implementation of a vector-based Perceptron model built from scratch using NumPy and Pandas. This project trains a binary classifier on a normalized dataset (Car vs. SUV) and visualizes the error rate surface across a 2D parameter grid.

---

## Features

* **Custom Vector Perceptron**: Built strictly using NumPy and Pandas without high-level ML libraries like `scikit-learn` or `PyTorch`.
* ** normalized Dataset**: Binary classification on 2 normalized features (`ZeroToSixty` and `PowerHP`).
* **Grid Evaluation**: Evaluates model loss landscapes over a grid of parameters $(w_1, w_2)$ for 2D and 3D visualization.

---

## Requirements

* Python 3.x
* `numpy`[cite: 1]
* `pandas`[cite: 1]
* `matplotlib`[cite: 1]

Install dependencies via pip:

```bash
pip install numpy pandas matplotlib
