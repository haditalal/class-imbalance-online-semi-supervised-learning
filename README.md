# Online Semi-Supervised Neural Network under Class Imbalance

## Overview

This repository contains the experimental materials and results for the paper:

**Publication:** 2026 International Joint Conference on Neural Networks (IJCNN)
**Title:** *Position Paper: The Significance of Class Imbalance in Online Semi-Supervised Data Stream Learning*
**Authors:** Hadi Talal Jaafar Al-Kadhimi, Leandro L. Minku
**Emails:** [hxa568@student.bham.ac.uk](mailto:hxa568@student.bham.ac.uk), [l.l.minku@bham.ac.uk](mailto:l.l.minku@bham.ac.uk)

This work studies the impact of class imbalance in online semi-supervised learning using an Online Semi-Supervised Neural Network (OSNN).

---

## Repository Structure

* **Real Dataset experiments/**
  Experiments on real-world datasets, including training, multiple runs, and evaluation.

* **Synthetic DataSet experiments/**
  Experiments on synthetic datasets to analyze controlled imbalance scenarios.

* **Synthetic_Tuning/**
  Hyperparameter tuning experiments and results.

* **Results, Anova and plots/**
  Statistical analysis and performance visualization.

* **Logs/**
  Execution logs.

* **Supplementary_material.pdf**
  Additional explanation of *G-Mean delta for uniform and non-uniform cases*.

---

## Content

* Multiple runs (e.g., 30 runs) are provided for statistical reliability
* Results are stored in CSV and Parquet formats
* Python and MATLAB scripts are included for experiments and analysis

---

## Purpose

* Study the effect of class imbalance in online semi-supervised data streams
* Provide reproducible experiments
* Support statistical analysis using metrics such as G-Mean

---

## Requirements

* Python 3.x
* NumPy, Pandas, Matplotlib
* (Optional) MATLAB

