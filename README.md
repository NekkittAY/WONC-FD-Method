# WONC-FD-Method

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=%white)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

WONC- FD (Wavelet-Based Optimization and Numerical Computing for Fault Detection)

WONC-FD (Wavelet-Based Optimization and Numerical Computing for Fault Detection) is a method for detecting and classifying faults in time-series data using wavelet analysis and numerical optimization. This project aims to provide a robust and efficient method for identifying various types of errors within a signal.

<img width="800px" src="https://github.com/NekkittAY/WONC-FD-Method/blob/main/doc/Algorithm.png"/>

## Wavelet-Based Optimization and Numerical Computing for Fault Detection Method—Signal Fault Localization and Classification Algorithm

### Abstract

This study focuses on the development of the WONC-FD (Wavelet-Based Optimization and Numerical Computing for Fault Detection) algorithm for the accurate detection and categorization of faults in signals using wavelet analysis augmented with numerical methods. Fault detection is a key problem in areas related to seismic activity analysis, vibration assessment of industrial equipment, structural integrity control, and electrical grid reliability. In the proposed methodology, wavelet transform serves to accurately localize anomalies in the data, and optimization techniques are introduced to refine the classification based on minimizing the error function. This not only improves the accuracy of fault identification but also provides a better understanding of its nature.

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Fault Classification](#fault-classification)
- [Modules](#modules)
- [Algorithm](#algorithm)

## Introduction

The WONC-FD-Method provides a comprehensive toolkit for fault detection and classification in time-series data. By combining wavelet decomposition, numerical optimization, and machine learning principles, this library offers a versatile approach for analyzing signals with various error types. It includes tools for preprocessing signals, detecting faults, and classifying these faults based on their shape and characteristics.

## Technologies Used

This project utilizes the following technologies:

*   **Python:** The primary programming language used for development.
*   **NumPy:** For numerical computations and array manipulation.
*   **SciPy:** For scientific and technical computing, particularly signal processing (`scipy.signal`) and optimization (`scipy.optimize`).
*   **PyWavelets:** For performing wavelet transforms.
*   **joblib:** For parallelizing computations.
*   **Optuna:** For hyperparameter optimization in machine learning models.
*   **Pandas:** For data manipulation and analysis, used for creating the datasets.
*  **Matplotlib:** For visualizing results of the detection.

## Features

- **Wavelet Decomposition:** Utilizes wavelet transforms for multi-resolution signal analysis.
- **Fault Detection:** Implements peak detection algorithms to locate potential faults.
- **Fault Classification:** Provides various methods for classifying detected faults based on predefined error types.
- **Numerical Optimization:** Employs optimization algorithms to find the best fit error functions.
- **Parallel Processing:** Utilizes `joblib` for parallelizing computationally intensive tasks.
- **Optuna Integration:** Offers an alternative fault classification using `optuna` for hyperparameter optimization.
- **Error Signal Generation**: Includes a variety of predefined error signals, including Haar wavelets, exponential functions, and polynomial functions.
- **Preprocessing**: Includes tools for filtering and cleaning signals.

## Fault Classification

The library provides several classification methods:

- **WoncFD_MSE_classification**: Uses MSE to classify faults.
- **WoncFD_classification_brute**: A brute-force approach to classification.
- **WoncFD_classification**: Employs optimization to classify faults.
- **WoncFD_classification_parallel**: Parallelized version of the WoncFD_classification function.
- **WoncFD_optuna_classification_parallel**: Classification using optuna for hyperparameter tuning.

## Modules

The repository contains the following modules:

- **math_module.py**: Contains core mathematical functions such as MSE calculation, array padding and resizing, and different WONC-FD classification methods, and objective function for optimization.
- **optuna_math_module.py**: Implements fault classification using the optuna library for hyperparameter optimization.
- **parallel_math_module.py**: Provides a parallelized version of the WONC-FD classification function using joblib.
- **preprocessing_signals.py**: Includes functions for signal preprocessing such as fault detection, filtering, and visualization.
- **multierror_dataset.py**: Provides functions for creating synthetic datasets with different error types, including various error signal generation functions.

## Algorithm

<img width="800px" src="https://github.com/NekkittAY/WONC-FD-Method/blob/main/doc/WONC-FD_Algorithm.png"/>
