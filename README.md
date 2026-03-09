# Machine-learning-algorithms-for-regression-spectroscopy-redshift

This repository contains the implementation of machine learning models for estimating photometric redshifts of galaxies based on photometric features.

The project was developed as part of a university coursework in machine learning and astrophysical data analysis.

## Problem

Photometric redshift estimation (photo-z) is an important problem in observational cosmology. It allows estimating galaxy distances using photometric measurements instead of expensive spectroscopic observations. The photometric method is less resource-intensive than the spectroscopic method.

## Methods

The repository contains implementations of several machine learning approaches for regression of spectroscopic redshift:

- Gradient Boosting
- Multilayer Perceptron (Neural Network) 
- Random Forest 

## Data

The models are trained using photometric parameters such as galaxy colors derived from astronomical survey data.

Example features:

- mag_u
- mag_g
- mag_r
- mag_i 

## Technologies

- Python
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib 
