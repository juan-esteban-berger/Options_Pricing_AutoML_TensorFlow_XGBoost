# Pricing European Options with Google AutoML, TensorFlow, and XGBoost

This repository contains the code and dataset used in the research paper "Pricing European Options with Google AutoML, TensorFlow, and XGBoost" by Juan Esteban Berger, University of Notre Dame. The full research paper is available on arXiv: https://arxiv.org/abs/2307.00476.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Research Paper](#research-paper)
- [Conclusion](#conclusion)
- [Contact](#contact)

## Introduction
This research paper explores the application of machine learning techniques in pricing European options. The study provides a comparison of Google Cloud’s AutoML Regressor, TensorFlow Neural Networks, and XGBoost Gradient Boosting Decision Trees in terms of their performance against the traditional Black Scholes Model.

The XGBoost algorithm has been regarded as the gold standard for machine learning since its inception in 2014. Neural Networks have also showcased incredible abilities in learning extremely complicated patterns in data with large numbers of input variables. Most recently, however, machine learning practitioners have praised Google Cloud’s AutoML models for their ease of use and incredible accuracy. This study hopes to discover if TensorFlow deep learning models, XGBoost gradient boosted decision trees, and Google Cloud’s AutoML Regressor can outperform the Black Scholes model.

## Dataset
The XGBoost models outperformed both TensorFlow and Google’s AutoML Regressor. All of the machine learning models were able to outperform the Black-Scholes model. The XGBoost model with a max depth of ten had the lowest mean absolute error and mean absolute percentage error.

## Models
Six models were implemented in Python for pricing options:

Black-Scholes Model
Three-Layer Feed-Forward Neural Network
Five-Layer Feed-Forward Neural Network
Gradient Boosted Decision Tree with Max Depth of Five
Gradient Boosted Decision Tree with Max Depth of Ten
Google Cloud AutoML Regressor

The XGBoost models outperformed both TensorFlow and Google’s AutoML Regressor. All of the machine learning models were able to outperform the Black-Scholes model. The XGBoost model with a max depth of ten had the lowest mean absolute error and mean absolute percentage error. The most accurate model, XGBoost with a max depth of ten, can be found on Hugging Face at the following link:
[https://huggingface.co/models/XGBoost-European-Options](https://huggingface.co/models/XGBoost-European-Options).

## Results
| Model          | MAE    | MAPE   | Training (s) |
|----------------|--------|--------|--------------|
| XGBoost 10     | 0.8093 | 42.23  | 1917         |
| AutoML         | 1.0248 | 42.73  | 174420       |
| XGBoost 5      | 1.6362 | 187.02 | 971          |
| 5 Layer FFNN   | 4.6374 | 243.90 | 3288         |
| 3 Layer FFNN   | 8.8075 | 323.77 | 3066         |
| Black Scholes  | 8.0082 | 63.88  | NA           |

## Conclusion
The study concludes that machine learning models, especially XGBoost, are effective in pricing European options and outperform the Black-Scholes model. The models were able to learn necessary features from the dataset without being given implied volatility as a feature.

## Contact
Author: Juan Esteban Berger
Institution: University of Notre Dame
Email: jberger8@nd.edu
