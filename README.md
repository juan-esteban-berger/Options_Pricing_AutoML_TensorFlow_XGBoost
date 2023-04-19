# Options Pricing with Neural Networks and Gradient Boosters

## Introduction:
 The objective of the project was to study the performance of deep learning models and gradient boosters in pricing options using the Black-Scholes model's inputs. We used a neural network to learn from historical data. The project included the development of 9 models, various including LSTMs, Vanilla RNNs, Bidirectional LSTM, Gated Recurrent Unit, Decision Tree, XGBoost, and finally compared the results with those of the Binomial Options Pricing Model.

## Running the Models
The models are in the notebook titled '04_Models.ipynb'. The notebook contains the code for all the models. It is not recommended to train the models since they take lots of time to train and may require advanced hardware such as GPUs or TPUs. Furthermore, the XGBoost model was trained on a Dask Cluster using SaturnCloud's servers. All of the Neural Networks were trained on GPUs on Google Colab. The remainder of the code (predictions and results should work). Furthermore, the financial data was downloaded from Yahoo Finance api and the code for downloading the data can be found in the notebook called '01_Downloading_Data.ipynb'. The code for general data cleansing can be foudn in '02_Data_Cleansing.ipynb'. Finally, the code for exploratory data analysis can be found in '03_Exploratory_Data_Analysis.ipynb'.

## Neural Network Architecture:
In this project, we experimented with several neural network architectures, vanilla recurrent neural networks (RNN), vanilla long short-term memory networks (LSTM), 2 layer stacked LSTM, a 4 layer stacked LSTM and bidirectional LSTM networks, and gated recurrent unit networks (GRU). We used Keras, a high-level neural network API that runs on top of TensorFlow, to implement these architectures. Also a decision tree and a gradient booster were also trained with SciKit-Learn and XGBoost respectively.

In terms of MSE the best Neural Network was the Gated Recurrent Unit. The model's architecture consists of a single GRU layer with a variable number of units, defined as a hyperparameter, and an input shape of (20, 6), indicating a sequence of 20 timesteps, each with 6 features.

The dropout layer is then added to reduce overfitting and improve the generalization of the model. The dropout rate is also defined as a hyperparameter with a range from 0.1 to 0.5, in steps of 0.1.

The output layer is a dense layer with a single unit, as the model is designed for regression, predicting a single output value.

The model is compiled with the Adam optimizer and the mean squared error loss function. The learning rate is defined as a hyperparameter with a range from 1e-4 to 1e-2, using a logarithmic scale for sampling.

A RandomSearch tuner is then used to search for the best hyperparameters, with a maximum of 10 trials and 3 executions per trial. The objective is to minimize the validation loss, as defined in the build_model function.

After hyperparameter tuning, the best model and hyperparameters are obtained from the tuner, and the best model is trained on the training data for a single epoch, with a batch size of 128. The model is then evaluated on the test set using the evaluate method, which returns the test loss.

Finally, the best model is saved in an H5 file format for later use.

## Results

## Comments
| Model                | MSE          | MAE    | MAPE   |
|--------------------- |--------------|--------|--------|
| Gated Recurrent Unit | 3.130        | 1.120  | 2.140  |
| Bidirectional LSTM   | 5.120        | 1.885  | 1.885  |
| 2 Layer LSTM         | 5.765        | 1.405  | 2.370  |
| XGBoost              | 6.085        | 0.845  | 1.350  |
| Vanilla RNN          | 11.120       | 2.600  | 6.985  |
| Vanilla LSTM         | 34.245       | 2.970  | 7.475  |
| 4 Layer LSTM         |   50.885     | 4.710  | 17.585 |
| Decision Tree        | 60.055       | 4.000  | 5.595  |
| Binomial Model       | Nan          | Nan    | Nan    |

*The rest of the results can be found in the notebook
titled '05_Results.ipynb'.*

**Note: The Code for the Binomial Options Pricing Model contains some errors and still needs to be fixed.**

Commentary:
Some questions still remain regarding how the models could be improved. The first is whether it was the correct choice to use MSE as the loss function. The second is whether the out-of-the-money options and any other outliers should be dropped (all the options worth 0.01 cents). Also, if an option was given a negative price or zero, it would be useful to automatically set the options price to the minimum price of 0.01.


This report was inspired by and exapanded on the methods found in the paper "Option Pricing with Deep Learning" by Alexander Ke and Andrew Yang (https://cs230.stanford.edu/projects_fall_2019/reports/26260984.pdf).