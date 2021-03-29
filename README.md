# Predicting Drought
## Table of Contents
* [Info](#info)
* [Requirements](#requirements)
* [Setup](#setup)
* [Results](#results)

## Info
This project is based on the data from Kaggle (https://www.kaggle.com/cdminix/us-drought-meteorological-data). The goal is to use various meteorological and soil data to predict drought level. I have chosen to use weekly data instead of given daily, this is due to the fact that the 'score' (target variable) is only given at weekly intervals. Also, I have split the data set into data_processing and main to avoid RAM issues, as all the data is over 3 GB. I have decided to go for a regression model over typical multi-class classification, as the scores provided in the train dataset where continuous. Also, I used 3 layer Neural Network with 1000 units in each layer. I implemented Batch Normalization to improve the performance of the model and reduce overfitting. 


## Requirements
-  python <= 3.8
-  pandas <= 1.2.0
-  seaborn <= 0.11.1
-  tensorflow <= 2.4.1
-  sklearn <= 0.24.1

## Setup
1. Download all the data from https://www.kaggle.com/cdminix/us-drought-meteorological-data.
2. Run the data processing.py script to convert to weekly, normalize the data and add extra soil features. (Requires more than 8 GB of RAM)
3. Then run the main.py script to run the actual model.

## Results
These are the results (with minibatch=64) I achieved with the weights included.

-  Validation Loss: 0.9437624216079712
-  Validation MAE: 0.6700882911682129
-  Validation F1 Score: 0.19843542239945847

![Validation Confusion Matrix](./images/Validation%20Matrix.png)

-  Test Loss: 0.8818807601928711
-  Test MAE: 0.6324158310890198
-  Test F1 Score: 0.1942177377204588

![Test Confusion Matrix](./images/Test%20Matrix.png)

We can see that the errors are smaller in the Test set compared to the Validation Set, unlike the F1 score, which is lower. This means that the model predicts more of the majority class ('0') in the test set.
In order to improve the model, I would consider switching the Neural Network type to Recurrent NN as the 'score' is not based on one instance, but on the historical data as well.

