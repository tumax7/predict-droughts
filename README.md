# Predicting Drought
## Table of Contents
* [Intro](#intro)
* [Requirements](#requirements)
* [Setup](#setup)
* [Results](#results)

## Intro
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
In the end I managed to achieve: Test F1 of about 0.2, with the weights I included. In order to improve the model, I would consider switching the Neural Network type to Recurrent NN as the 'score' is not based on one instance, but on the historical data as well.


