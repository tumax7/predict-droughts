# predict-drought
This project is based on the data from Kaggle (https://www.kaggle.com/cdminix/us-drought-meteorological-data). The goal is to use various meteorological and soil data to predict drought level. I have chosen to use weekly data instead of given daily, this is due to the fact that the 'score' (target variable) is only given at weekly intervals. Also I have split the data set into data_processing and main to avoid RAM issues, as all the data is over 3GB. So if you want to run this files:

1. Download all the data from https://www.kaggle.com/cdminix/us-drought-meteorological-data.
2. Run the data processing.py script to convert to weekly, normalize the data and add extra soil features. (Requires more than 8GB of RAM)
3. Then run the main.py script to run the actual model.

In the end I managed to achieve: F1 of about 0.2. with the weights I included.
