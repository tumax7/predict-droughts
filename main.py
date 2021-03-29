import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import PowerTransformer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score


# Importing_data
data_train = pd.read_csv('train_with_soil.csv', nrows=10000)
data_test = pd.read_csv('validation_with_soil.csv', nrows=3000)

# Train Test Split
Y_train = data_train['score']
data_train.drop('score', inplace=True, axis=1)
# data_train = data_train.drop(data_train.query('score == 0').sample(frac=0.5).index)
Y_test = data_test['score']
data_test.drop('score', inplace=True, axis=1)

numerical_columns = [x for x in data_train.columns if 'SQ' not in x]
scaler = PowerTransformer()
data_train[numerical_columns] = scaler.fit_transform(data_train[numerical_columns])
data_test[numerical_columns] = scaler.transform(data_test[numerical_columns])

data_train[['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']] = data_train[
    ['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']].astype(str)
data_test[['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']] = data_test[
    ['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']].astype(str)

data_train = pd.get_dummies(data_train)
data_test = pd.get_dummies(data_test)
print(data_train.shape)
print(data_test.shape)

# Data processing
X_train = data_train.to_numpy()
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = data_test.to_numpy()
X_test = X_test.reshape(X_test.shape[0], -1)
bounded_relu = lambda x: tf.keras.activations.relu(x, max_value=5, threshold=0)


# Creating our model function
def fully_connected_model(input_shape):
    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.Input(input_shape)
    X = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer=tf.keras.initializers.glorot_normal(seed=1),
                              bias_initializer=tf.keras.initializers.Zeros())(X_input)
    X = tf.keras.layers.BatchNormalization(axis=1)(X)
    X = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer=tf.keras.initializers.glorot_normal(seed=1),
                              bias_initializer=tf.keras.initializers.Zeros())(X)
    X = tf.keras.layers.BatchNormalization(axis=1)(X)

    X = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer=tf.keras.initializers.glorot_normal(seed=1),
                              bias_initializer=tf.keras.initializers.Zeros())(X)
    X = tf.keras.layers.BatchNormalization(axis=1)(X)

    X = tf.keras.layers.Dense(1, activation=bounded_relu,
                              kernel_initializer=tf.keras.initializers.glorot_normal(seed=1),
                              bias_initializer=tf.keras.initializers.Zeros())(X)
    model = tf.keras.Model(inputs=X_input, outputs=X, name='fully_connected')

    return model


if __name__ == '__main__':
    model = fully_connected_model(input_shape=(X_train.shape[1],))
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    # model.load_weights('/home/tumax/PycharmProjects/Drought/model_weights')
    model.fit(X_train, Y_train, epochs=1, batch_size=64, workers=-1)
    # model.save_weights('/home/tumax/PycharmProjects/Drought/model_weights', overwrite=True)
    preds = model.evaluate(X_test, Y_test, batch_size=64, workers=-1)

    # Error Analysis

    # Rounding our predictions and y test
    y_hat = pd.Series([round(value[0]) for value in model.predict(X_test, workers=-1)])
    y_test = pd.Series([round(value) for value in Y_test.to_numpy()])
    #
    ans = y_hat.to_frame(name='Y_hat').join(y_test.to_frame(name='Y_test'))
    print('Test F1: {}'.format(f1_score(ans['Y_test'], ans['Y_hat'], average='macro')))
    sns.heatmap(confusion_matrix(ans['Y_test'], ans['Y_hat'], normalize='all'))
    plt.show()
