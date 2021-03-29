import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import PowerTransformer

# To avoid extra output from tensoflow module
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score


# Importing_data
data_train = pd.read_csv('scaled_train_with_soil.csv', nrows=100000)
data_valid = pd.read_csv('scaled_valid_with_soil.csv')
data_test = pd.read_csv('scaled_test_with_soil.csv')

# Train Test Split
Y_train = data_train['score']
data_train.drop('score', inplace=True, axis=1)
Y_valid = data_valid['score']
data_valid.drop('score', inplace=True, axis=1)
Y_test = data_test['score']
data_test.drop('score', inplace=True, axis=1)

# Data processing
X_train = data_train.to_numpy()
X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = data_valid.to_numpy()
X_valid = X_valid.reshape(X_valid.shape[0], -1)
X_test = data_test.to_numpy()
X_test = X_test.reshape(X_test.shape[0], -1)

# Creating our bounded activation function for the final layer to avoid values higher than 5
bounded_relu = lambda x: tf.keras.activations.relu(x, max_value=5, threshold=0)


# Creating our model function
def fully_connected_model(input_shape):
    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.Input(input_shape)
    X = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer=tf.keras.initializers.glorot_normal(seed=1),
                              bias_initializer=tf.keras.initializers.Zeros())(X_input)
    X = tf.keras.layers.BatchNormalization(axis=1)(X)
    X = tf.keras.layers.Dropout(rate=0.5)(X)
    X = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer=tf.keras.initializers.glorot_normal(seed=1),
                              bias_initializer=tf.keras.initializers.Zeros())(X)
    X = tf.keras.layers.BatchNormalization(axis=1)(X)
    X = tf.keras.layers.Dropout(rate=0.5)(X)

    X = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer=tf.keras.initializers.glorot_normal(seed=1),
                              bias_initializer=tf.keras.initializers.Zeros())(X)
    X = tf.keras.layers.BatchNormalization(axis=1)(X)
    X = tf.keras.layers.Dropout(rate=0.8)(X)

    X = tf.keras.layers.Dense(1, activation=bounded_relu,
                              kernel_initializer=tf.keras.initializers.glorot_normal(seed=1),
                              bias_initializer=tf.keras.initializers.Zeros())(X)
    model = tf.keras.Model(inputs=X_input, outputs=X, name='fully_connected')

    return model


# Initializing model
model = fully_connected_model(input_shape=(X_train.shape[1],))
model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
model.load_weights('./model_weights.h5')
#model.fit(X_train, Y_train, epochs=100, batch_size=64, workers=-1)
loss_metrics = model.evaluate(X_valid, Y_valid, batch_size=64, workers=-1, verbose=0)
print('Validation Loss: {}'.format(loss_metrics[0]))
print('Validation MAE: {}'.format(loss_metrics[1]))

# Rounding our predictions and y test
y_valid_hat = pd.Series([round(value[0]) for value in model.predict(X_valid, workers=-1)])
y_valid = pd.Series([round(value) for value in Y_valid.to_numpy()])
ans = y_valid_hat.to_frame(name='Y_valid_hat').join(y_valid.to_frame(name='Y_valid'))
print('Validation F1 Score: {}'.format(f1_score(ans['Y_valid'], ans['Y_valid_hat'], average='macro')))
sns.heatmap(confusion_matrix(ans['Y_valid'], ans['Y_valid_hat'], normalize='all'))

plt.xlabel('Y Predictied')
plt.ylabel('Y True')
plt.title('Confusion Matix of Validaton Data')
plt.show()

loss_metrics = model.evaluate(X_test, Y_test, batch_size=64, workers=-1, verbose=0)
print('Test Loss: {}'.format(loss_metrics[0]))
print('Test MAE: {}'.format(loss_metrics[1]))
# Rounding our predictions and y test
y_test_hat = pd.Series([round(value[0]) for value in model.predict(X_test, workers=-1)])
y_test = pd.Series([round(value) for value in Y_test.to_numpy()])
ans = y_test_hat.to_frame(name='Y_test_hat').join(y_test.to_frame(name='Y_test'))
print('Test F1 Score: {}'.format(f1_score(ans['Y_test'], ans['Y_test_hat'], average='macro')))
sns.heatmap(confusion_matrix(ans['Y_test'], ans['Y_test_hat'], normalize='all'))
plt.xlabel('Y Predictied')
plt.ylabel('Y True')
plt.title('Confusion Matix of Test Data')
plt.show()
