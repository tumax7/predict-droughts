import pandas as pd
from sklearn.preprocessing import PowerTransformer

# Reading the soild data
soil_data = pd.read_csv('soil_data.csv', index_col='fips')
# print(soil_data.shape)

# Setting fips as index to make it easier to add the soil data
train_with_soil = pd.read_csv('train_timeseries.csv', index_col='fips')

# Dropping and the date
train_with_soil.drop('date', inplace=True, axis=1)
# This converts this data to weekly data from daily as the score (target variable) is given only weekly
train_with_soil.dropna(inplace=True)
# Joining soil data
train_with_soil[soil_data.columns] = soil_data
# Shuffling dataset to as we are going to be using mini-batches
train_with_soil = train_with_soil.sample(frac=1)

# Getting our numerical values and scaling them
numerical_columns = [x for x in train_with_soil.columns if 'SQ' not in x]
numerical_columns.remove('score')
scaler = PowerTransformer()
train_with_soil[numerical_columns] = scaler.fit_transform(train_with_soil[numerical_columns])

# Transfroming out categorical variables into dummy variables
train_with_soil[['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']] = train_with_soil[
    ['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']].astype(str)
train_with_soil = pd.get_dummies(train_with_soil)
train_with_soil.to_csv('scaled_train_with_soil.csv', index=False)
print(train_with_soil.shape)


# Repeating for validation and test sets
valid = pd.read_csv('validation_timeseries.csv', index_col='fips')
valid.drop('date', inplace=True, axis=1)
valid_with_soil = valid.copy()
valid_with_soil[soil_data.columns] = soil_data
valid_with_soil.dropna(inplace=True)
valid_with_soil = valid_with_soil.sample(frac=1)
valid_with_soil.to_csv('validation_with_soil.csv', index=False)

valid_with_soil = pd.read_csv('validation_with_soil.csv')

valid_with_soil[numerical_columns] = scaler.transform(valid_with_soil[numerical_columns])

valid_with_soil[['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']] = valid_with_soil[
    ['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']].astype(str)


valid_with_soil = pd.get_dummies(valid_with_soil)
valid_with_soil.to_csv('scaled_valid_with_soil.csv', index=False)
print(valid_with_soil.shape)


test = pd.read_csv('test_timeseries.csv', index_col='fips')
test.drop('date', inplace=True, axis=1)
test_with_soil = test.copy()
test_with_soil[soil_data.columns] = soil_data
test_with_soil.dropna(inplace=True)
test_with_soil = test_with_soil.sample(frac=1)
test_with_soil.to_csv('test_with_soil.csv', index=False)

test_with_soil = pd.read_csv('test_with_soil.csv')

test_with_soil[numerical_columns] = scaler.transform(test_with_soil[numerical_columns])

test_with_soil[['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']] = test_with_soil[
    ['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']].astype(str)


test_with_soil = pd.get_dummies(test_with_soil)
test_with_soil.to_csv('scaled_test_with_soil.csv', index=False)
print(test_with_soil.shape)