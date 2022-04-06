# %% [markdown]
# # Electricity Usage Prediction

# %% [markdown]
# ## Introduction
# 
# In a world where sustainable development goals (SGD) are of paramount importance, it is crucial for stakeholders to predict the energy usage in advance so as to maximise energy utilisation. This project attempts to address this problem by regressing energy usage against weather conditions, a feature that should provide an abundance of information towards electricity consumption.

# %% [markdown]
# ## Preliminary Work

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from datetime import datetime
from sklearn.metrics import f1_score

# %%
energy = pd.read_csv('energy_data.csv')
weather = pd.read_csv('weather_data.csv')

# %% [markdown]
# The records in energy_data.csv are measured half-hourly. We first sum up the energy usage to get per day usage and merge it with weather data. 

# %%
def getdate1(datetime):
    return datetime.split()[0]
def gettime(datetime):
    return datetime.split()[1]
energy['Date'] = energy['Date & Time'].apply(getdate1)
energy['Time'] = energy['Date & Time'].apply(gettime)

# %%
daily_use = energy.groupby('Date').sum()['use [kW]']
daily_use

# %% [markdown]
# The records in weather_data.csv are hourly data. Therefore, we calculate the daily mean values to convert it to daily records.

# %%
def getdate2(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
weather['Date'] = weather['time'].apply(getdate2)

# %%
merged = pd.concat([weather.groupby('Date').mean(), daily_use], axis=1).drop('time', axis=1)
merged

# %% [markdown]
# We aim to predict the usage for each day in December using the weather data, so the training set will contain data from January to November, while the testing set will contain December only.

# %%
train = merged.loc[: '2014-11-30', :]
x_train = train.drop('use [kW]', axis=1)
y_train = train.loc[:, 'use [kW]']
test = merged.loc['2014-12-01': , :]
x_test = test.drop('use [kW]', axis=1)
y_test = test.loc[:, 'use [kW]']

# %% [markdown]
# We aim to predict the temperature for each day in December, so the training set will contain data from January to November, while the testing set will contain December only. The only dependent variable is temperature. Other numerical fields in the weather data is used as an independent variable.

# %%
def classification_threshold(temp):
    return 1 if temp >= 35 else 0
    
x_train_temp = x_train.drop('temperature', axis=1)
y_train_temp = x_train.loc[:, 'temperature'].apply(classification_threshold)
x_test_temp = x_test.drop('temperature', axis=1)
y_test_temp = x_test.loc[:, 'temperature'].apply(classification_threshold)

# %% [markdown]
# ## Predicting Energy Usage

# %% [markdown]
# With the data set ready, we will now train a linear regression model, and then predict energy usage for each day in the month of December using features from weather data.

# %% [markdown]
# Note the normalisation layer in the network. When creating a model with multiple features, the values of each feature should cover roughly the same range. For example, if one feature's range spans 500 to 100,000 and another feature's range spans 2 to 12, then the model will be difficult or impossible to train. Therefore, you should normalize features in a multi-feature model. (Google Machine Learning Crash Course)

# %%
normalizer = layers.experimental.preprocessing.Normalization(axis=None, input_shape=[10,])
normalizer.adapt(np.array(x_train))

# %%
energy_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1, dtype='float64'),])

energy_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')

history = energy_model.fit(
    x_train,
    y_train,
    verbose=0,
    epochs=100,
    validation_split = 0.2)

# %% [markdown]
# We define a function to plot the train and validation loss.

# %%
def plot_loss(history, lim):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, lim])
    plt.xlabel('Epoch')
    plt.ylabel('Error [kw]')
    plt.legend()
    plt.grid(True)
  
plot_loss(history, 1000)
plt.show()

# %% [markdown]
# After training the model, we evaluate its performance using root MSE loss on December data.

# %%
mse = energy_model.evaluate(x_test, y_test, verbose=0)
print(f'The root mean squared error = {np.sqrt(mse)}.')

# %% [markdown]
# We can also create a csv dump for the predicted values in December.

# %%
pred = energy_model.predict(x_test, verbose=0)
dump = pd.DataFrame(y_test)
dump['pred use [kW]'] = pred
dump.to_csv('lrdump.csv')

# %% [markdown]
# ## Temperature classification

# %% [markdown]
# Using only weather data we will classify if the temperature is high or low based on numerical fields.

# %%
normalizer = layers.experimental.preprocessing.Normalization(axis=None, input_shape=[9,])
normalizer.adapt(np.array(x_train_temp))

# %%
weather_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1, dtype='float64', activation=tf.sigmoid),])

weather_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"])

history = weather_model.fit(
    x_train_temp,
    y_train_temp,
    verbose=0,
    epochs=100,
    validation_split = 0.2)

# %%
plot_loss(history, 10)
plt.show()

# %%
pred = weather_model.predict(x_test_temp)
pred = [1 if x >= 0.5 else 0 for x in pred]
f1_score(y_test_temp, pred)
dump = pd.DataFrame(y_test_temp)
dump['Hot'] = pred
dump.to_csv('logisticdump.csv')

# %% [markdown]
# ## Energy usage data Analysis

# %% [markdown]
# We will now analyze how different devices are being used in different times of the day.

# %%
devices = ['AC', 'Furnace', 'Cellar Lights', 'Washer', 'First Floor lights', 'Dryer + egauge', 'Microwave (R)', 'Fridge (R)']

fig = plt.figure(dpi=100, figsize=(10, 20), tight_layout=True)
for i, device in enumerate(devices):
    ax = fig.add_subplot(10, 3, i + 1)
    energy.groupby('Time').mean()[f'{device} [kW]'].plot(title=f'Average {device} Usage [kW]', ax=ax)
    ax.set_ylabel('Usage')
    ax.set_xlabel('Time')

plt.show()

# %% [markdown]
# We can draw the conclusions from the charts above:
# - The washer is not only used during the day. However, it is true that its usage from 21:30 to 06:00 is much lower than that of other times in the day.
# - During the night, at around 7pm to 8pm, the AC is used the most.
# - Floor lights are used more at night than during the day, while cellar lights have peak usage in the morning and at night.
# - The fridge has peak usage in the morning and at night. This may be due to making breakfast and dinner require frequently retrieving food from the fridge.


