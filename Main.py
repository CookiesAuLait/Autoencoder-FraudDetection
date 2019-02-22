#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:41:48 2019

@author: hml1204
"""

import sys  

#reload(sys)  
#sys.setdefaultencoding('utf8')

import pandas as pd
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
#%matplotlib inline
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
#### Plotly ####
import plotly
import plotly.plotly as py
import plotly.offline as pyo
import plotly.figure_factory as ff
from plotly.tools import FigureFactory as FF, make_subplots
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import cufflinks as cf
init_notebook_mode(connected = True)
cf.go_offline()
pyo.offline.init_notebook_mode()

#### Deep Learning Libraries ####
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from ann_visualizer.visualize import ann_viz
#### ####
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from IPython.display import display, Math, Latex

# For coloring
import random
r = lambda: random.randint(0,255)

# Data loading
df = pd.read_csv('GL_DATA_SAMPLE.csv', header = 0)

# Dropping a few non-numeric columns
df = df.drop(['EffectiveDate', 'EnteredDate', 'PreparerUserID', 'AccountType', 'AccountClass', 'Account TypeClass', 'JEIdentifierID', 'JENumberID', 'GLAccountNumber', 'GLAccountName'], axis=1)

# Data Split
RANDOM_SEED = 101
X_train, X_test = train_test_split(df, test_size = 0.2, random_state = RANDOM_SEED)

# Data (Further) Pre-processing: We need scaler since we have un-normalised amount.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Number of Neurons in each layer [22, 6, 3, 2, 3, 6, 22]
input_dim = X_train.shape[1] # 22 features
encoding_dim = 6

# Autoencoders Structure
input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation = "tanh", activity_regularizer = regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation = "tanh")(encoder)
encoder = Dense(int(2), activation = "tanh")(encoder)
decoder = Dense(int(encoding_dim / 2), activation = "tanh")(encoder)
decoder = Dense(int(encoding_dim), activation = "tanh")(decoder)
decoder = Dense(input_dim, activation = "tanh")(decoder)
autoencoder = Model(inputs = input_layer, outputs = decoder)
autoencoder.summary()

# Training the model
nb_epoch = 100
batch_size = 50
autoencoder.compile(optimizer = 'adam', loss = 'mse')

t_ini = datetime.datetime.now()
history = autoencoder.fit(X_train_scaled, X_train_scaled, epochs = nb_epoch,
                          batch_size = batch_size,
                          shuffle = True,
                          validation_split = 0.1,
                          verbose = 0)

t_fin = datetime.datetime.now()
print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))

df_history = pd.DataFrame(history.history)

# Convergence
def train_validation_loss(df_history):
    trace = []
    for label, loss in zip(['Train', 'Validation'], ['loss', 'val_loss']):
        trace0 = {'type' : 'scatter',
                   'x' : df_history.index.tolist(),
                   'y' : df_history[loss].tolist(),
                   'name' : label,
                   'mode' : 'lines'}
        trace.append(trace0)
    data = trace
    layout = {'title' : 'Model train-vs-validation loss',
               'titlefont' : {'size' : 30},
               'xaxis' : {'title' : '<b> Epochs', 'titlefont':{'size' : 25}},
               'yaxis' : {'title' : '<b> Loss', 'titlefont':{'size' : 25}}
               }
    fig = Figure(data = data, layout = layout)
    # iplot does not work in Spyder!
    return pyo.plot(fig, filename = "train_validation_loss.html")

train_validation_loss(df_history)

# Predictions
predictions = autoencoder.predict(X_test_scaled)
error = X_test_scaled - predictions
mse = np.mean(np.power(X_test_scaled - predictions, 2), axis = 1)
df_msre = pd.DataFrame({'Mean_Squared_Reconstruction_Error': mse}, index = X_test.index)
df_mare = pd.DataFrame(error, index = X_test.index, columns = df.keys().tolist()) # Don't use iloc/loc which returns only the first numerical row!
print(df_msre.describe())

# Outliers
outliers = df_msre.index[df_msre.Mean_Squared_Reconstruction_Error > 0.1].tolist() 
print(outliers)
df_msre.to_csv("Mean_Squared_Reconstruction_Error_reporting.csv")

# For computing MAE by Features
df_mare.to_csv("Mean_Absolute_Reconstruction_Error_reporting.csv")

# Locating the outliers
df_test = pd.DataFrame(X_test, index = X_test.index, columns = df.keys().tolist())
df_test.to_csv("Test_Set.csv")

# Plotting
def scatter_plot(df, col):
    trace = dict( 
                x = df.index.tolist(),
                y = df[col].tolist(), 
                mode = 'markers',
                marker = dict(color = 'red', size = 10)
            )
    data = [trace]
    layout = {'title' : 'Scatter plot of {}'.format(col),
              'titlefont' : {'size' : 30},
              'xaxis' : {'title' : 'Data Index', 'titlefont' : {'size' : 20}},
              'yaxis' : {'title' : col, 'titlefont' : {'size' : 20}},
              'hovermode' : 'closest'}
    fig = dict(data = data, layout = layout)
    return pyo.plot(fig, filename = "Scatter_plot_of_{}.html".format(col))

for col in df.columns.tolist():
    scatter_plot(df, col)


def scatter_multiple_plot(df, target, col):
    trace = dict( 
                x = df.index.tolist(),
                y = df[df[col] == 1][target].tolist(), 
                mode = 'markers',
                marker = dict(color = '#{:02x}{:02x}{:02x}'.format(r(), r(), r()), size = 10),
                name = col
            )
    return trace

databasket = []
for col in df.columns.tolist():
    if col != "Amount":
        trace = scatter_multiple_plot(df, "Amount", col)
        databasket.append(trace)
layout = {'title' : 'Scatter plot of Amount under different category',
          'titlefont' : {'size' : 30},
          'xaxis' : {'title' : 'Data Index', 'titlefont' : {'size' : 20}},
          'yaxis' : {'title' : "Amount", 'titlefont' : {'size' : 20}},
          'hovermode' : 'closest'}
fig = dict(data = databasket, layout = layout)
pyo.plot(fig, filename = "Scatter_plot_of_Amount_under_different_category.html")
