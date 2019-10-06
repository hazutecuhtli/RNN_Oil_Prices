#*********************************************************************************************************
#Importing libraries
#*********************************************************************************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
#*********************************************************************************************************
#Defining functions
#*********************************************************************************************************

def data_split_per(data, train_per = None, val_per= None, test_per = None):
    
    '''
    Function to calculate or estimate the percentages to split a datasaet for the training, 
    valudation and testing of a model
    
    Input:
    data -->   Dataset to split
    train_per --> Percentage of the dataset used for the training of a model
    val_per --> Percentage of the dataset used for the validation of a model
    test_per --> Percentage of the dataset used for the testing of a model
    
    Output:
    train_per --> Percentage of the dataset used for the training of a model
    val_per --> Percentage of the dataset used for the validation of a model
    test_per --> Percentage of the dataset used for the testing of a model
    '''
    
    #calculating or fingind percentages
    if train_per==None and test_per==None:
        print('Error 1')
        return
    elif train_per!= None and test_per==None:
        if val_per==None and test_per==None:
            test_per = 1 - train_per
            if (train_per+test_per) != 1:
                print('Error 2')
                return
        elif val_per!= None and test_per==None:
            test_per = 1 - val_per - train_per
            print(test_per)
            if (test_per + val_per + train_per) != 1 or test_per< 0:
                print('Error 3')
                return
    elif test_per!=None and train_per==None:
        if train_per == None and val_per==None:
            train_per = 1 - test_per
            if (train_per + test_per) != 1:
                print('Error 4')
                return
        elif train_per==None and val_per!=None:
            train_per = 1 - val_per - test_per
            if (train_per + val_per + test_per) != 1:
                print('Error 5')
                return
            
    if val_per == None:
        val_per = 0
            
    return train_per, val_per, test_per

#----------------------------------------------------------------------------------------------------

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):

    '''
    Function that gives the shape for training, validation and testing datasets to be used create tensorflow
    datasets to be used for the training, validation and tesist of a model
    
    Input:
    dataset -->  Dataset to split
    target --> Labels
    start_index --> Index where the dataset to create starts, bases on the input used as dataset
    end_index --> Last index for the dataset to create
    history_size --> Number of elements used for each prediction, elements to consider for each prediction
    target_size --> Number of elements to predict, elements to consider in the future
    step --> Varibale used to downsample the input dataset
    single_step --> Varibale to select the downsample option
    
    Output:
    np.array(data) --> Numpy array in the sape required to create a tensorflow dataset for the features
    np.array(labels) --> Numpy array in the sape required to create a tensorflow dataset for the labels
    '''
    
    data = []
    labels = []
    Indexes = []

    #print(start_index, end_index)
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        
    #print(start_index, end_index, '\n')

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:     
            labels.append(target[i+target_size])
            Indexes.append(i+target_size)
        else:
            labels.append(target[i:i+target_size])
            Indexes.append(i)
            
    return np.array(data), np.array(labels), Indexes

#----------------------------------------------------------------------------------------------------

def plot_train_history(history, title):
    
    '''
    Function to plot the performance of the model while training
    
    Input:
    history -->  Performance history
    title --> Title to the figure to display
    '''    
    
    #Ploting the results
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure(figsize=(12, 6))

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.show()

#----------------------------------------------------------------------------------------------------

def evaluation(y, y_pred, df, dates, future_target, Scaler):
    
    '''
    Function to plot the performance of a trained model for preudcting oil prices
    
    Input:
    y --> tensor dataframe containing expected oil prices
    y_pred -->  tensor dataframe containing the predictions
    df --> oil prices dataframe
    future_target --> Number of elements to predict, elements to consider in the future
    Scaler --> Scikitlearn scaler used to normalize the dataset
    '''      
   
    dates = df.iloc[dates].index.to_list()
    
    #Plotting the results
    f, ax = plt.subplots(figsize=(12, 6))
    plt.locator_params(axis='x', nbins=5)
    ax.plot(dates, Scaler.inverse_transform(y[:-future_target,0]), 'r.-', label='Real oil prices')
    ax.plot(dates, Scaler.inverse_transform(y_pred[:-future_target,0]), 'b.-', label='Predicted oil prices')
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.round(np.linspace(xmin, xmax, 20), 2))
    plt.title('Forecasting oil prices (Testing dataset)')
    plt.ylabel('Oil barrel price (USD)')
    plt.legend()
    plt.xticks(rotation=90);
    plt.grid()

#----------------------------------------------------------------------------------------------------

def PlottingFprices(dfPriceDS):

    '''
    Function that plots the dataset used for the training, validation and modelling 
    
    Input:
    dfPriceDS -->  Pandas dataframe used
    '''    
    
    f, ax = plt.subplots(figsize=(12, 6))
    plt.locator_params(axis='x', nbins=5)
    ax.plot(dfPriceDS.DATE, dfPriceDS.DCOILBRENTEU)
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.round(np.linspace(xmin, xmax, 20), 2))
    plt.title('Dataset used for the forecasting')
    plt.ylabel('Barrel price (USD)')
    plt.xticks(rotation=90);
    plt.grid()
    
#*********************************************************************************************************
#End
#*********************************************************************************************************
