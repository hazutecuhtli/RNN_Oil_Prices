#*********************************************************************************************************
#Importing libraries
#*********************************************************************************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from country_list import countries_for_language
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#*********************************************************************************************************
#Defining functions
#*********************************************************************************************************
    
def CleaningProdConsump(dfInput, Label, SpeCols = None):

    '''
    Function used to clean the dataset containing oil prices from the the BP dataset
    
    Input:
    dfInput --> Dataset to clean
    Label --> Label used to define the values, the unit of measure
    SpecCols --> Columns with names of countries to correct
    
    Output:
    df --> Cleaned dataframe
    '''  
    
    #Transposing the data
    df = dfInput.copy()
    df = df.transpose()
    
    #Reseting the indexes 
    df.reset_index(drop=True, inplace=True)
    
    #Dropping unnecesary rows
    df = df.drop(df.index[-5:])
    
    #Dropping columns with undefined names
    df.drop(df.columns[df.iloc[0].isnull() == True], axis=1, inplace=True)
    
    #Renaming the columns of the dataframe
    df.columns = df.loc[0]
    df.drop([0], inplace=True)
    
    #Renaming the indexes of the dataframe
    df[Label] = df[Label].astype(int)
    
    dictIndex = {}
    for i, index in enumerate(df.index):
        dictIndex[index] = df[Label].iloc[i]
    
    df.rename(index=dictIndex, inplace=True)

    #Changin selected columns for the USSR
    if df.index.min() < 1908: 

        if SpeCols != None:
            dfTemp = df[SpeCols].copy()
            
        for col in SpeCols[:-1]:
            dfTemp[col] = dfTemp[col]/dfTemp[SpeCols[-1]]
            
        Percentage = dfTemp[SpeCols[:-1]].loc[1985:1995].mean().to_list()
        
        for i, col in enumerate(SpeCols[:-1]):
            df[col].loc[:1984] = Percentage[i]*df['Total CIS'].loc[:1984]
    
    #Removing uneccesary columns
    countries = dict(countries_for_language('en'))
    countries = list(countries.values())
    countries += ['US', 'Russian Federation']
    
    columns2remove = []
    for column in df.columns:
        if column not in countries:
            columns2remove.append(column)
            
    df.drop(columns2remove, axis=1, inplace=True)
    
    #Sorting the columns by name
    df = df.reindex(sorted(df.columns), axis=1)
    
    #Converting to floats
    columns = df.columns[df.dtypes == 'object']
    df[columns] = df[columns].astype(float)
    
    #Returning the transformer dataframe
    return df

#----------------------------------------------------------------------------------------------------    

def CleaningPrice(dfInput):

    '''
    Function used to clean the dataset containing oil prices from the the BP dataset
    
    Input:
    dfInput --> Dataset to clean
    
    Output:
    df --> Cleaned dataframe
    '''  
    
    #Creating a copy of the dataframe
    df = dfInput.copy()
    
    #Dropping undefined names for the columns
    df.drop(df.columns[df.iloc[2].isnull() == True], axis=1, inplace=True)
    
    #Renaming the dataframe columns names
    df.columns = df.iloc[2].to_list()
    
    #Reseting the indexes 
    df = df.drop(df.index[0:3])
    df.reset_index(drop=True, inplace=True)
    
    #Removing unecessary rows
    df = df.drop(df.index[-5:])
    
    #Renaming the dataframe indexes
    dictIndex = {}
    for i, index in enumerate(df.index):
        dictIndex[index] = df.Year.iloc[i] 
        
    df.rename(index=dictIndex, inplace=True)
    
    #Dropping unnecesary columns
    df = df.drop(['Year'], axis=1)
    
    #Converting to floats
    columns = df.columns[df.dtypes == 'object']
    df[columns] = df[columns].astype(float)
    
    #Returning the transformer dataframe
    
    return df

#----------------------------------------------------------------------------------------------------    

def NaNValuesDistribution(df1, df2):
    
    '''
    Function that display the number of undefined values on the input dataframes columns
    
    Input:
    df1 --> Daframe that represent specific data
    df2 --> Daframe that represent specific data
    '''
    
    fig, ax = plt.subplots(2, 1, figsize=(18,9))
    ax1 = plt.subplot(2,1,1) 
    sns.barplot(df1.columns,df1.isnull().sum().values);
    plt.xticks(rotation=90);
    plt.ylabel('NaN values (Production)');
    plt.xlabel(' ');
    ax1.set_ylim([0,df1.shape[0]])
    plt.grid()
    ax2 = plt.subplot(2,1,2)
    ax2 = sns.barplot(df2.columns,df2.isnull().sum().values);
    plt.xticks(rotation=90);
    plt.ylabel('NaN values (Consumption)');
    plt.xlabel(' ');
    ax2.set_ylim([0,df2.shape[0]])
    plt.grid()

    plt.subplots_adjust(top=1, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35);

#----------------------------------------------------------------------------------------------------    

def CleanPriceBrent(df):

    '''
    Function used to clean the dataset containing oil prices from the Fred dataset
    
    Input:
    df --> Dataset to clean
    
    Output:
    df --> Cleaned dataframe
    '''
    
    for i, item in enumerate(df.DCOILBRENTEU):
        if len(item) == 1:
            df.DCOILBRENTEU.iloc[i] = -1    
            
    df.DCOILBRENTEU = df.DCOILBRENTEU.astype(float).replace(-1, np.nan).tolist()
    
    for indice in df.index[df.DCOILBRENTEU.isnull()]:
        if indice >= 1 and indice < df.DCOILBRENTEU.index[-1]:
            df.DCOILBRENTEU.iloc[indice] = df.DCOILBRENTEU.iloc[[indice-1,indice+1]].mean()
        elif indice == 0:
            df.DCOILBRENTEU.iloc[indice] = df.DCOILBRENTEU.iloc[indice+1]
        elif indice == df.DCOILBRENTEU.index[-1]:
            df.DCOILBRENTEU.iloc[indice] = df.DCOILBRENTEU.iloc[-2]    
            
    return df

#*********************************************************************************************************
#End
#*********************************************************************************************************
