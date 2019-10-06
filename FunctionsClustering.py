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

def KmeanProportion(PredictedGroups, Num_Groups, Label):
    
    '''
    Function that display the percentage of elements related to the found clusters
    
    Input:
    PredictedGroups --> Dataset clustered
    Num_Groups --> Number of groups where the data was clustered
    Label --> Label for the plotting
    '''
    
    # Compare the proportion of data in each cluster 
    Counts = []
    
    for n in range(Num_Groups):

        Counts.append(sum(PredictedGroups == n)/len(PredictedGroups))

    d= {'Cluster':range(1,Num_Groups+1), 'General': Counts}
    General = pd.DataFrame(d)

    General.plot(x='Cluster', y = [Label], kind='bar', figsize=(15,5), color=['royalblue'])
    plt.xticks(rotation=0);
    plt.ylabel('Population proportion');
    plt.title('Proportion of the total population by cluster')
    plt.grid()
    
#----------------------------------------------------------------------------------------------------

def PCATest(df, n_scor, rango, layout=(1,1), Figsize=(15,13)):

    '''
    Function used to plot the variance related with the PCA components used
    
    Input:
    df -->  Dataframe to cluster
    n_scor --> Number of scores used for the elbow method
    rango --> Amount of elements used to cluster
    layout --> Layout used tor the subplot
    Figsize --> Size of the figure to plot
    '''       
    
    fig, ax = plt.subplots(layout[0], layout[1], figsize=Figsize)
    
    for i, m in enumerate(n_scor):

        df = df[0:rango, :]

        interia = []

        for center in range(1,m+1):
    
            kmeans = KMeans(center, random_state=0)
    
            model = kmeans.fit(df)

            interia.append(model.inertia_)
    
        centers = list(range(1,m+1))

        plt.subplot(layout[0], layout[1], i+1) 
        plt.plot(centers,interia)
        plt.title('Kmeans (PCA components)')
        plt.xlabel('Centers');
        plt.ylabel('Average distance');
        plt.grid()
        i+=1
        
    plt.subplots_adjust(top=1, bottom=0.08, left=0.10, right=0.95, hspace=0.3, wspace=0.35);

#----------------------------------------------------------------------------------------------------

def DimensionReduction(df, components=None, n_scor=30):

    '''
    Function that display the percentage of elements related to the found clusters
    
    Input:
    df --> Dataset to be clustered
    components --> Number of PCA components to keep 
    n_scor --> Number of scores to implement for the elbow method
    
    Output:
    dfPCA --> Clustered dataframe
    '''    
    
    #In case no PCA components are selected, all are used
    if components == None:
        components = df.shape[1]
        
    # Normalizing the data
    columns = df.columns
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])

    # Applying PCA to the data.
    columns = df.columns
    pca = PCA(n_components=components)
    pca.fit(df[columns])    

    dfPCA = pca.transform(df[columns])
    VarRat = pca.explained_variance_ratio_

    Suma = 0
    VarRatAcum = []

    for var in pca.explained_variance_ratio_:
        Suma+=var
        VarRatAcum.append(Suma)

    PCAvariance(VarRat, VarRatAcum)

    PCATest(dfPCA, [n_scor], dfPCA.shape[0], layout=(1,1), Figsize=(7,4))
    
    return dfPCA

#----------------------------------------------------------------------------------------------------

def CorrePrice(df, dfClus):
    
    '''
    Function used to correlated the oil production and consumption features with the oil prices
    
    Input:
    df --> Dataset containing the oil prices
    dfClus --> Clustered oil production and consumption features
        
    Output:
    CorrDF --> Data frame which columns are composed by the amount of correlation that exist between the found clusters and
               the data prices, for different years. 
    '''    
    
    #Renaming the clusters
    Names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    Groups = []

    for item in dfClus:
        if item not in Groups:
            Groups.append(item)

    Clusters = []

    index = 0
    for item in dfClus:
        Clusters.append(Names[Groups.index(item)])

    df['Cluster'] = Clusters
    
    
    #Creating a dictionary of the pearson correlation for each of the oil features with the oil prices
    Newdf = {}

    for item in df['Cluster'].unique():
        Name = (item + ' (' + '-'.join(df[df.Cluster == item].index[[0,-1]].astype(str).tolist()) + ')')
        Newdf[Name] = abs(df[df.Cluster == item].corr().Price)
        
    #Creating the dataframe to be returned
    CorrDF = pd.DataFrame(Newdf)
    CorrDF.drop(['Price'], inplace = True)
    
    return df, CorrDF

#----------------------------------------------------------------------------------------------------

def PCAvariance(VarRat, VarRatAcum):
    
    '''
    Function used to plot the variance related with the PCA components used
    
    Input:
    VarRat --> Variance ratio related with each pf the PCA components
    VarRatAcum --> Accumlated variance for the PCA componets
    '''       
    
    #Plotting variances
    fig, ax1 = plt.subplots(figsize=(15,5))

    plt.rcParams.update({'font.size': 14})

    ax1.bar(range(len(VarRat)),VarRat, color='c')
    ax1.set_xlabel('PCA components')
    # Make the y-label, ticks and tick labels match the line color.
    ax1.set_ylabel('Variance ratio', color='k')
    ax1.tick_params('y', colors='k')
    plt.grid(axis='x')

    ax2 = ax1.twinx()
    ax2.plot(VarRatAcum, 'r-*')
    ax2.set_ylabel('Commulative Variance ratio', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.grid(axis='y')
    plt.show()

#----------------------------------------------------------------------------------------------------    
   
def PlotPriceClusters(df, dfPriceA, dfPriceB):
    
    '''
    Function used to plot oil prices corresponding to two differnt data sets, one realted with BP where the prices are represented by year,
    and one related to Freed, where the prices are presented day by day.
    
    Input:
    dfPriceA --> BP yearly oil prices dataframe
    VarRatAcum --> Freed daily oil prices dataframe
    '''           
    
    fig, ax = plt.subplots(len(df.Cluster.unique()[3:]), 2, figsize=(18,9))

    for row, cluster in zip(ax, df.Cluster.unique()[3:]):
        for i, col in enumerate(row):
            if i == 0:
                col.plot(dfPriceA['$ money of the day'].loc[df[df.Cluster == cluster].index], 'b')
                plt.xticks(rotation=90);
                maximun = dfPriceA['$ 2018'].max()
                col.set_ylim([0,maximun])
            if i == 1:
                index1 = dfPriceB.index[dfPriceB.DATE.str.contains(str(df.index[df.Cluster == cluster][0])) == True][0]
                index2 = dfPriceB.index[dfPriceB.DATE.str.contains(str(df.index[df.Cluster == cluster][-1])) == True][-1]
                col.plot(range(len(dfPriceB.DATE.iloc[index1:index2].tolist())), dfPriceB.DCOILBRENTEU.iloc[index1:index2], 'r')
                maximun = dfPriceB.DCOILBRENTEU.max()
                col.set_ylim([0,maximun])

                Step = col.get_xticks()[2]-col.get_xticks()[1]
                plt.sca(col)
                NewXticks = index1 +col.get_xticks()
                NewTicks = []
                for temp in range(len(NewXticks)):
                    if NewXticks[temp] > dfPriceB.shape[0]:
                        NewTicks.append(np.nan)
                    else:
                        NewTicks.append(dfPriceB.DATE[NewXticks[temp]][0:4])

                plt.xticks(col.get_xticks(),NewTicks)
                col.set_xlim([col.get_xticks()[1],col.get_xticks()[-2]])

    Groups = ['D', 'D', 'E', 'E', 'F', 'F']
    for i, ax in enumerate(fig.axes):
        if i in [1,3,5]:
            rot = 90
            plt.ylabel('Oil Price (Brent)')
            plt.title('BP data - Yearly - Group ' + Groups[i])
        else:
            rot = 90
            plt.title('FRED data - Daily - Group ' + Groups[i])
        plt.sca(ax)
        plt.xticks(rotation=rot)
        plt.grid()

    plt.subplots_adjust(top=1, bottom=0.08, left=0.10, right=0.95, hspace=0.7, wspace=0.1);
    plt.show()
    
#*********************************************************************************************************
#End
#*********************************************************************************************************
