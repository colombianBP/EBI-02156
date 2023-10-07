import pandas as pd
import numpy as np 
import seaborn as sn
import matplotlib.pyplot as plt
import time

ti=time.time()

# read and see red wine data
redwine=pd.read_csv('winequality/winequality-red.csv',sep=';')

# read and see white wine data
whitewine=pd.read_csv('winequality/winequality-white.csv',sep=';')

# store color as a new variable and join
redwine['color']=[1]*redwine.shape[0]
whitewine['color']=[0]*whitewine.shape[0]
data=pd.concat([redwine,whitewine])
data=data.reset_index(drop=True)

# get basic data statistics
data.describe().to_csv('Plots/EDA/SummaryStatistics.csv')

#Make some plots showing red vs white wine 
b=50
datr=data[data.color==1].drop(['color'],axis=1).reset_index(drop=True)
datw=data[data.color==0].drop(['color'],axis=1).reset_index(drop=True)

for i in [1,2,6]:
    plt.hist(datr[datr.columns[i]],bins=b,color='red',alpha=0.5,density=True)
    plt.hist(datw[datw.columns[i]],bins=b,color='green',alpha=0.5,density=True)
    plt.xlabel(datw.columns[i].capitalize())
    plt.ylabel('Normaliced frequency')
    plt.title('Frequency of '+datw.columns[i]+' in red and white wine')
    plt.savefig('Plots/EDA/FrequencyHist'+datw.columns[i].title().replace(' ','')+'.pdf',bbox_inches='tight',dpi=200)
    plt.clf()

# make and save a correlation matrix and its plot; first for both
corr_matrix = data.drop(['quality'],axis=1).corr()
corr_matrix = corr_matrix.where(np.tril(np.ones(corr_matrix.shape),-1).astype(bool))
corr_matrix = corr_matrix.iloc[1:,:-1].round(2)
corr_matrix.to_csv('Plots/Correlation/CorrelationMatrixBoth.csv')
plt.figure(figsize = (7,6.8))
plt.title('Correlation matrix for Both wine types')
sn.heatmap(corr_matrix, annot=True)
plt.savefig('Plots/Correlation/CorrelationMatrixBoth.pdf',bbox_inches='tight',dpi=120)
plt.clf()

#Now red wine
corr_matrixr = datr.drop(['quality'],axis=1).corr()
corr_matrixr = corr_matrixr.where(np.tril(np.ones(corr_matrixr.shape),-1).astype(bool))
corr_matrixr = corr_matrixr.iloc[1:,:-1].round(2)
corr_matrixr.to_csv('Plots/Correlation/CorrelationMatrixRed.csv')
plt.figure(figsize = (7,6.8))
plt.title('Correlation matrix for red wine')
sn.heatmap(corr_matrixr, annot=True)
plt.savefig('Plots/Correlation/CorrelationMatrixRed.pdf',bbox_inches='tight',dpi=120)
plt.clf()

#and white wine
corr_matrixw = datw.drop(['quality'],axis=1).corr()
corr_matrixw = corr_matrixw.where(np.tril(np.ones(corr_matrixw.shape),-1).astype(bool))
corr_matrixw = corr_matrixw.iloc[1:,:-1].round(2)
corr_matrixw.to_csv('Plots/Correlation/CorrelationMatrixWhite.csv')
plt.figure(figsize = (7,6.8))
plt.title('Correlation matrix for white wine')
sn.heatmap(corr_matrixw, annot=True)
plt.savefig('Plots/Correlation/CorrelationMatrixWhite.pdf',bbox_inches='tight',dpi=120)
plt.clf()

#Compare white and red matices
comp_matrix=np.absolute(np.array(corr_matrixr)-np.array(corr_matrixw))#/np.maximum(np.absolute(np.array(corr_matrixr)),np.absolute(np.array(corr_matrixw)))
# a choise was made here to not normalize feel free to uncoment the line fragment above to do so
comp_matrix=comp_matrix.round(2)
comp_matrix=pd.DataFrame(comp_matrix)
comp_matrix.index=corr_matrix.index[0:-1]
comp_matrix.columns=corr_matrix.columns[0:-1]
comp_matrix.to_csv('Plots/Correlation/CorrelationMatrixComparison.csv')
plt.suptitle('Comparison of correlation matrices for white VS red wine')
plt.title('As the absolute value of the difference')
sn.heatmap(comp_matrix, annot=True)
plt.savefig('Plots/Correlation/CorrelationMatrixComparison.pdf',bbox_inches='tight',dpi=120)
plt.clf()

# Write variance unflation factors to evaluate feature engeniering results
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Normalization of variables
data=data.reset_index(drop=True)
for column in data.columns:
    data[column] = (data[column] - data[column].mean()) / data[column].std()
data=data.reset_index(drop=True)

#VIF analysis without feature engeniering
vifs=[variance_inflation_factor(data.drop(['quality'],axis=1).values, i) for i in range(len(data.drop(['quality'],axis=1).columns))]
pd.DataFrame({'Columns':data.drop(['quality'],axis=1).columns,'VIF':vifs}).to_csv('Plots/VarianceInflation/RedAndWhiteWine.csv',index=False)

datr=data[data.color>0].reset_index(drop=True).drop(['color'],axis=1)
vifs=[variance_inflation_factor(datr.drop(['quality'],axis=1).values, i) for i in range(len(datr.drop(['quality'],axis=1).columns))]
pd.DataFrame({'Columns':datr.drop(['quality'],axis=1).columns,'VIF':vifs}).to_csv('Plots/VarianceInflation/redWine.csv',index=False)

datw=data[data.color<0].reset_index(drop=True).drop(['color'],axis=1)
vifs=[variance_inflation_factor(datw.drop(['quality'],axis=1).values, i) for i in range(len(datw.drop(['quality'],axis=1).columns))]
pd.DataFrame({'Columns':datw.drop(['quality'],axis=1).columns,'VIF':vifs}).to_csv('Plots/VarianceInflation/whiteWine.csv',index=False)

#Feature engineering 
drops=['density','residual sugar','alcohol']
data['fermentation']=[np.mean(data.loc[i,drops]) for i in range(data.shape[0])]
data=data.drop(drops,axis=1)

drops=['free sulfur dioxide','total sulfur dioxide']
data['sulfur dioxide']=[np.mean(data.loc[i,drops]) for i in range(data.shape[0])]
data=data.drop(drops,axis=1)

#VIF analysis with feature engeniering
vifs=[variance_inflation_factor(data.drop(['quality'],axis=1).values, i) for i in range(len(data.drop(['quality'],axis=1).columns))]
pd.DataFrame({'Columns':data.drop(['quality'],axis=1).columns,'VIF':vifs}).to_csv('Plots/VarianceInflation/TransformedRedAndWhiteWine.csv',index=False)

datr=data[data.color>0].reset_index(drop=True).drop(['color'],axis=1)
vifs=[variance_inflation_factor(datr.drop(['quality'],axis=1).values, i) for i in range(len(datr.drop(['quality'],axis=1).columns))]
pd.DataFrame({'Columns':datr.drop(['quality'],axis=1).columns,'VIF':vifs}).to_csv('Plots/VarianceInflation/TransformedRedWine.csv',index=False)

datw=data[data.color<0].reset_index(drop=True).drop(['color'],axis=1)
vifs=[variance_inflation_factor(datw.drop(['quality'],axis=1).values, i) for i in range(len(datw.drop(['quality'],axis=1).columns))]
pd.DataFrame({'Columns':datw.drop(['quality'],axis=1).columns,'VIF':vifs}).to_csv('Plots/VarianceInflation/TransformedWhiteWine.csv',index=False)

#Model construction, training and statistics
modelb = sm.OLS(data.quality,np.array(data.drop(['quality'],axis=1)))
modelb = modelb.fit()
summ=modelb.summary2()
t=summ.tables[0]
t.to_csv('Plots/ModelStatistics/bothWinesGeneralStatistics.csv',index=False,header=False)
t=summ.tables[1]
t.index=data.drop(['quality'],axis=1).columns
t.to_csv('Plots/ModelStatistics/bothWinesFeaturesStatistics.csv')

modelw = sm.OLS(np.array(datw.quality),np.array(datw.drop(['quality'],axis=1)))
modelw = modelw.fit()
summ=modelw.summary2()
t=summ.tables[0]
t.to_csv('Plots/ModelStatistics/whiteWineGeneralStatistics.csv',index=False,header=False)
t=summ.tables[1]
t.index=data.drop(['quality','color'],axis=1).columns
t.to_csv('Plots/ModelStatistics/whiteWineFeaturesStatistics.csv')

modelr = sm.OLS(np.array(datr.quality),np.array(datr.drop(['quality'],axis=1)))
modelr = modelr.fit()
summ=modelr.summary2()
t=summ.tables[0]
t.to_csv('Plots/ModelStatistics/redWineGeneralStatistics.csv',index=False,header=False)
t=summ.tables[1]
t.index=data.drop(['quality','color'],axis=1).columns
t.to_csv('Plots/ModelStatistics/redWineFeaturesStatistics.csv')

#Print time to run
tf=time.time()
print('Analysis took '+str(np.round(tf-ti,2))+' seconds to run')