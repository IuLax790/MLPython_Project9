import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import confusion_matrix,accuracy_score,mean_squared_error,r2_score
from sklearn import datasets
Movies = pd.read_csv("C:\Information_Science\movies.csv")
X = Movies.iloc[:,[3,4,5,6,7]].values
print(X)
y = Movies.iloc[:,-1].values

onehotencoder = OneHotEncoder(sparse=False)
Z = onehotencoder.fit_transform(X[:,[0]])
X=np.hstack((X[:,:0],Z)).astype('int')
Z = onehotencoder.fit_transform(X[:,[2]])
X=np.hstack((X[:,:2],Z)).astype('int')
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(y_pred)
import statsmodels.api as sm
import statsmodels.tools.tools as tl
X = tl.add_constant(X)

SL = 0.05
X_opt = X[:, [0,1,2,3,4]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

numVars = len(X_opt[0])
for i in range(0,numVars):
    regressor_OLS = sm.OLS(y,X_opt).fit()
    max_var = max(regressor_OLS.pvalues).astype(float)
    if max_var > SL:
        new_Num_Vars = len(X_opt[0])
        for j in range(0,new_Num_Vars):
            if (regressor_OLS.pvalues[j].astype(float)==max_var):
                X_opt = np.delete(X_opt,j,1)
print(regressor_OLS.summary())

print(y_pred)
print(regressor_OLS.pvalues)
print(regressor_OLS.params)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
plt.figure(figsize=(10,8))
plt.scatter(y_test,y_pred)
plt.xlabel('Variables')
plt.ylabel('Revenue in millions of $')
plt.title('Movie Revenue Prediction')

print(plt.show())


#When using only movie length variable


plt.xlabel('movie length')
plt.ylabel('Revenue in millions of $')
plt.title('Movie Revenue Prediction')
plt.scatter(Movies.run_time,Movies.Revenues_in_millions,color='red',marker='^')
print(plt.show())

#When using only movie rating variable


plt.xlabel('movie rating')
plt.ylabel('Revenue in millions of $')
plt.title('Movie Revenue Prediction')
plt.scatter(Movies.imdb_rating,Movies.Revenues_in_millions,color='red',marker='^')
print(plt.show())
