import pandas as pd
import numpy as np
from statistics import mean
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score



df = pd.read_csv("Clean_Dataset.csv")
df.drop(columns='Unnamed: 0', inplace=True)
df.drop(columns='flight', inplace=True)

column=['airline','source_city','departure_time','stops','arrival_time','destination_city','class']
df[column] = df[column].apply(LabelEncoder().fit_transform)

x = df.drop(['price'],axis=1)
y = df['price']

#print(df.head())



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)


r_squared_Ridge_list = []
r_squared_Lasso_list = []
alpha_val = np.linspace(0,1,11)

'''
for i in alpha_val:
    Ridge_model = Ridge(alpha=i)
    Lasso_model = Lasso(alpha=i)
    r_squared_Ridge_list.append(mean(cross_val_score(Ridge_model, x, y, cv=15)))
    r_squared_Lasso_list.append(mean(cross_val_score(Lasso_model, x, y, cv=15)))
'''

Ridge_model = Ridge(alpha=1)
Lasso_model = Lasso(alpha=1)
r_squared_Ridge = mean(cross_val_score(Ridge_model, x, y, cv=15))
r_squared_Lasso = mean(cross_val_score(Lasso_model, x, y, cv=15))

mse_Ridge= -mean(cross_val_score(Ridge_model, x, y, cv=15, scoring="neg_mean_squared_error"))
mse_Lasso= -mean(cross_val_score(Lasso_model, x, y, cv=15, scoring="neg_mean_squared_error" ))

mae_Ridge = -mean(cross_val_score(Ridge_model, x, y, cv=15, scoring="neg_median_absolute_error"))
mae_Lasso = -mean(cross_val_score(Lasso_model, x, y, cv=15, scoring="neg_median_absolute_error" ))


print("R2:", r_squared_Ridge, r_squared_Lasso)
print("MSE:", mse_Ridge, mse_Lasso)
print("MAE:", mae_Ridge, mae_Lasso)

