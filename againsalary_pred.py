import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xticks
from matplotlib.pyplot import yticks 

df = pd.read_csv("Salary_Data.csv")
#print(df)

row,col = df.shape
print("No of rows = ",row)
print("No of cols = ",col)

df.describe()

df.isnull().sum()

df.isnull().sum()

df = df.dropna()

df.isna().sum()

df.columns

df.shape

df.info()

auto = df[['Age','Gender','Education Level','Job Title','Years of Experience','Salary']]

print(sns.pairplot(auto))

auto1 = df[['Age','Years of Experience','Salary']]
print(auto1.corr())

"""    Starting machine learning"""
plt.figure(figsize=(15,5))
sns.heatmap(auto1.corr(),annot=True)
plt.show()

gender = pd.get_dummies(auto['Gender'],drop_first = True)
print(gender)

auto = pd.concat([auto,gender],axis = 1)
print(auto)

education = pd.get_dummies(auto['Education Level'])
print(education)

auto = pd.concat([auto,education],axis=1)
print(auto)

t=len(set(auto['Job Title']))
print(t)

auto = auto.drop(['Gender' , 'Education Level','Job Title'],axis = 1)
print(auto)

plt.figure(figsize=(14,5))
sns.heatmap(auto.corr(),annot=True)
plt.show()

auto = auto.drop(['Male'],axis=1)
print(auto)

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(auto, train_size= 0.85,test_size=0.15, random_state =1)

print(df_train)

print(df_train.shape)

x_train = df_train[['Age','Years of Experience',"Bachelor's","Master's","PhD"]]

y_train = df_train['Salary']

# # for linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr_model = lr.fit(x_train, y_train)

# #for logistic regression
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg_model = lg.fit(x_train, y_train)

# # # # # # # #for polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
#--------------------------------------------------
X_poly = poly_reg.fit_transform(x_train)
# Y_poly = poly_reg.fit_transform(y_train)
pol_reg = LinearRegression()
model = pol_reg.fit(X_poly, y_train)
#--------------------------------------------------
#poly_reg.fit(x_train, y_train)

import pickle as pk
filename = 'model.pickle'
pk.dump(lr_model, open(filename, 'wb'))

data = auto.iloc[300:301]
data

actual_salary =data['Salary']
data =data.drop(['Salary'],axis = 1)
data

print("Predicted Salary",lr_model.predict(data))
print("Actual Salary",actual_salary)

print("Predicted Salary",lg_model.predict(data))
print("Actual Salary",actual_salary)

predicted_salary = pol_reg.predict(poly_reg.fit_transform(data))
print("Actual Salary",actual_salary)
print("predicted Salary",predicted_salary)


#train acc
print("Linear regression =",round(lr_model.score(x_train,y_train)*100,2))
print("Logistic regression =",lg_model.score(x_train, y_train)*100)
print("Polynomial regression = ", pol_reg.score(poly_reg.fit_transform(x_train),y_train)*100)

#test acc

test_data=df_test
y_test = test_data['Salary']
x_test = test_data.drop(['Salary'],axis =1)

print("Linear regression = ",round(lr_model.score(x_test,y_test)*100,2))
print("Logistic regression = ",lg_model.score(x_test,y_test)*100)
print("Polynomial regression = ",pol_reg.score(poly_reg.fit_transform(x_test),y_test)*100)

age = [int(input("Enter age = ")) for i in range(1)]
exp = [int(input("Enter exp = ")) for i in range(1)]
bach = [int(input("Enter 1 for bach otherwise 0 = ")) for i in range(1)]
master = [int(input("Enter 1 masters otherwise 0 = ")) for i in range(1)]
phd  = [int(input("Enter 1 phd otherwise 0 = ")) for i in range(1)]

df = pd.DataFrame({'Age':age,'Years of Experience':exp,"Bachelor's":bach,"Master's ":master,"PhD":phd})
print(df)

predicted_salary= lr_model.predict(df)
print(round(predicted_salary[0],0))

x_train

y_train

plt.scatter(y_train,lr_model.predict(x_train),color = 'red')

x = np.array([int(i) for i in range(50)])
y = x
plt.scatter(x,y,color= 'black')