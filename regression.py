import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
dataset = dataset.dropna() #drops nan value
#print(dataset)
dataset =dataset[['Years of Experience','Salary']]
#print(dataset)

x = dataset.iloc[:,0:1].values
#print(x)

y = dataset.iloc[:,1:].values
#print(y)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,train_size=0.8,random_state = 0)
#print(train_test_split(x,y,test_size=1/3,random_state =0))

#print(len(y_train),len(x_train))
#print(len(y_test),len(x_test))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#print(x_test)

#print(y_test)

#print(dataset[dataset['Years of Experience'] == 2])

np.float64(72657.69)

for i in x_test:
    print(i,model.predict([i]))


plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,model.predict(x_train),color = 'blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,model.predict(x_train),color= 'blue')
plt.show()

plt.scatter(y_train,model.predict(x_train),color = 'pink')
plt.show()

d = pd.DataFrame({'exp':x_test.flatten(),'pred':model.predict(x_test).flatten()}) 
d.corr