import matplotlib,numpy,pandas,scipy,sklearn
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split as TT_sk

path = os.path.join('datasets','housing','housing.csv')
data = pd.read_csv(path)
print("the top 5 is")
print(data.head())

print("the information is ")
print(data.info())

print("the tag is")
print(data["ocean_proximity"].value_counts())

data.hist(bins= 50,figsize = (20,15))
plt.show()

'''
a = [1,2,3,4,5,6,7,8,9]
b = a[:2]
print(b)
c = a[2:]
print(c)
d = a[-1:]
print(d) 
'''


# data snooping bias
# create test data 
def split_train_test(data,TestRatio):
    shuffled_indices = np.random.permutation(len(data))
    test_size = int(len(data)*TestRatio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    return data.iloc[test_indices],data.iloc[train_indices]

data_test,data_train = split_train_test(data,0.2)
data_test.hist(bins=50,figsize =(20,15))
plt.show()
data_train.hist(bins=50,figsize =(20,15))
plt.show()

# sklearn 

data_train,data_test = TT_sk(data,test_size = 0.2,random_state = 42)
data_test.hist(bins=50,figsize =(20,15))
plt.show()
data_train.hist(bins=50,figsize =(20,15))
plt.show()

# stratified sampling
data["income_cat"]= np.ceil(data["median_income"]/1.5)
data["income_cat"].where(data["income_cat"]<5,5.0,inplace = True)
data["income_cat"].hist(bins=10)
plt.show()

data_train,data_test = TT_sk(data,test_size = 0.2,random_state = 42)
print(data_test["income_cat"].value_counts()/len(data_test))




