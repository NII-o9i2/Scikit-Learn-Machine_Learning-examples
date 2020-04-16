import matplotlib,numpy,pandas,scipy,sklearn
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

path = os.path.join('datasets','housing','housing.csv')
data = pd.read_csv(path)

print(data.info())
from sklearn.model_selection import StratifiedShuffleSplit

data["income_cat"]= np.ceil(data["median_income"]/1.5)
data["income_cat"].where(data["income_cat"]<5,5.0,inplace = True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["income_cat"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

print(strat_train_set.info())
print(strat_test_set.info())

dataTrain = strat_train_set.copy()
dataTest = strat_test_set.copy()

for set in (dataTrain,dataTest):
    set.drop(["income_cat"],axis = 1, inplace = True)

print(dataTrain.info())
print(dataTest.info())
dataTrain.plot(kind = "scatter",x = "longitude",y="latitude",s=dataTrain["population"]/100, label="population",
               c="median_house_value", cmap=plt.get_cmap("jet"),alpha = 0.2,colorbar = True)
plt.legend()
plt.show()

corr_matrix = data.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False)) 

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(data[attributes], figsize=(12, 8))
#plt.show()

#data.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
#plt.show()

# data prepare

housing = strat_test_set.drop("median_house_value",axis=1)
housing_labels = strat_test_set["median_house_value"].copy()

# use sklearn wash data 
skdataTxt = housing["ocean_proximity"].copy()
skdata = housing.drop("ocean_proximity",axis =1)

from sklearn.impute import SimpleImputer
imputer_  = SimpleImputer(strategy = "median")
imputer_.fit(skdata)
X = imputer_.transform(skdata)
data_Tr=pd.DataFrame(X,columns=skdata.columns)
print('----------------------------')
print("data for train info is :")
print(data_Tr.info())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
print(skdataTxt)
house_encoded = encoder.fit_transform(skdataTxt)
print(house_encoded)
print(encoder.classes_)

# add user-defined pipeline function
from sklearn.base import BaseEstimator,TransformerMixin
roomid,bedroomid,populationid,householdsid = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.flag = True
    def fit(self,X,y=None):
        return self
    def transform(self,X,y =None):
        rooms_per_household = X[:,roomid]/X[:,householdsid]
        bedrooms_per_household = X[:,bedroomid]/X[:,householdsid]
        population_per_household = X[:,populationid]/X[:,householdsid]
        return np.c_[X,rooms_per_household,bedrooms_per_household,population_per_household]

class dataselector(BaseEstimator,TransformerMixin):
    def __init__(self,names):
        self.data = names
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return X[self.data].values
        
    


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
# num data pipeline
num_data_np = list(skdata)
print('----list data is -----')
print(num_data_np)
print(type(num_data_np))

num_pipeline = Pipeline([
    ('selector',dataselector(num_data_np)),
    ('Imputer',SimpleImputer(strategy='median')),
    ('add_attribute',CombinedAttributesAdder()),
    ('std_scalar',StandardScaler())
]
)
# cat data pipeline
skdataTxt = housing["ocean_proximity"].copy()
cat_np = ['ocean_proximity']

cat_pipeline = Pipeline([
    ('selector',dataselector(cat_np)),
    ('catencoder',LabelBinarizer())
])
full_pipeline = FeatureUnion(transformer_list=[
    ("num",num_pipeline),
    ("cat",cat_pipeline),
])

# comment out num pipeline 
'''
print('--------------------')
print('origin num data is :')
print(skdata.info())

num_data_np = num_pipeline.fit_transform(housing)
columnslist = list(skdata)
addlist = ['rooms_per_household','bedrooms_per_household','population_per_household']
columnslist += addlist
num_data = pd.DataFrame(num_data_np,columns=columnslist)
print('--------------------')
print('pipelined num data is :')
print(num_data.info())
'''



print('--------------------')

cat_data_np = cat_pipeline.fit_transform(housing)

#cat_data = pd.DataFrame(cat_data_np,columns=['ocean_proximity'])
la = ['ocean_proximity']
encoder1 = LabelBinarizer()
housing_cat = encoder1.fit_transform(housing[la]) 
print(housing_cat)


print('--------------------')
print('pipelined cat data is :')
print(cat_data.info())
#print(cat_data.info())






