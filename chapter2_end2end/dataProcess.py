import matplotlib,numpy,pandas,scipy,sklearn
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

path = os.path.join('datasets','housing','housing.csv')
data = pd.read_csv(path)
data["income_cat"]= np.ceil(data["median_income"]/1.5)
data["income_cat"].where(data["income_cat"]<5,5.0,inplace = True)
print(data["income_cat"].value_counts())
print(data.info())
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = 42)
co = np.arange(0,17)

for train_index, test_index in split.split(data, data["income_cat"]):
    train_data = data.loc[train_index]
    test_data = data.loc[test_index]
train_data = train_data.drop("income_cat",axis =1)
test_data = test_data.drop("income_cat",axis =1)

print(train_data.info())


train_label = train_data["median_house_value"]
train_data  = train_data.drop("median_house_value",axis= 1)


# add user-defined pipeline function
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
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
        
class MyLabelBinarizer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.encoder = LabelBinarizer()
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
# num data pipeline

num_data_np = list(train_data)
num_data_np.remove("ocean_proximity")
print(num_data_np)


num_pipeline = Pipeline([
    ('selector',dataselector(num_data_np)),
    ('Imputer',SimpleImputer(strategy='median')),
    ('add_attribute',CombinedAttributesAdder()),
    ('std_scalar',StandardScaler())
]
)
# cat data pipeline
skdataTxt = train_data["ocean_proximity"].copy()
cat_np = ['ocean_proximity']

cat_pipeline = Pipeline([
    ('selector',dataselector(cat_np)),
    ('catencoder',MyLabelBinarizer())
])
full_pipeline = FeatureUnion(transformer_list=[
    ("num",num_pipeline),
    ("cat",cat_pipeline),
])

# comment out num pipeline 

print('+++++\t the origin data info')
print(train_data.info())
print(train_data.head())
data_list = full_pipeline.fit_transform(train_data)
print(data_list[0,:])

 
# *** use linear regression ***

data_np = np.array(data_list)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(data_list,train_label)
somedata = train_data
somelabels = train_label
somdata_prepares = full_pipeline.fit_transform(somedata)
predict = lin_reg.predict(somdata_prepares)
print(predict[:5])
print(list(somelabels[:5]))


#  rmse = 60000+
from sklearn.metrics import mean_squared_error

lin_rmse = mean_squared_error(predict,somelabels)
print(np.sqrt(lin_rmse))


# *** use decision tree ***

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_list,train_label)

somedata = train_data
somelabels = train_label
somdata_prepares = full_pipeline.fit_transform(somedata)
predict = tree_reg.predict(somdata_prepares)
print(predict[:5])
print(list(somelabels[:5]))
# rmse = 0.
lin_rmse = mean_squared_error(predict,somelabels)
print(np.sqrt(lin_rmse))

# k-fold cross validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg,somdata_prepares,somelabels,scoring="neg_mean_squared_error",cv=10)
rmse_score = np.sqrt(-scores)

print(rmse_score)
print("mean is :\t",rmse_score.mean())
print('std is :\t',rmse_score.std())

# save model 
from sklearn.externals import joblib

joblib.dump(tree_reg,"my_tree.m")

# grid - search
from sklearn.model_selection import GridSearchCV

garam_grid = [
    {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}
    ]
from sklearn.ensemble import RandomForestRegressor
forest_reg =RandomForestRegressor()
gridserch = GridSearchCV(forest_reg,garam_grid,cv=5,scoring='neg_mean_squared_error')
gridserch.fit(data_list,train_label)
print(gridserch.best_params_)
bestreg = gridserch.best_estimator_
joblib.dump(bestreg,"bestreg.m")

print(bestreg)
'''
print(strat_train_set_np.shape)

full_data_np = full_pipeline.fit_transform(housing)

print('--------------------')
print('full pipelined cat data is :')
print(full_data_np.shape)
'''

# leaning model 
# linear regression
'''
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(strat_train_set_np,housing_labels)

somedata = housing.iloc[:40]
data_some = full_pipeline.fit_transform(somedata)
print(data_some.shape)
#print("predict data is :\t")
#print(lin_reg.predict(data_some))
print("label is :\t")
#print(housing_labels.iloc[:5])

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

# data.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
# plt.show()
# data prepare

housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# use sklearn wash data 
skdataTxt = housing["ocean_proximity"].copy()
skdata = housing.drop("ocean_proximity",axis =1)


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
'''



