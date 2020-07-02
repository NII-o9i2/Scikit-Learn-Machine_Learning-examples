import pandas as pd 
import os
#datapath = os.path.join('data/fashion-mnist_train.csv')
datapath = os.path.join('chapter3_classify/data/fashion-mnist_train.csv')
data  = pd.read_csv(datapath)

#print(data.info())

import matplotlib 
import matplotlib.pyplot as plt 


pixel = data.drop(['label'],axis = 1)

print(pixel.info())
one_digit = pixel.iloc[200]
one_digitp = one_digit.values.reshape(28,28)
plt.imshow(one_digitp, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()
print(data['label'][200])