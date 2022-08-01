import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('dataset.csv')

X = dataset.iloc[:, :33]

y = dataset.iloc[:, -1]

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion= 'gini', max_depth = 9, max_features = 'log2', max_leaf_nodes=9, n_estimators=50).fit(X,y)


pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
