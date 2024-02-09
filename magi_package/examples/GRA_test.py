import pandas as pd
from magi_package.magi.GRA import GRA
from sklearn import datasets

dataset = datasets.load_iris()
data = dataset['data']
iris_type = dataset['target']
data_df = pd.DataFrame(data=data)
data_df['4'] = iris_type

gra = GRA(data=data_df).run()
