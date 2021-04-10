#Full Time Result (H=Home Win, D=Draw, A=Away Win)
#HTGD - Home team goal difference
#ATGD - away team goal difference
#HTP - Home team points
#ATP - Away team points
#DiffFormPts Diff in points
#DiffLP - Differnece in last years prediction

#Input - 12 other features (fouls, shots, goals, misses,corners, red card, yellow cards)
#Output - Full Time Result (H=Home Win, D=Draw, A=Away Win)



import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xbg
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display #for displayin result

data=pd.read_csv('2019-20.csv')

#data exploration
#win rate for home team?

matches=data.shape[0]
features=data.shape[1]-1 #minus 1 because one will be the feature used (FTR)
homewins=len(data[data.FTR=='H'])
homeWinRate=float(homewins/matches *100)

# print('Total num of matches : {}'.format(matches))
# print('Number of features : {}'.format(features))
# print('Number of home wins: {}'.format(homewins))
# print('The home win rate : {:.2f}%'.format(homeWinRate))

#visualizing data
from pandas.plotting import scatter_matrix
scatter_matrix(data[['HTHG','HTAG','HC','AC','FTR','HTR']], figsize=(10,10))
print(plt.show())

