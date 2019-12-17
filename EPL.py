import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import math
import datetime
import collections
import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv('train_epl.csv')
df.drop(df.iloc[:,25:],inplace=True,axis=1)
df.dropna(inplace=True)
#print(df.head())
team_names=pd.read_csv('Team_Names2019.csv')
team_list=team_names['Team_Name'].tolist()
#print(team_names)

# Feature Creation
def Team_Stats(team,year):
	stats=df[df['Year']==year]
	# total Number of games played and total goals scored by the selected team
	home_games=stats[stats['HomeTeam']==team]
	away_games=stats[stats['AwayTeam']==team]
	total=home_games.append(away_games)
	total_games=len(total)
	total_goals_scored=home_games['FTHG'].sum()
	total_goals_scored+=away_games['FTAG'].sum()

	#Goals Conceded
	total_goals_conceded=home_games['FTAG'].sum()
	total_goals_conceded+=away_games['FTHG'].sum()

	#yellow Cards
	total_yellow_cards=home_games['HY'].sum()
	total_yellow_cards+=away_games['AY'].sum()

	#Red Cards
	total_red_cards=home_games['HR'].sum()
	total_red_cards+=away_games['AR'].sum()
	#Corners
	total_corners=home_games['HC'].sum()
	total_corners+=away_games['AC'].sum()
	#fouls
	total_fouls=home_games['HF'].sum()
	total_fouls+=away_games['AF'].sum()

	#Shots taken per Game
	total_shots_taken=home_games['HS'].sum()
	total_shots_taken+=away_games['AS'].sum()
	if total_games !=0:
		shots_taken_per_game=total_shots_taken/total_games


	#Shots conceded per game
	total_shots_conceded=home_games['AS'].sum()
	total_shots_conceded+=away_games['HS'].sum()
	if total_games!=0:
		shots_conceded_per_game=total_shots_conceded/total_games

	#Goalkeeper Stats
	shots_on_target_opp=home_games['AST'].sum()
	shots_on_target_opp+=away_games['HST'].sum()
	Team_goalie_saves=shots_on_target_opp- total_goals_conceded

	shots_target_team=home_games['HST'].sum()
	shots_target_team=away_games['AST'].sum()
	opp_goalie_saves=shots_target_team- total_goals_scored


	#Saves percentage and Ratio for team in consideration

	if shots_on_target_opp!=0:
		Team_save_percentage= Team_goalie_saves/shots_on_target_opp
	if Team_goalie_saves!=0:
		Team_save_ratio=shots_on_target_opp/Team_goalie_saves
	# Save percentage and Ratio for the team opposing to the team in consderation
	if shots_target_team!=0:
		opp_save_percentage=opp_goalie_saves/shots_target_team
	if opp_goalie_saves!=0:
		opp_save_ratio=shots_target_team/opp_goalie_saves

	#Goal Scoring Rate for Team in consideration
	if total_shots_taken!=0:
		Scoring_Rate_percentage=total_goals_scored/total_shots_taken
	if total_goals_scored!=0:
		Scoring_Rate_ratio=total_shots_taken/total_goals_scored
	#Goal Against Rate
	if total_shots_conceded!=0:
		Conceding_Rate_percentage=total_goals_conceded/total_shots_conceded
	if total_goals_conceded!=0:
		Conceding_Rate_ratio=total_shots_conceded/total_goals_conceded
	games_won=stats[stats['Winner']==team]
	games_lost=stats[stats['Loser']==team]
	total_games_won=len(games_won)
	total_games_lost=len(games_lost)
	if total_games!=0:
		win_percentage=total_games_won/total_games
		loose_percentage=1-win_percentage

	if total_games==0:
		total_games_won=0
		total_games_lost=0
		total_goals_scored=0
		total_goals_conceded=0
		total_yellow_cards=0
		total_red_cards=0
		total_fouls=0
		total_corners=0
		shots_taken_per_game=0
		shots_conceded_per_game=0
		win_percentage=0
		loose_percentage=0
		Team_goalie_saves=0
		opp_goalie_saves=0
		Team_save_ratio=0
		Team_save_percentage=0
		opp_save_ratio=0
		opp_save_percentage=0
		Scoring_Rate_ratio=0
		Scoring_Rate_percentage=0
		Conceding_Rate_ratio=0
		Conceding_Rate_percentage=0

	return(total_goals_scored,total_goals_conceded,total_yellow_cards,total_red_cards,total_fouls,total_corners,shots_taken_per_game,
		shots_conceded_per_game,win_percentage,Team_goalie_saves,opp_goalie_saves,Team_save_percentage,Team_save_ratio,opp_save_percentage, 
		Scoring_Rate_percentage,Conceding_Rate_percentage,Scoring_Rate_ratio,Conceding_Rate_ratio)
def Team_dict(year):
	dict=collections.defaultdict(list)
	for team in team_list:
		team_=Team_Stats(team,year)
		dict[team]=team_
	return (dict)


df=df.drop(['ID','Date'],axis=1)

feature_table=df.iloc[:,:]

table_1= pd.DataFrame(columns=('Team','HGS','AGS','HAS','AAS','HGC','AGC','HDS','ADS','HSR','ASR','HCR','ACR',))

table_2=feature_table
Home_avg_score=table_2.FTHG.sum()*1.0/table_2.shape[0]
Away_avg_score=table_2.FTAG.sum()*1.0/table_2.shape[0]
Home_avg_conceded=Away_avg_score
Away_avg_conceded=Home_avg_score

home=table_2.groupby("HomeTeam")
away=table_2.groupby("AwayTeam")
table_1.Team = df['HomeTeam'].unique()
table_1=table_1.sort_values(by='Team')
table_1.HGS = home.FTHG.sum().values
table_1.HGC = home.FTAG.sum().values
table_1.AGS = away.FTAG.sum().values
table_1.AGC = away.FTHG.sum().values
table_1.HSR = (home.AST.sum().values-home.FTAG.sum().values)
table_1.ASR = (away.HST.sum().values-away.FTHG.sum().values)
table_1.HCR = home.FTHG.sum().values/home.HS.sum().values
table_1.ACR = away.FTAG.sum().values/away.AS.sum().values



def match_count(team):
	total=df[df['HomeTeam']==team]
	return(total.shape[0])
def match_count1(team):
	total=df[df['AwayTeam']==team]
	return(total.shape[0])
lis=[]
for i in table_1.Team:
	lis.append(match_count(i))
table_1['HG']=lis

lis1=[]
for i in table_1.Team:
	lis1.append(match_count1(i))
table_1['AG']=lis1


table_1['HAS']=table_1['HGS']/table_1['HG']
table_1['AAS']=table_1['AGS']/table_1['AG']
table_1['HDS']=(table_1['HGS']/table_1['HG'])/Home_avg_conceded
table_1['ADS']=(table_1['AGS']/table_1['AG'])/Away_avg_conceded
table_1['HSR']=table_1['HSR']/table_1['HG']
table_1['ASR']=table_1['ASR']/table_1['AG']


table_1=table_1.reset_index()
table_1=table_1.drop(['index','HG','AG'],axis=1)

f, axes = plt.subplots(2, 2, figsize=(5, 5))

h_plot = sns.barplot(table_1.Team,table_1.HAS,ax=axes[0,0])
h_plot.set_xticklabels(h_plot.get_xticklabels(), rotation=90)
h_plot = sns.barplot(table_1.Team,table_1.AAS,ax=axes[0,1])
h_plot.set_xticklabels(h_plot.get_xticklabels(), rotation=90)

h_plot = sns.barplot(table_1.Team,table_1.HDS,ax=axes[1,0])
h_plot.set_xticklabels(h_plot.get_xticklabels(), rotation=90)
h_plot = sns.barplot(table_1.Team,table_1.ADS,ax=axes[1,1])
h_plot.set_xticklabels(h_plot.get_xticklabels(), rotation=90)
#plt.show()


feature_table = feature_table[['HomeTeam','AwayTeam','FTR','HST','AST','HC','AC','HY','AY','HR','AR','HF','AF','HTHG','HTAG']]
f_HAS = []
f_HDS = []
f_AAS = []
f_ADS = []
f_HSR = []
f_ASR = []
f_HCR = []
f_ACR = []


for index,row in feature_table.iterrows():
    f_HAS.append(table_1[table_1['Team'] == row['HomeTeam']]['HAS'].values[0])
    f_HDS.append(table_1[table_1['Team'] == row['HomeTeam']]['HDS'].values[0])
    f_AAS.append(table_1[table_1['Team'] == row['AwayTeam']]['AAS'].values[0])
    f_ADS.append(table_1[table_1['Team'] == row['AwayTeam']]['ADS'].values[0])
    f_HSR.append(table_1[table_1['Team'] == row['AwayTeam']]['HSR'].values[0])
    f_ASR.append(table_1[table_1['Team'] == row['AwayTeam']]['ASR'].values[0])
    f_HCR.append(table_1[table_1['Team'] == row['AwayTeam']]['HCR'].values[0])
    f_ACR.append(table_1[table_1['Team'] == row['AwayTeam']]['ACR'].values[0])
    
feature_table['HAS'] = f_HAS
feature_table['HDS'] = f_HDS
feature_table['AAS'] = f_AAS
feature_table['ADS'] = f_ADS
feature_table['HSR'] = f_HSR
feature_table['ASR'] = f_ASR
feature_table['HCR'] = f_HCR
feature_table['ACR'] = f_ACR

Result={'A':-1,'D':0,'H':1}

data1=[feature_table]
for i in data1:
	i['FTR']=i['FTR'].map(Result)
X = feature_table[['HomeTeam','AwayTeam','HST','AST','HC','AC','HY','AY','HR','AR','HF','AF','HTHG','HTAG']]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
y = feature_table['FTR']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=96)
x_test_=X_test
X_train=X_train.iloc[:,2:]
X_test=X_test.iloc[:,2:]
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

clf1 = RandomForestClassifier

y_pred_1 = clf1.fit(X_train,y_train).predict(X_test)

print(accuracy_score(y_test,y_pred_1))
scores = cross_val_score(clf1, X_test, y_test, cv=10)
print(classification_report(y_test, y_pred_1))
print (scores)
print (scores.mean())


from sklearn.neighbors import KNeighborsClassifier
clf_2 = KNeighborsClassifier(n_neighbors = 50)
y_pred_2 = clf_2.fit(X_train,y_train).predict(X_test)

print(accuracy_score(y_test,y_pred_2))
print(classification_report(y_test, y_pred_2))
scores = cross_val_score(clf_2, X_test, y_test, cv=10)
print (scores)
print (scores.mean())




from sklearn.svm import SVC
from sklearn.svm.libsvm import predict_proba
clf_3=SVC(kernel='linear',C=1,random_state=912,probability=True)
y_pred_3=clf_3.fit(X_train,y_train).predict(X_test)
predictedProbability_3 = clf_3.predict_proba(X_test)*100
print(accuracy_score(y_test,y_pred_3))
print(classification_report(y_test, y_pred_3))
scores=cross_val_score(clf_3, X_test, y_test, cv=10)
print (scores)
print (scores.mean())

from sklearn.model_selection import GridSearchCV
#parameters = [{'C': [0.1,0.5,1, 10, 100, 1000], 'kernel': ['linear'],},
        #      {'C': [0.1,0.5,1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001,0.005,0.01,0.05,0.07,0.1, 0.2, 0.5,10,50]}]
#grid_search = GridSearchCV(estimator = clf_3,
 #                          param_grid = parameters,
  #                         scoring = 'accuracy',
   #                        cv = 10,
    #                       n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)

#print(grid_search.best_score_)
#print(grid_search.best_params_)


x_test_=pd.DataFrame(x_test_)
test_table=pd.DataFrame()
test_table=x_test_[['HomeTeam','AwayTeam']]
test_table['Result_RF']=y_pred_1
test_table['Result_KN']=y_pred_2
test_table['Result_SVM']=y_pred_3
test_table['Actual_Result']=y_test
test_table=test_table.reset_index()
PredictedProbability = pd.DataFrame(predictedProbability_3, columns=['Away win %','Draw %','Home win %'])
test_table=test_table.drop(['index'],axis=1)

test_table['Away WIn %']=PredictedProbability[['Away win %']]
test_table['Draw %']=PredictedProbability[['Draw %']]
test_table['Home WIn %']=PredictedProbability[['Home win %']]

from sklearn.metrics import confusion_matrix
a=confusion_matrix(y_test,y_pred_3)
print(a)
