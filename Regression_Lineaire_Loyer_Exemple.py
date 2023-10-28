import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score

# Chargement des données
raw_data = pd.read_csv('house_data.csv')

# Résumé des données brutes
raw_data.describe()

# Il y a quelques valeurs manquantes, on supprime ces lignes
data_na = raw_data.dropna()

# Comme vu dans le TP, on a des outliers sur les grands propriétés
data = data_na[data_na["price"] < 8000]

# On reindexe
data = data.reset_index(drop = True)

# On affiche les données nettoyées
data.plot.scatter("price", "surface", c="arrondissement", colormap='viridis')
plt.show()

#On affiche maintenant la variable prédite (loyer) en fonction de l'arrondissement
ax1 = sns.violinplot(data=data, x="arrondissement", y="price",  hue='arrondissement')
ax1.minorticks_on()
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.grid(which='minor', axis='x', linewidth=1)
plt.show()

# croiser avec la surface pour avoir une vision plus claire
# Pour faciliter la visualisation, on va changer la valeur de l'arrondissement (10)
fig = plt.figure().add_subplot(projection='3d')
tmp_arr = data['arrondissement'][:]
tmp_arr[tmp_arr == 10] = 5
fig.scatter(tmp_arr, data['surface'], data['price'], c=tmp_arr, cmap="viridis")
plt.show()

xtrain, xtest, ytrain, ytest = train_test_split(data[["surface", "arrondissement"]], data[["price"]], test_size=0.3)

#1er modele Regression lineaire variable surface
lr1 = LinearRegression()
baseline_pred = lr1.fit(xtrain[["surface"]], ytrain).predict(xtest[["surface"]])
rmse = (np.sqrt(mean_squared_error(ytest, baseline_pred)))
r2 = r2_score(ytest, baseline_pred)
print("l'erreur RMSE est {}".format(rmse))
print('le score R2 score est {}'.format(r2))

#2eme modele Regression lineaire variable surface et arrondissement
lr2 = LinearRegression()
ytest_predict = lr2.fit(xtrain, ytrain).predict(xtest)
fig1 = plt.figure().add_subplot(projection='3d')
fig2 = plt.figure().add_subplot(projection='3d')
fig1.scatter(xtest[["arrondissement"]], xtest[["surface"]], ytest, c=xtest[["arrondissement"]], cmap="viridis")
fig2.scatter(xtest[["arrondissement"]], xtest[["surface"]], ytest_predict, c=xtest[["arrondissement"]], cmap="viridis")
plt.show()
rmse = (np.sqrt(mean_squared_error(ytest, ytest_predict)))
r2 = r2_score(ytest, ytest_predict)
print("l'erreur RMSE est {}".format(rmse))
print('le score R2 score est {}'.format(r2))



#3eme modele Regression lineaire separation des données par arrondissement
lrs = []
for i in np.unique(xtrain["arrondissement"]):
    
    # On génère un jeu de données par arrondissement
    tr_arr = xtrain['arrondissement']==i
    te_arr = xtest['arrondissement']==i
    
    xtrain_arr = xtrain[tr_arr]
    ytrain_arr = ytrain[tr_arr]

    xtest_arr = xtest[te_arr]
    ytest_arr = ytest[te_arr]

    lr = LinearRegression()
    lr.fit(xtrain_arr[["surface"]], ytrain_arr)
    lrs.append(lr)
    
#On effectue la prédiction finale sur le jeu de donnée test avec notre nouveau modèle, 
#qui combine les différents modèles par arrondissement
final_pred = []

for idx,val in xtest.iterrows():
    final_pred.append(lrs[int(val["arrondissement"]-1)].predict([[val["surface"]]])[0][0])
 
    
rmse = (np.sqrt(mean_squared_error(ytest, final_pred)))
r2 = r2_score(ytest, final_pred)
print("l'erreur RMSE est {}".format(rmse))
print('le score R2 score est {}'.format(r2))

#On peut afficher cette prédiction finale
plt.plot(xtest[["surface"]], ytest, 'bo', markersize = 5)
plt.plot(xtest[["surface"]], lrs[0].predict(xtest[["surface"]]), color="#00FFFF", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[1].predict(xtest[["surface"]]), color="#0000FF", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[2].predict(xtest[["surface"]]), color="#00FF00", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[3].predict(xtest[["surface"]]), color="#FF0000", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[4].predict(xtest[["surface"]]), color="#FFFF00", linewidth = 2)
plt.show()