# Kaggle_MINST Vincent Gariépy

Pour éxectur le code dans ce projet, il faut télécharger les librairires qui se retrouvent dans le fichier requirements.txt. Ensuite, il faut télécharger les données d'entrainement, les labels et les données test et les déposer dans un folder nommé "Data".

Il y a 3 Jupyter notebooks et 1 fichier script de python. 

Le code avec la classe de régression logistique est dans le fichier LogRegression.py. Le prétraitement des données et l'entrainement de ce modèle sont détaillés dans le notebook ProcdeureLogReg.ipynb. Dans ce notebook il y a également des graphiques, des résultats et une analyse supplémentaire sur certain hyperparamètres.

Le prétraitement des données et l'entrainement du modèle CatBoost sont détaillés dans le notebook CatBoost_Model.ipynb.

Le prétraitement des données, la construction des modèles de réseau de neurones convolutifs et l'entrainement de ceux-ci sont dans tous dans notebook CNN_Model.ipynb. Il y a également l'analyse d'entrainement et de résultat de 2 architectures différentes ainsi que quelques graphiques supplémentaires sur la perte pendant l'entrainement.