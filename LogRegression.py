import pandas as pd
import numpy as np

class LogisiticRegression:
    def __init__(self, trainData, trainLabels):
        #Sauver les données et calculer les variables importantes
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.featureSize = len(trainData.columns)
        self.trainSize = len(self.trainData.index)
        self.labels = np.sort(trainLabels.unique())
        self.labelSize = self.labels.size

        self.weights = pd.DataFrame(np.zeros((self.featureSize,self.labelSize)))
        self.bias = pd.Series(np.zeros(self.labelSize))

        loss = None

    def softmax(self,x):
        #Fonction de softmax, peut prendre une matrice en entrée, retourne un numpy array de la meme forme que l'entrée
        x = x.transpose()
        e_x = np.exp(x - np.max(x,axis=0))
        return (e_x / e_x.sum(axis=0)).transpose()

    def oneHotEncoder(self,Y):
        #Transformer les labels en one hot pour calculer le loss
        df = pd.DataFrame(np.zeros((self.labelSize,Y.size)))
        for i,val in enumerate(Y):
            ind = np.nonzero(self.labels == val)[0][0]
            df.loc[ind,i] = 1
        return df

    def plotLoss(self):
        #Pour faire un graphique du loss pour analyser
        return self.loss.plot(xlabel='itération',ylabel='Coût (entropie croisé)',kind='line',
                                title="Perte d'entrainement en fonction du nombre d'itération")
    
    def predictSoftmax(self, df):
        #Fonction qui donne les probabilités de prediction de la matrice d'entrée
        #Calculer le le produit de W^T*X^T+b pour tous les echantillons
        newW = self.weights.transpose()
        newX = df.transpose()
        wx = newW.dot(newX.to_numpy())
        gFunction = -(wx.transpose()+self.bias)

        #Faire le argmax(softmax(g(x))) pour tous les echantillons
        softMaxResult = self.softmax(gFunction)

        return softMaxResult

    def predict(self, df):
        #Fonction qui donne les predictions de la matrice d'entrée
        #Calculer le le produit de W^T*X^T+b pour tous les echantillons
        newW = self.weights.transpose()
        newX = df.transpose()
        wx = newW.dot(newX.to_numpy())
        gFunction = -(wx.transpose()+self.bias)

        #Faire le argmax(softmax(g(x))) pour tous les echantillons
        softMaxResult = gFunction.apply(self.softmax,axis=0)
        predictions = softMaxResult.idxmax(axis=1)
        predictionLabels = [self.labels[i] for i in predictions]

        return predictionLabels
    
    def loss(self,Y,Yhat,batchSize):
        #retourner la cout (entropie croise) 
        return (-np.sum(np.log(np.sum(Y.mul(Yhat.transpose()),axis=0))))/batchSize

    def gradient(self,X,Y,Yhat,bathSize):
        #retourner le gradient avec une regularisation L2 
        #Ajouter le biais au X et W pour avoir son gradient
        X = pd.concat([X.transpose(),pd.Series(1,index=X.index).to_frame(1).transpose()],axis=0)
        W = pd.concat([self.weights,pd.Series(1,index=range(self.labelSize)).to_frame(1).transpose()],axis=0)
        return (X.dot((Y-Yhat.transpose()).transpose().to_numpy()))/bathSize
    
    def accuracy(self,X,Y):
        #Calcul la précision des prédictions
        predictions = self.predict(X)
        return (np.sum(predictions==Y))/len(Y)


    def train(self, iter, batchSize, alpha):
        #Pour enregistrer l'évolution de la perte
        loss = pd.Series(np.zeros(iter))

        #itérer par le nombre de itéreation entrée
        for i in range(iter):
            #Sample aléatoirement un ensemble de données de la grandeur de batch size 
            sampleTrain = self.trainData.sample(batchSize,replace=False,axis=0)
            sampleLabels = self.trainLabels.iloc[sampleTrain.index]
            yOneHot = self.oneHotEncoder(sampleLabels)
            
            #Predire les probabilite des labels et calculer le cout
            predictionProb = self.predictSoftmax(sampleTrain)
            loss[i] = self.loss(yOneHot,predictionProb,batchSize)

            #Calculer le gradient
            grad = self.gradient(sampleTrain,yOneHot,predictionProb,batchSize)

            #Ameliorer W et B
            self.weights = self.weights - alpha*grad.iloc[:-1,:].to_numpy()
            self.bias = self.bias - alpha*grad.iloc[-1,:].to_numpy()

        self.loss = loss    
        