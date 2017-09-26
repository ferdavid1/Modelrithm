from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, Perceptron, SGDRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, fbeta_score
import matplotlib.pyplot as plt
import numpy as np

class Modelrithm:

	def __init__(self):
		self.algtypes = ['Regression', "Classification"]


	def Classification(Xtrain, Xtest, Ytrain, Ytest):
		classifiernames = ['SVC', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GaussianNB', 'LogisticRegression']
		callclassifiers = []
		accuracy = []
		precision = []
		fbeta= []

		print("-------------------------------------------------------------------\n")
		print("Testing your data on the following models: {}".format(callclassifiers))
		print("\n-------------------------------------------------------------------\n")
		print("This may take a while, make yourself comfortable...\n")

		svc = SVC()
		svc.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, svc.predict(Xtest)))
		precision.append(precision_score(Ytest, svc.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, svc.predict(Xtest), beta=1))

		knn = KNeighborsClassifier()
		knn.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, knn.predict(Xtest)))
		precision.append(precision_score(Ytest, knn.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, knn.predict(Xtest), beta=1))

		dt = DecisionTreeClassifier()
		dt.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, dt.predict(Xtest)))
		precision.append(precision_score(Ytest, dt.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, dt.predict(Xtest), beta=1))

		rf = RandomForestClassifier()
		rf.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, rf.predict(Xtest)))
		precision.append(precision_score(Ytest, rf.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, rf.predict(Xtest), beta=1))

		adaboost = AdaBoostClassifier()
		adaboost.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, adaboost.predict(Xtest)))
		precision.append(precision_score(Ytest, adaboost.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, adaboost.predict(Xtest), beta=1))

		gauss_nb = GaussianNB()
		gauss_nb.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, gauss_nb.predict(Xtest)))
		precision.append(precision_score(Ytest, gauss_nb.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, gauss_nb.predict(Xtest), beta=1))

		lr = LogisticRegression()
		lr.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, lr.predict(Xtest)))
		precision.append(precision_score(Ytest, lr.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, lr.predict(Xtest), beta=1))


		plt.plot(accuracy)
		plt.show()

		#accuracy.sort()
		#mostaccurate = accuracy[0]
		accuracy_dict = {"SVC":accuracy[0], "KNeighborsClassifier": accuracy[1], 'DecisionTreeClassifier':accuracy[2], 'RandomForestClassifier':accuracy[3], 'AdaBoostClassifier':accuracy[4], 'GaussianNB':accuracy[5], 'LogisticRegression':accuracy[6]}
		precision_dict = {"SVC":precision[0], "KNeighborsClassifier": precision[1], 'DecisionTreeClassifier':precision[2], 'RandomForestClassifier':precision[3], 'AdaBoostClassifier':precision[4], 'GaussianNB':precision[5], 'LogisticRegression': precision[6]}
		fbeta_dict = {"SVC": fbeta[0], "KNeighborsClassifier": fbeta[1], 'DecisionTreeClassifier':fbeta[2], 'RandomForestClassifier': fbeta[3], 'AdaBoostClassifier':fbeta[4], 'GaussianNB':fbeta[5], 'LogisticRegression':fbeta[6]}
		print("*****")
		print("\nThe accuracy of each model is: \n{}\n".format(accuracy_dict))
		print("The precision of each model is: \n{}\n".format(precision_dict))
		print("The F-beta score of each model is: \n{}\n".format(fbeta_dict))
		print("*****")
		return accuracy

