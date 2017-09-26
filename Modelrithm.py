from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, Perceptron, SGDRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
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
		for x in classifiernames:
			model = x + '()'
			callclassifiers.append(model)

		print("-------------------------------------------------------------------\n")
		print("Testing your data on the following models: {}".format(callclassifiers))
		print("\n-------------------------------------------------------------------\n")
		print("This may take a while, make yourself comfortable...\n")

		alg1 = SVC()
		alg1.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, alg1.predict(Xtest)))
		precision.append(precision_score(Ytest, alg1.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, alg1.predict(Xtest), beta=1))

		alg2 = KNeighborsClassifier()
		alg2.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, alg2.predict(Xtest)))
		precision.append(precision_score(Ytest, alg2.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, alg2.predict(Xtest), beta=1))

		alg3 = DecisionTreeClassifier()
		alg3.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, alg3.predict(Xtest)))
		precision.append(precision_score(Ytest, alg3.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, alg3.predict(Xtest), beta=1))

		alg4 = RandomForestClassifier()
		alg4.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, alg4.predict(Xtest)))
		precision.append(precision_score(Ytest, alg4.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, alg4.predict(Xtest), beta=1))

		alg5 = AdaBoostClassifier()
		alg5.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, alg5.predict(Xtest)))
		precision.append(precision_score(Ytest, alg5.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, alg5.predict(Xtest), beta=1))

		alg6 = GaussianNB()
		alg6.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, alg6.predict(Xtest)))
		precision.append(precision_score(Ytest, alg6.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, alg6.predict(Xtest), beta=1))

		alg7 = LogisticRegression()
		alg7.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, alg7.predict(Xtest)))
		precision.append(precision_score(Ytest, alg7.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, alg7.predict(Xtest), beta=1))


		plt.plot(accuracy)
		plt.show()

		#accuracy.sort()
		#mostaccurate = accuracy[0]
		accuracy_dict = {"SVC()":accuracy[0], "KNeighborsClassifier": accuracy[1], 'DecisionTreeClassifier':accuracy[2], 'RandomForestClassifier':accuracy[3], 'AdaBoostClassifier':accuracy[4], 'GaussianNB':accuracy[5], 'LogisticRegression':accuracy[6]}
		precision_dict = {"SVC()":precision[0], "KNeighborsClassifier": precision[1], 'DecisionTreeClassifier':precision[2], 'RandomForestClassifier':precision[3], 'AdaBoostClassifier':precision[4], 'GaussianNB':precision[5], 'LogisticRegression': precision[6]}
		fbeta_dict = {"SVC()": fbeta[0], "KNeighborsClassifier": fbeta[1], 'DecisionTreeClassifier':fbeta[2], 'RandomForestClassifier': fbeta[3], 'AdaBoostClassifier':fbeta[4], 'GaussianNB':fbeta[5], 'LogisticRegression':fbeta[6]}
		print("*****")
		print("\nThe accuracy of each model is: \n{}\n".format(accuracy_dict))
		print("The precision of each model is: \n{}\n".format(precision_dict))
		print("The F-beta score of each model is: \n{}\n".format(fbeta_dict))
		print("*****")
		return accuracy

