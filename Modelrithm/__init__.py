# from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, Perceptron, SGDRegressor
from sklearn.svm import SVC#, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, fbeta_score
import matplotlib.pyplot as plt
import numpy as np
# from classifier_threading import CThread
import threading

exitFlag = 0
class CThread(threading.Thread):

	def __init__(self, threadID, name, Xtrain, Xtest, Ytrain, Ytest):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = Xtrain, Xtest, Ytrain, Ytest

	def run(self):
		print("Starting {} Classifier in a new thread...".format(self.name))
		# Get lock to synchronize threads
		algo = self.name
		threadLock.acquire()
		c = Modelrithm()
		cl = c.Classification(self.Xtrain, self.Xtest, self.Ytrain, self.Ytest)
		cl.algo
		threadLock.release()
		print("Exiting thread for Classifier: {}".format(self.name))

threadLock = threading.Lock()
threads = []

class Modelrithm:

	def __init__(self):
		self.algtypes = ['Regression', "Classification"]


	def Classification(Xtrain, Xtest, Ytrain, Ytest):
		classifiernames = ['SupportVectorClassifier', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GaussianNaiveBayes', 'LogisticRegression']
		accuracy = []
		precision = []
		fbeta= []

		def Support():
			svc = SVC()
			svc.fit(Xtrain, Ytrain)
			accuracy.append(accuracy_score(Ytest, svc.predict(Xtest)))
			precision.append(precision_score(Ytest, svc.predict(Xtest)))
			fbeta.append(fbeta_score(Ytest, svc.predict(Xtest), beta=1))

		def KNearest():
			knn = KNeighborsClassifier()
			knn.fit(Xtrain, Ytrain)
			accuracy.append(accuracy_score(Ytest, knn.predict(Xtest)))
			precision.append(precision_score(Ytest, knn.predict(Xtest)))
			fbeta.append(fbeta_score(Ytest, knn.predict(Xtest), beta=1))

		def DecisionTree():
			dt = DecisionTreeClassifier()
			dt.fit(Xtrain, Ytrain)
			accuracy.append(accuracy_score(Ytest, dt.predict(Xtest)))
			precision.append(precision_score(Ytest, dt.predict(Xtest)))
			fbeta.append(fbeta_score(Ytest, dt.predict(Xtest), beta=1))

		def RandomForest():
			rf = RandomForestClassifier()
			rf.fit(Xtrain, Ytrain)
			accuracy.append(accuracy_score(Ytest, rf.predict(Xtest)))
			precision.append(precision_score(Ytest, rf.predict(Xtest)))
			fbeta.append(fbeta_score(Ytest, rf.predict(Xtest), beta=1))

		def Ada():
			adaboost = AdaBoostClassifier()
			adaboost.fit(Xtrain, Ytrain)
			accuracy.append(accuracy_score(Ytest, adaboost.predict(Xtest)))
			precision.append(precision_score(Ytest, adaboost.predict(Xtest)))
			fbeta.append(fbeta_score(Ytest, adaboost.predict(Xtest), beta=1))

		def GNB():
			gauss_nb = GaussianNB()
			gauss_nb.fit(Xtrain, Ytrain)
			accuracy.append(accuracy_score(Ytest, gauss_nb.predict(Xtest)))
			precision.append(precision_score(Ytest, gauss_nb.predict(Xtest)))
			fbeta.append(fbeta_score(Ytest, gauss_nb.predict(Xtest), beta=1))

		def LogReg():
			lr = LogisticRegression()
			lr.fit(Xtrain, Ytrain)
			accuracy.append(accuracy_score(Ytest, lr.predict(Xtest)))
			precision.append(precision_score(Ytest, lr.predict(Xtest)))
			fbeta.append(fbeta_score(Ytest, lr.predict(Xtest), beta=1))

		CL = {}
		for ind, x in enumerate(classifiernames):
			CL[ind] = x
		prompt = input("Please choose which classifiers you wish to compare. \nThe options are: \n{}".format(classifiernames) + "\nPlease input the indices of the algorithms you want, separated by commas, and with no spaces.")
		chosen = ",".join(prompt)
		available = [Support(), KNearest(), DecisionTree(), RandomForest(), Ada(), GaussianNaiveBayes(), LogReg()]
		final_choice = [available[c] for c in chosen]
		for ind, f in enumerate(final_choice):
			CThread(ind, final_choice[f], Xtrain, Xtest, Ytrain, Ytest).start()
		for ind, f in enumerate(final_choice):
			threads.append(CThread(ind, final_choice[f], Xtrain, Xtest, Ytrain, Ytest))
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
