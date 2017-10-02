# from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, Perceptron, SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC#, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, fbeta_score
from sklearn.grid_search import RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np
# from classifier_threading import CThread
import threading
import time
from xgboost import XGBClassifier

class CThread(threading.Thread):

	def __init__(self, threadID, name, function):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.function = function

	def run(self):
		print("Starting {} Classifier in a new thread...".format(str(self.name)))
		# Get lock to synchronize threads
		threadLock.acquire()
		self.function
		# Free lock to release next thread
		threadLock.release()
		print("Exiting thread for Classifier: {}".format(str(self.name)))

threadLock = threading.Lock()
threads = []

class Modelrithm(object):

	def __init__(self, Xtrain, Xtest, Ytrain, Ytest):
		self.algtypes = ['Regression', "Classification"]
		self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = Xtrain, Xtest, Ytrain, Ytest
		self.accuracy = []
		self.precision = []
		self.fbeta= []
		self.classifiernames = ['SupportVectorClassifier', 'KNeighborsClassifier', \
		'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GaussianNaiveBayes',\
		'LogisticRegression', 'XGBClassifier']

	def Support(self):
		Xtrain, Xtest, Ytrain, Ytest = self.Xtrain, self.Xtest, self.Ytrain, self.Ytest
		accuracy = self.accuracy
		precision = self.precision
		fbeta = self.fbeta
		svc = SVC()
		svc.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, svc.predict(Xtest)))
		precision.append(precision_score(Ytest, svc.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, svc.predict(Xtest), beta=1))

	def KNearest(self):
		Xtrain, Xtest, Ytrain, Ytest = self.Xtrain, self.Xtest, self.Ytrain, self.Ytest
		accuracy = self.accuracy
		precision = self.precision
		fbeta = self.fbeta
		knn = KNeighborsClassifier()
		knn.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, knn.predict(Xtest)))
		precision.append(precision_score(Ytest, knn.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, knn.predict(Xtest), beta=1))

	def DecisionTree(self):
		Xtrain, Xtest, Ytrain, Ytest = self.Xtrain, self.Xtest, self.Ytrain, self.Ytest
		accuracy = self.accuracy
		precision = self.precision
		fbeta = self.fbeta			
		dt = DecisionTreeClassifier()
		dt.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, dt.predict(Xtest)))
		precision.append(precision_score(Ytest, dt.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, dt.predict(Xtest), beta=1))

	def RandomForest(self):
		Xtrain, Xtest, Ytrain, Ytest = self.Xtrain, self.Xtest, self.Ytrain, self.Ytest
		accuracy = self.accuracy
		precision = self.precision
		fbeta = self.fbeta
		rf = RandomForestClassifier()
		rf.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, rf.predict(Xtest)))
		precision.append(precision_score(Ytest, rf.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, rf.predict(Xtest), beta=1))

	def Ada(self):
		Xtrain, Xtest, Ytrain, Ytest = self.Xtrain, self.Xtest, self.Ytrain, self.Ytest
		accuracy = self.accuracy
		precision = self.precision
		fbeta = self.fbeta
		adaboost = AdaBoostClassifier()
		adaboost.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, adaboost.predict(Xtest)))
		precision.append(precision_score(Ytest, adaboost.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, adaboost.predict(Xtest), beta=1))

	def XGB(self):
		Xtrain, Xtest, Ytrain, Ytest = self.Xtrain, self.Xtest, self.Ytrain, self.Ytest
		accuracy = self.accuracy
		precision = self.precision
		fbeta = self.fbeta
		xgb = XGBClassifier()
		xgb.fit(np.array(Xtrain), np.array(Ytrain))
		accuracy.append(accuracy_score(Ytest, xgb.predict(Xtest)))
		precision.append(precision_score(Ytest, xgb.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, xgb.predict(Xtest), beta=1))

	def GNB(self):
		Xtrain, Xtest, Ytrain, Ytest = self.Xtrain, self.Xtest, self.Ytrain, self.Ytest
		accuracy = self.accuracy
		precision = self.precision
		fbeta = self.fbeta
		gauss_nb = GaussianNB()
		gauss_nb.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, gauss_nb.predict(Xtest)))
		precision.append(precision_score(Ytest, gauss_nb.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, gauss_nb.predict(Xtest), beta=1))

	def LogReg(self):
		Xtrain, Xtest, Ytrain, Ytest = self.Xtrain, self.Xtest, self.Ytrain, self.Ytest
		accuracy = self.accuracy
		precision = self.precision
		fbeta = self.fbeta
		lr = LogisticRegression()
		lr.fit(Xtrain, Ytrain)
		accuracy.append(accuracy_score(Ytest, lr.predict(Xtest)))
		precision.append(precision_score(Ytest, lr.predict(Xtest)))
		fbeta.append(fbeta_score(Ytest, lr.predict(Xtest), beta=1))

	def Classification(self):
		Xtrain, Xtest, Ytrain, Ytest = self.Xtrain, self.Xtest, self.Ytrain, self.Ytest
		classifiernames = self.classifiernames
		n = time.time()
		svmthread = CThread(1, classifiernames[0], self.Support())
		knthread = CThread(2, classifiernames[1], self.KNearest())
		dtthread = CThread(3, classifiernames[2], self.DecisionTree())
		rfthread = CThread(4, classifiernames[3], self.RandomForest())
		adathread = CThread(5, classifiernames[4], self.Ada())
		gnbthread = CThread(6, classifiernames[5], self.GNB())
		lrthread = CThread(7, classifiernames[6], self.LogReg())
		xgbthread = CThread(8, classifiernames[7], self.XGB())

		threads.append(svmthread)
		threads.append(knthread)
		threads.append(dtthread)
		threads.append(rfthread)
		threads.append(adathread)
		threads.append(gnbthread)
		threads.append(lrthread)
		threads.append(xgbthread)

		for t in threads:
			t.start()
		for t in threads:
			t.join()
		p = time.time()
		print('Time taken to run classifiers: ', p-n)
		print("Exiting Main Thread...")

		self.results()

	def results(self):
		accuracy = self.accuracy
		precision = self.precision
		fbeta = self.fbeta

		#accuracy.sort()
		# mostaccurate = accuracy[0]
		accuracy_dict = {"SVC":accuracy[0], "KNeighborsClassifier": accuracy[1], 'DecisionTreeClassifier':accuracy[2], 'RandomForestClassifier':accuracy[3], 'AdaBoostClassifier':accuracy[4], 'GaussianNB':accuracy[5], 'LogisticRegression':accuracy[6], 'XGBClassifier':accuracy[7]}
		plt.figure()
		plt.title("Accuracy per Classifier")
		plt.xticks(np.arange(len(accuracy_dict)), list(accuracy_dict.keys()))
		plt.plot(accuracy)
		plt.tight_layout()
		plt.show()

		precision_dict = {"SVC":precision[0], "KNeighborsClassifier": precision[1], 'DecisionTreeClassifier':precision[2], 'RandomForestClassifier':precision[3], 'AdaBoostClassifier':precision[4], 'GaussianNB':precision[5], 'LogisticRegression': precision[6], 'XGBClassifier':precision[7]}
		plt.figure()
		plt.title("Precision per Classifier")
		plt.xticks(np.arange(len(precision_dict)), list(precision_dict.keys()))
		plt.plot(precision)
		plt.tight_layout()
		plt.show()

		fbeta_dict = {"SVC": fbeta[0], "KNeighborsClassifier": fbeta[1], 'DecisionTreeClassifier':fbeta[2], 'RandomForestClassifier': fbeta[3], 'AdaBoostClassifier':fbeta[4], 'GaussianNB':fbeta[5], 'LogisticRegression':fbeta[6], 'XGBClassifier':fbeta[7]}
		plt.figure()
		plt.title("F-Beta per Classifier")
		plt.xticks(np.arange(len(fbeta_dict)), list(fbeta_dict.keys()))
		plt.plot(fbeta)
		plt.tight_layout()
		plt.show()

		print("*****")
		print("\nThe accuracy of each model is: \n{}\n".format(accuracy_dict))
		print("The precision of each model is: \n{}\n".format(precision_dict))
		print("The F-beta score of each model is: \n{}\n".format(fbeta_dict))
		print("*****")