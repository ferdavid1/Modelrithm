# Modelrithm
Python Framework that compares 7 different Machine Learning algorithms' accuracy, precision, and F-Beta scores efficiently (~0.15 second runtime) for a given user's training and testing data, and returns the highest scoring of each, along with automated hyperparameter optimization.

# Installation
pip install Modelrithm

# Requirements
- sklearn
- matplotlib
- numpy

# Usage
	from Modelrithm import Modelrithm

	model = Modelrithm(X_test, Y_train, X_test, Y_train)
	model.Classification()

# Examples
- Aided in increasing accuracy, precision, and f-beta score of a classification problem using satellite pictures of the earth and moon.

	- This application is included under the 'Examples' folder

# Accolades
Winner, Lockheed Martin Aerospace Challenge - SBHacks 2017

# TODO
- Hyperparameter optimization by Random Search (not Grid Search, takes too long).
- Add XGBoost