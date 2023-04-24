# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:38:32 2023

@author: DELL
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
iris = load_iris()

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=10)

# Fit the model on the iris dataset
rf.fit(iris.data, iris.target)

# Save the model as a SavedModel
joblib.dump(rf, "saved_model.joblib")