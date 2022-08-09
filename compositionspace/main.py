from ast import Raise
from cgitb import reset
from Classifier.Classifier import Classifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
import json 
import pandas as pd

try:
    experiment_param = json.load(open("experiment_params.json","r"))
except:
    ValueError("Could not read parameter file")

experiment = Classifier(experiment_param)
experiment.get_training_data()
experiment.get_test_data()



