import h2o
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from h2o.estimators.random_forest import H2ORandomForestEstimator

# H2O Init
h2o.init(ip="localhost", port=54321, max_mem_size_GB=2)

cancer = h2o.import_file("C:/Users/Utilizador/Documents/Faculdade/2Semestre/IA/Cancer_Data.csv")

# Split train data into training and testing (80/20)
train_frame, test_frame = cancer.split_frame(ratios=[0.8])

# Specify the target column
train_frame['diagnosis'] = train_frame['diagnosis'].asfactor()
model = H2ORandomForestEstimator(ntrees=3, max_depth=3)

# Train population with train data
model.train(x=train_frame.columns, y='diagnosis', training_frame=train_frame)

# Make predictions on the test data
predictions = model.predict(test_frame).as_data_frame()['predict'].tolist()

# retrieve the model performance
perf = model.model_performance(test_frame)
print(perf)

print(predictions)
# print("Tamanho do conjunto de treino:", train_frame)
# print("Tamanho do conjunto de teste:", test_frame)
