# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

# load the model
loaded_model = pickle.load(open(r"C:\Users\HP\OneDrive\Desktop\Deploy\trained_model.sav", "rb"))

input_data = (5,166,72,19,175,25.8,0.587,51)

# convert to numpy array
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# directly predict (if model is already trained with scaling inside)
prediction = loaded_model.predict(input_data_as_numpy_array)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
