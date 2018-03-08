from keras.models import load_model
import keras_new
import keras_svm

import matplotlib as plt
import numpy as np


model = load_model('logreg.h5')
#yFit = model.predict(xVal, batch_size=10, verbose=1)
#print()
#print(yFit)
