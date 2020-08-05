# Carrega o Modelo

import os
import numpy as np
import keras.models
from keras.models import model_from_json
from tensorflow.keras import layers
import tensorflow.python.keras.backend as K
import imageio
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# Init Function
def init(): 
	json_file = open('model/model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model/model.h5")
	print("Modelo Carregado")

	# Compila e Avalia o Modelo
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	graph = K.get_graph()

	return loaded_model, graph
