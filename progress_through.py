
import numpy as np
import pickle
from model_def import vggvox_model

def progress_through(inp):

	# Get clip duration (may vary)
	length = inp.shape[1]


	# Load model + weights
	model = vggvox_model((512,length,1))
	#model.summary()
	model.load_weights('vgg_vox.h5', by_name=True)


	# Prepare input tensor and make predictions
	x = inp.reshape(-1, 512, length, 1)
	predictions = model.predict(x)


	# Present assigned class and score
	arr = list(np.squeeze(predictions))
	score = np.max(arr)
	rec_class = arr.index(score)
	print("Score: ", score)
	print("Class: ", rec_class+1)

	return rec_class+1
