import numpy as np 
import sys
import pickle

def loading_verbose(string):
	msg = (string)
	sys.stdout.write('\r'+msg)
	sys.stdout.flush()

def spherical_to_cartesian(theta):
	x = np.zeros_like(theta)
	x[0] = theta[0]*np.cos(theta[1])
	for i in range(1,theta.shape[0]-1):
		x[i] = x[i-1]*np.tan(theta[i])*np.cos(theta[i+1])
	x[-1] = x[-2]*np.tan(theta[-1])
	return x

def save_model(filename, data):
	f = open(filename, 'wb')
	pickle.dump(data, f)
	f.close()
	print('Model saved at', filename)

def load_model(filename):
	f = open(filename, 'rb')
	data = pickle.load(f)
	f.close()
	print('Model read')
	return data

