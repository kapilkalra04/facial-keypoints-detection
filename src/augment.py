import numpy as np
import matplotlib.pyplot as plt

def invert(X):
	for i in range(0,len(X)) :
		X[i,:,:,0] = np.fliplr(X[i,:,:,0])
	return X

def swap(Y):
	col0 = np.copy(Y[:,0])
	col2 = np.copy(Y[:,2])
	Y[:,0] = col2
	Y[:,2] = col0
	col1 = np.copy(Y[:,1])
	col3 = np.copy(Y[:,3])
	Y[:,1] = col3
	Y[:,3] = col1
	return Y