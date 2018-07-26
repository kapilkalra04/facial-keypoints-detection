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

    col4 = np.copy(Y[:,4])
    col8 = np.copy(Y[:,8])
    Y[:,4] = col8
    Y[:,8] = col4

    col5 = np.copy(Y[:,5])
    col9 = np.copy(Y[:,9])
    Y[:,5] = col9
    Y[:,9] = col5

    col6 = np.copy(Y[:,6])
    col10 = np.copy(Y[:,10])
    Y[:,6] = col10
    Y[:,10] = col6

    col7 = np.copy(Y[:,7])
    col11 = np.copy(Y[:,11])
    Y[:,7] = col11
    Y[:,11] = col7
    
    col12 = np.copy(Y[:,12])
    col16 = np.copy(Y[:,16])
    Y[:,12] = col16
    Y[:,16] = col12
    
    col13 = np.copy(Y[:,13])
    col17 = np.copy(Y[:,17])
    Y[:,13] = col17
    Y[:,17] = col13
    
    col14 = np.copy(Y[:,14])
    col18 = np.copy(Y[:,18])
    Y[:,14] = col18
    Y[:,18] = col14
    
    col15 = np.copy(Y[:,15])
    col19 = np.copy(Y[:,19])
    Y[:,15] = col19
    Y[:,19] = col15
    
    col22 = np.copy(Y[:,22])
    col24 = np.copy(Y[:,24])
    Y[:,22] = col24
    Y[:,24] = col22

    col23 = np.copy(Y[:,23])
    col25 = np.copy(Y[:,25])
    Y[:,23] = col25
    Y[:,25] = col23
    
    
    return Y
