import numpy as np
from sklearn.metrics import confusion_matrix


def compute_mmae(pred, actual):
	cmt = confusion_matrix(actual, pred)
	n = np.shape(cmt)[0]

	cost = np.abs(np.tile(np.linspace(1,n,n), (n,1)) - np.tile(np.transpose(np.linspace(1,n,n).reshape(1,n)), (1,n)))

	cost = np.transpose(cost)

	mae = []
	for i in range(n):
		mae.append(np.sum(np.multiply(cost[i],cmt[i])) / np.sum(cmt[i]))
		#mae[i] = sum(cost(1+(i*n):(i*n)+n).*cmt(1+(i*n):(i*n)+n)) / sum(cmt(1+(i*n):(i*n)+n));            

	maxmae = max(mae)
	return maxmae
