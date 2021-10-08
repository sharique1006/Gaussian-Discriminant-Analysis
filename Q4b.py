import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
import os

data_dir = sys.argv[1]
out_dir = sys.argv[2]
dataX = os.path.join(sys.argv[1], 'q4x.dat')
dataY = os.path.join(sys.argv[1], 'q4y.dat')

# 4. Gaussian and Discriminant Analysis
trainX = np.loadtxt(dataX)
trainY = np.loadtxt(dataY, dtype=str)

def normalize(X):
	mu = np.mean(X)
	std = np.std(X)
	return (X - mu)/std

X = normalize(trainX)
Y = np.array([0 if y == 'Alaska' else 1 for y in trainY]).reshape(-1, 1)
m = len(Y)

def plotData():
	x1 = np.array([X[i,:] for i in np.where(Y == 0)[0]])
	x2 = np.array([X[i,:] for i in np.where(Y == 1)[0]])

	plt.plot(x1[:,0], x1[:,1], 'ro', marker = '.', label = 'Alaska')
	plt.plot(x2[:,0], x2[:,1], 'bo', marker = '+', label = 'Canada')
	plt.xlabel(r'$X_1$')
	plt.ylabel(r'$X_2$')
	plt.title('Training Data')
	plt.legend()
	training_data = os.path.join(sys.argv[2], "Q4bTrainingData.png")
	plt.savefig(training_data)
	plt.close()

plotData()