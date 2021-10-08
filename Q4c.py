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

def covMatrix(mu0, mu1):
	mu0 = mu0.reshape(-1, 1)
	mu1 = mu1.reshape(-1, 1)
	sigma = 0
	for i in np.where(Y == 0)[0]:
		a = X[i].reshape(-1, 1)
		sigma += np.dot((a - mu0), (a - mu0).T)

	for i in np.where(Y == 1)[0]:
		a = X[i].reshape(-1, 1)
		sigma += np.dot((a - mu1), (a - mu1).T)

	return sigma/float(m)

def GDA(x, y):
	label0 = (y == 0).sum()
	label1 = (y == 1).sum()

	phi = float(label1)/m
	mu0 = np.sum((x * (y == 0)), axis = 0)/float(label0)
	mu1 = np.sum((x * (y == 1)), axis = 0)/float(label1)
	sigma = covMatrix(mu0, mu1)
	sigma0 = np.dot(((x - mu0)*(y == 0)).T, (x - mu0)*(y == 0))/float(label0)
	sigma1 = np.dot(((x - mu1)*(y == 1)).T, (x - mu1)*(y == 1))/float(label1)

	return phi, mu0, mu1, sigma, sigma0, sigma1

phi, mu0, mu1, sigma, sigma0, sigma1 = GDA(X, Y)

def computeYLinear(x):
	sigma_inv = np.linalg.pinv(sigma)
	p = np.dot(mu1, sigma_inv) - np.dot(mu0, sigma_inv)
	q = np.log(phi/float(1 - phi)) + 0.5*(np.dot(np.dot(mu0, sigma_inv), mu0.T) - np.dot(np.dot(mu1, sigma_inv), mu1.T))
	yv = -1*(q + p[0]*x)/p[1]
	return yv

def plotLinearBoundary():
	x1 = np.array([X[i,:] for i in np.where(Y == 0)[0]])
	x2 = np.array([X[i,:] for i in np.where(Y == 1)[0]])

	plt.plot(x1[:,0], x1[:,1], 'ro', marker = '.', label = 'Alaska')
	plt.plot(x2[:,0], x2[:,1], 'bo', marker = '+', label = 'Canada')

	axes = plt.gca()
	xv = np.array(axes.get_xlim())
	yv = computeYLinear(xv)

	plt.plot(xv, yv, color = 'Green')
	plt.xlabel(r'$X_1$')
	plt.ylabel(r'$X_2$')
	plt.title('Linear Decision Boundary')
	plt.legend()
	linear_boundary = os.path.join(sys.argv[2], "Q4cLinear.png")
	plt.savefig(linear_boundary)
	plt.close()

plotLinearBoundary()