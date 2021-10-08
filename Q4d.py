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
out = os.path.join(sys.argv[2], 'Q4d.txt')
outfile = open(out, "w")

# 4. Gaussian and Discriminant Analysis
print("################ 4. Gaussian and Discriminant Analysis ################", file=outfile)

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

print('phi = {}'.format(phi), file=outfile)
print('mu0 = {}'.format(mu0), file=outfile)
print('mu1 = {}'.format(mu1), file=outfile)
print('sigma = [{0},{1}]'.format(sigma[0], sigma[1]), file=outfile)
print('sigma0 = [{0},{1}]'.format(sigma0[0], sigma0[1]), file=outfile)
print('sigma1 = [{0},{1}]'.format(sigma1[0], sigma1[1]), file=outfile)