#    Copyright 2016 Stefan Steidl
#    Friedrich-Alexander-Universität Erlangen-Nürnberg
#    Lehrstuhl für Informatik 5 (Mustererkennung)
#    Martensstraße 3, 91058 Erlangen, GERMANY
#    stefan.steidl@fau.de


#    This file is part of the Python Classification Toolbox.
#
#    The Python Classification Toolbox is free software: 
#    you can redistribute it and/or modify it under the terms of the 
#    GNU General Public License as published by the Free Software Foundation, 
#    either version 3 of the License, or (at your option) any later version.
#
#    The Python Classification Toolbox is distributed in the hope that 
#    it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
#    See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with the Python Classification Toolbox.  
#    If not, see <http://www.gnu.org/licenses/>.


import math
import numpy
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from KMeansClustering import KMeansClustering
from PyClassificationToolbox_students.KMeansClustering import KMeansClustering

class GaussianMixtureModel(object):

    def __init__(self, numComponents, maxIterations = 500):
        self._numComponents = numComponents
        self._maxIterations = maxIterations

    # the Main EM Steps
    def fit(self, X):
        nbData, nbVar = X.shape
        prior0, mu0, cov0 = self.getComponents(X)
        r = self.evaluateGaussian(X, prior0, mu0, cov0)
        mu = numpy.zeros((self._numComponents, nbVar))

        cov = numpy.ndarray(shape=(2, 2, self._numComponents))

        for i in range(self._maxIterations):
            # M steps
            sumR = numpy.sum(r, axis= 0)

            # Mu calculation and Prior
            for j in range(self._numComponents):
                mu[j, :] = numpy.sum(r[:, j].reshape(nbData, 1) * X, axis= 0) / sumR[j]

            prior = numpy.sum(r, axis= 0).reshape(r.shape[1], 1)
            prior /= nbData


            # covariance Matrix calculation
            for j in range(self._numComponents):
                cov_temp = numpy.sum(r[:, j].reshape(nbData, 1) * numpy.square(X - mu[j, :]), axis= 0)
                cov[0, 0, j] = cov_temp[0] / numpy.sum(r[:, j])
                cov[1, 1, j] = cov_temp[1] / numpy.sum(r[:, j])
                cov_temp = X - mu[j, :]
                cov_temp = numpy.sum(r[:, j] * cov_temp[:, 0] * cov_temp[:, 1], axis= 0)
                cov[0, 1, j] = cov_temp / numpy.sum(r[:, j])
                cov[1, 0, j] = cov_temp / numpy.sum(r[:, j])
        #    E steps

            r = self.evaluateGaussian(X, prior, mu, cov)

        return prior, mu, cov

    # get the initializationvfrom KMeansCluster and initialize Prior Mu and Covariance Matrix
    def getComponents(self, X):
        nbData, nbVar = X.shape
        p= KMeansClustering(numClusters = self._numComponents).fit(X)
        plabel = [];
        for i in p:
            if i not in plabel:
                plabel.append(i)
        mu = numpy.zeros((self._numComponents, nbVar))
        priors = numpy.zeros((self._numComponents, 1))
        covar = numpy.ndarray(shape=(nbVar, nbVar, self._numComponents))
        for i in range(self._numComponents):
            num = p[p == plabel[i]]
            mu[i, :] = numpy.sum(X[p == plabel[i]], axis= 0) / nbData
            priors[i, 0] = num.size
            covar[:, :, i] = self.cov_mat(X.T)

        priors /= nbData

        return priors, mu, covar


    def getProbabilityScores(self, X):
        return None

    # this function is used to update the E step, for weights r
    def evaluateGaussian(self, X, prior, mean, cov):
        nbData, nbVar = X.shape
        r = numpy.zeros((nbData, self._numComponents))
        for i in range(self._numComponents):
            r[:, i] = prior[i, 0] * self.gaussPDF(X, mean[i, :], cov[:, :, i])


        r /= numpy.sum(r, axis= 1).reshape(-1, 1)
        return r

    def evaluateLikelihood(self, X, prior, mean, cov):
        nbData, nbVar = X.shape
        r = numpy.zeros((nbData, self._numComponents))
        for i in range(self._numComponents):
            r[:, i] = prior[i, 0] * self.gaussPDF(X, mean[i, :], cov[:, :, i])

        
        # likelihood = numpy.log(numpy.sum(r, axis= 1))
        likelihood = numpy.sum(r, axis=1)
        return likelihood

    def covDistance(self, covs1, covs2):
        return None


    # the gaussian PDF
    def gaussPDF(self, X, mu, covar):
        nbData, nbVar = X.shape
        # assume mu (1 * features)
        # assume covar (features * features)
        data = X - mu
        prob = numpy.sum(numpy.dot(data, numpy.linalg.inv(covar)) * data, axis= 1)
        prob = (1 / numpy.sqrt((numpy.power(2 * math.pi, nbVar)) * numpy.absolute(numpy.linalg.det(covar)))) * numpy.exp(-(1/2) * prob)

        return prob


    # calculate the covariance matrix for two feature vectors
    def cov(self, x, y):
        xmean = numpy.mean(x)
        ymean = numpy.mean(y)
        return numpy.sum((x - xmean) * (y - ymean)) / (len(x) - 1)

    def cov_mat(self, x):
        return numpy.array([[self.cov(x[0], x[0]), self.cov(x[0], x[1])],
                            [self.cov(x[1], x[0]), self.cov(x[1], x[1])]])

