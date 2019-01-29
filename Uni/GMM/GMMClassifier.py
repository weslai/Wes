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


import numpy
# from GaussianMixtureModel import GaussianMixtureModel
from PyClassificationToolbox_students.GaussianMixtureModel import GaussianMixtureModel


class GMMClassifier(object):

    def __init__(self, numComponentsPerClass, maxIterations):
        self._numComponentsPerClass = numComponentsPerClass
        self._maxIterations = maxIterations
        self._numCOmponents = numpy.sum(self._numComponentsPerClass)

        self._prior = {}
        self._mu = {}
        self._covar = {}

    def fit(self, X, y):
        # the label of y [0.0, 2.0, 6.0] in gmm.py
        self._ylabel = []
        for i in y:
            if i not in self._ylabel:
                self._ylabel.append(i)
        self._nbClasses = len(self._ylabel)

        # prior from y
        self._Proy = numpy.zeros((self._nbClasses, 1))

        for i in range(self._nbClasses):
            # calculate the prior of y from training data
            self._Proy[i] = len(y[y == self._ylabel[i]]) / len(y)
            # the prior, mu, and covariance matrix from each color
            self._prior["class" + str(i)], self._mu["class" + str(i)], self._covar["class" + str(i)] = GaussianMixtureModel(numComponents=self._numComponentsPerClass[i]).fit(
                X[y == self._ylabel[i]])


    def predict(self, X):
        L = numpy.zeros((X.shape[0], self._nbClasses))
        for i in range(self._nbClasses):
            gaussLikelihood = GaussianMixtureModel(numComponents=self._numComponentsPerClass[i]).evaluateLikelihood(X, self._prior["class" + str(i)],
                                                                                                 self._mu["class" + str(i)],
                                                                                                 self._covar["class" + str(i)])

            gaussLikelihood *= self._Proy[i]
            L[:, i] = gaussLikelihood

        # the largest Likelihood to decide the color Label
        Label = numpy.argmax(L, axis= 1)



        # assign to orginal label
        # tricky, because ylabel = [0.0, 2.0, 6.0] in gmm.py
        # so I did reassign the new Labels, which were assigned to 2.0, to 6.0. Because 6.0 was before assigned to 2
        for i in reversed(range(self._nbClasses)):
            Label[Label == i] = self._ylabel[i]

        
        return Label
