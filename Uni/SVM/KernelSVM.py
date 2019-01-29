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


import cvxopt
import numpy


class KernelSVM(object):

    def __init__(self, C = 1.0, gamma = 0.5):
        self.__C = C
        self.__gamma = gamma

    def fit(self, X, y):
        # training data
        self.__X = X
        self.__y = y

        # assign two label +1 and -1 to two classes in y
        self.__y[self.__y == self.__y[0]] = 1
        self.__y[self.__y != self.__y[0]] = -1
        self.__y = numpy.reshape(self.__y, (self.__y.size, 1))


        # fit the quadratic optimization problem into cvxopt.solvers.qp

        kernel = self.GaussianRBFKernelMatrix(self.__X, self.__X)

        P = cvxopt.matrix(numpy.outer(self.__y, self.__y) * kernel)
        
        q = cvxopt.matrix(-1 * numpy.ones((self.__y.shape[0], 1)))

        G = cvxopt.matrix(numpy.vstack((numpy.identity(self.__y.shape[0]) * -1, numpy.identity(self.__y.shape[0]))))
        
        h = cvxopt.matrix(numpy.hstack((numpy.zeros(self.__y.shape[0]), numpy.ones(self.__y.shape[0]) * self.__C )))
        
        A = cvxopt.matrix(self.__y.T)
        
        b = cvxopt.matrix(0.0)
        
        # optSol = cvxopt.solvers.qp(P, q, G, h, A, b, kktsolver='ldl', options={'kktreg':1e-9})
        optSol = cvxopt.solvers.qp(P, q, G, h, A, b)
        # print(optSol['x'])

        #
        # this is phi that we have from cvxopt.solvers.qp
        fi = numpy.array(optSol['x'])

        # fi that bigger than thread is actually the Support Vectors
        indSV = (fi > 1e-4)
        # Ns is the number of the Support Vectors
        Ns = (indSV == True).sum()
        index = numpy.nonzero(indSV == True)

        # valKernel is the relations between support vectors inside the Gaussian Kernel
        valKernel = kernel[:, index[0]]
        valKernel = valKernel[index[0],:]

        # compute the bias
        temp = fi[indSV] * self.__y[indSV]
        bias = self.__y[indSV].T - numpy.dot(temp.T, valKernel)

        self.__bias = bias.sum() / Ns
        self.__fi = fi[fi > 1e-4]
        self.__indexSV = index[0]


    def GaussianRBFKernelMatrix(self, X1, X2):

        kernel = numpy.zeros((X1.shape[0], X2.shape[0]))

        d1 = numpy.square(X1).sum(axis = 1)
        d2 = numpy.square(X2).sum(axis = 1)
        D = -2 * numpy.dot(X1, X2.T)
        kernel += D
        kernel += d1.reshape((-1,1))
        kernel += d2.reshape((1,-1))
        kernel = numpy.exp((abs(kernel) / (-2 * self.__gamma * self.__gamma)))
        
        return kernel


    def predict(self, X, mapping = True):
    	# fit the testing data to SVM model

        kernel = self.GaussianRBFKernelMatrix(self.__X[self.__indexSV, :], X)
        fi = self.__fi.reshape((-1,1))
        temp = fi * self.__y[self.__indexSV]
        f = numpy.dot(temp.T, kernel) + self.__bias
        result = numpy.sign(f)
        return result

