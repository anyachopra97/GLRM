'''
Created on May 24, 2017

@author: anyachopra
'''
from numpy import sqrt, repeat, tile, hstack, array, zeros, ones, sqrt, diag, asarray, hstack, vstack, split, cumsum
from numpy.random import randn
from abc import ABCMeta, abstractmethod

class GLRM(object):

    def __init__(A, losses, regularizer, ry, k, X, Y, obs = None, observed_features, observed_examples, offset = False, scale = False, checknan = True, sparse_na = True):
        self.scale = scale
        if X is None:
            X = numpy.random.randn(k, len(A))
        if Y is None:
            Y = numpy.random.randn(k, embedding_dim()) #figure out embedding_dim meaning
        if observed_features is None:
            numpy.full(len(A), list(range(1, len(A[0]))))
        if observed_examples is None:
            numpy.full(len(A[0]), list(range(1, len(A))))
            m = len(A)
            n = len(A[0])
        if len(losses) != n:
            raise ValueError("There must be as many losses as there are columns in the data matrix")
        if len(ry) != n:
            raise ValueError("There must be ...")
        if len(X) != k or len(X[0]) != m:
            raise ValueError("There must be ...")
        if len(Y) != k or len(Y[0]) != sum_embedded_dim():
            raise ValueError("There must be ...") #sum of embedding dimensions of all the losses
        if obs is None and sparse_na and isSparseMatrix():
            I,J = numpy.nonzero(A)
            for i in range(1, len(I)):
                numpy.array([(I[i],J[i]) for i in range(1, len(I))])
        if obs is None:
            glrm = GLRM(A, losses, regularizer, ry, k, observed_features, observed_examples, X, Y)
        else:
            glrm = GLRM(A, losses, regularizer, ry, k, X, Y)
        
    #check to make sure X is properly oriented
        if len(glrm.X) != k or len(glrm.X[0]) != len(A):
        #transposing X
            grlm.X = grlm.X.transpose()
        if chacknan:
            for i in range(1, len(A)):
                for j = glrm.observed_features[i]:
                    if numpy.isnan(A[i][j]):
                        raise ValueError("Observed value in entry (i,j) is not a number (NaN)")
    
        
    def embedding_dim():
        
    def sum_embedded_dim():

    def isSparseMatrix():
        
    return self.X, self.Y

class lastentry1(object):
    
class Regularizer(object):
    @abstractmethod
    def prox(self, regularizer, array, alpha):
        
    def proxnot(self, regularizer, array, alpha):
        def v = prox(regularizer, array, alpha):
        for i = 1 to length(array):
            array[i] = v[i]
        return array
        
    def scale(self, regularizer):
        return regularizer.scale

    def scalenot(self, regularizer, newscale):
        if isinstance(array, regularizer):
            for r in regularizer:
                regularizer.scalenot(regularizer, newscale)
            return regularizer
        else:
            regularizer.scale = newscale
            return r   
      
    @abstractmethod
    def evaluate(self, regularizer, array):

class ZeroReg(Regularizer):
    def prox(self, regularizer, array, alpha):
        return array
    def proxnot(self, regularizer, array, alpha):
        return array
    def evaluate(self, regularizer, array):
        return 0
    def scale(self):
        return 0
    def scalenot(self, newscale):
        return 0
    
class OneReg(Regularizer, float scale = 1):
    def prox(self, regularizer, array, alpha):
        return max(array-alpha,0)
    def proxnot(self, regularizer, array, alpha):
        def softthreshold(x):
            return max(x-alpha, 0) + min(x+alpha, 0)
        map(softthreshold(), array)
    def evaluate(self, regularizer, array):
        return regularizer.scale * sum(numpy.absolute(array))
    
class QuadReg(Regularizer, float64 scale = 1):
    def prox(self, regularizer, array, alpha):
        return 1/(1+2*alpha*regularizer.scale)*array
    def proxnot(self, regularizer, array, alpha):
        regularizer.scalenot(array, 1/(1+2*alpha*regularizer.scale))
    def evaluate(self, regularizer, array):
        return regularizer.scale * sum(numpy.absolute(array)*numpy.absolute(array))
    
class QuadConstraint(Regularizer, float64 max_2norm = 1):
    def prox(self, regularizer, array, alpha):
        return regularizer.max_2norm/(numpy.linalg.norm(array))*array
    def proxnot(self, regularizer, array, alpha):
        return regularizer.scalenot(array,regularizer.max_2norm/(numpy.linalg.norm(array)))
    def scale(self):
        return 1
    def scalenot(self, newscale):
        return 1
    def evaluate(self, regularizer, array):
        if numpy.linalg.norm(array) > regularizer.max_2norm + 1e-12:
            return float('inf')
        else:
            return 0
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        