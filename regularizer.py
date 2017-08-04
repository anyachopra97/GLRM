'''
Created on Jun 1, 2017

@author: anyachopra
'''

from numpy import absolute
from numpy.linalg import norm
from numpy.random import randn
from abc import ABCMeta, abstractmethod

class Regularizer(object):
    @abstractmethod
    def prox(self, regularizer, array, alpha):
        return
    
    def _prox(self, regularizer, array, alpha):
        for i in 1 to length(array):
            array[i] = prox(regularizer, array, alpha)[i]
        return array
        
    def scale(self, regularizer):
        return regularizer.scale

    def _scale(self, regularizer, newscale):
        if isinstance(array, regularizer):
            for r in regularizer:
                regularizer._scale(regularizer, newscale)
            return regularizer
        else:
            regularizer.scale = newscale
            return r   
      
    @abstractmethod
    def evaluate(self, regularizer, array):

class ZeroReg(Regularizer):
    def prox(self, regularizer, array, alpha):
        return array
    def _prox(self, regularizer, array, alpha):
        return array
    def evaluate(self, regularizer, array):
        return 0
    def scale(self):
        return 0
    def _scale(self, newscale):
        return 0
    
class OneReg(Regularizer, float scale = 1):
    def prox(self, regularizer, array, alpha):
        return softthreshold()
    def _prox(self, regularizer, array, alpha):
        def softthreshold(x):
            return max(x-alpha, 0) + min(x+alpha, 0)
        map(softthreshold(), array)
    def evaluate(self, regularizer, array):
        return regularizer.scale * sum(numpy.absolute(array))
    
class QuadReg(Regularizer, float64 scale = 1):
    def prox(self, regularizer, array, alpha):
        return 1/(1+2*alpha*regularizer.scale)*array
    def _prox(self, regularizer, array, alpha):
        regularizer._scale(array, 1/(1+2*alpha*regularizer.scale))
    def evaluate(self, regularizer, array):
        return regularizer.scale * numpy.linalg.norm(array, ord = 2) #check syntax
    
class QuadConstraint(Regularizer, float64 max_2norm = 1):
    def prox(self, regularizer, array, alpha):
        return regularizer.max_2norm/(numpy.linalg.norm(array))*array
    def _prox(self, regularizer, array, alpha):
        return regularizer._scale(array,regularizer.max_2norm/(numpy.linalg.norm(array)))
    def scale(self):
        return 1
    def _scale(self, newscale):
        max_2norm = newscale
        return 1
    def evaluate(self, regularizer, array):
        if numpy.linalg.norm(array) > regularizer.max_2norm + 1e-12:
            return float('inf')
        else:
            return 0
    
class NonNegConstraint(Regularizer):
#FINISH

class NonNegOneReg(Regularizer, float64 scale = 1):
    def prox(self, regularizer, array, alpha):
        return max(array - alpha, 0)
    def _prox(self, regularizer, array, alpha):
        def nonnegsoftthreshold(x):
            return max(x - alpha,0)
        return map(nonnegsoftthreshold(), array)
    def scale(self):
        return 1
    def _scale(self, newscale):
        return 1
    def evaluate(self, regularizer, array):
        for a in array:
            if a < 0:
                return float('inf')
            else:
                return regularizer.scale * sum(a)
        
class NonNegQuadReg(Regularizer, float64 scale = 1):
    def prox(self, regularizer, array, alpha):
        return max(1/(1+2*alpha*regularizer.scale)*array, 0)
    def _prox(self, regularizer, array, alpha):
        regularizer._scale(array, 1/(1+2*alpha*regularizer.scale))
        maxval = numpy.amax(array)
        return numpy.clip(array, 0, maxval)
    def evaluate(self, regularizer, array):
        for a in array:
            if a < 0:
                return float('inf')
            else:
                return regularizer.scale * numpy.linalg.norm(array, ord = 2) #check syntax
            
        