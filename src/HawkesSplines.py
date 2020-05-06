# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

class HawkesSplines():
    
    def __init__(self, events, order=3, internal=8):
        """
        Implementation of a non-parametric estimation of the kernel of a Hawkes process
        Based on n independent observations over [0,1]
        Baseline with one possible change-point
        """
        self.events = events
        self.n_realizations = len(self.events)
        self.order = order
        self.internal = internal
        
        # definition of the knots of the splines
        self.knots = np.array([0.0 for k in range(self.order -1)] \
                + [k/self.internal for k in range(self.internal +1)] \
                + [1.0 for k in range(self.order -1)])
    
    def compute_mle(self, start_params, delta0=0.8 , epsilon=1e-5):
        """
        Compute the semi-parametric MLE estimator with a time varying baseline
        """
        self.delta0 = delta0 # compute MLE for a fixed value of delta
        objective_function = lambda x :  \
            self.loglikelihood(x, self.delta0).ll

        bnds=((epsilon,None),(epsilon,None))
        for k in range(self.order + self.internal -1):
            bnds += ((None,None),)
        self.resoptim = minimize(fun = objective_function, x0 = start_params, bounds=bnds)
        return

    
    def loglikelihood(self, par, delta=0.8):
        """
        Computation of the opposite of the log-likelihood
        mu_0 and mu_1 are baseline parameters
        coeffs are free constraints-reparametrized (gamma') coefficients for the spline function
        """
        self.delta = delta
        
        self.ll = 0.0
 
        # parameters of the baseline
        self.mu_0 = par[0]
        self.mu_1 = par[1]
        
        # parameters for the kernel
        self.coeffs = par[2:]
        self.set_params() # for the parametrization with decreasing sequence

        # define the spline function for kernel approximation
        # define also the antiderivative
        self.spl = BSpline(self.knots, self.newcoeffs, self.order)
        self.spli = self.spl.antiderivative()        
            
        for k in range(self.n_realizations):
            self.ll -= self.llcompute(obs_index = k)
       
        return self
    
    def llcompute(self,obs_index=0):
        """
        Log likelihood for one observation
        """
        self.setdeltaindex(obs_index=obs_index) 
        return self.G() + self.G(calc_type = 1) - self.B()
    
    def set_params(self):
        """
        transformation of the coefficients (gamma' -> gamma)
        return : decreasing sequence of positive numbers
        """
        temp = np.exp(self.coeffs) 
        self.newcoeffs = np.flip(np.cumsum(np.flip(temp,0)),0)
        return
    
    def setdeltaindex(self,obs_index=0):
        """
        set the index corresponding to delta
        """
        self.obs_index = obs_index
        temp = np.where(self.events[obs_index] < self.delta)[0]
        if (temp.size != 0) :
            self.deltaindex = temp.max() +1
        else :
            self.deltaindex = 0
        self.nticks = self.events[obs_index].shape[0]
        return
    
    def A(self,i):
        """
        Compute A(t_i)
        """
        time = self.events[self.obs_index][:i]
        inter_time = self.events[self.obs_index][i] - time
        return np.sum(self.spl(inter_time))
    
    def G(self,calc_type=0):
        """
        Compute G(delta)
        """
        if (calc_type == 0):
            temp = 0
            for i in range(self.deltaindex):
                temp += np.log(self.mu_0 + self.A(i) )
            return temp - self.mu_0*self.delta
        elif (calc_type == 1):
            temp = 0
            for i in range(self.deltaindex,self.nticks):
                temp += np.log(self.mu_1 + self.A(i) )
            return temp - self.mu_1*(1 - self.delta)
        
    def B(self):
        """
        compute B()
        """
        time = 1 - self.events[self.obs_index]
        temp = self.spli(time) - self.spli(0) # compute the integral of the kernel
        return np.sum(temp)
        
    def kernel_viz(self,window=1):
        """
        Plot the kernel
        """
        xx = np.linspace(0, window, 50)
        plt.plot(xx, self.spl(xx), 'b-')
        plt.show()
        return
    
    
    
    