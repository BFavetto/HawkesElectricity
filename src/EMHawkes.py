# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 21:09:21 2018

@author: benja
"""

import numpy as np
from EMutils import relative_distance
import matplotlib.pyplot as plt


class EM():
    
    def __init__(self, events, tol=1e-5, max_iter = 100, end_time = 1):
        """
        Univariate Hawkes process with non constant baseline over [0,1]
        """

        self.events = events
        self.tol = tol
        self.max_iter = max_iter
        self.end_time = end_time
        self.n_realizations = len(self.events)
    
    
    def setEMparam(self, start_params, delta = 0.8):
        """
        initialize the EM parametric algorith (baseline with one change point)
        default value : no change point
        """        
        
        self.delta = delta
        
        # parameter of the baseline
        self.mu_0 = start_params[0]
        self.mu_1 = start_params[1]

        # parameters for the kernel
        self.alpha = start_params[2]
        self.beta = start_params[3] 
         
        return
        
    def setEMnonparam(self,start_params, kernel_start=None, kernel_support=1, n_bins_kernel=10, delta=0.8):
        """
        Initialization : Non parametric estimation of the kernel (histogram)
        """
        self.kernel_support = kernel_support
        self.n_bins_kernel = n_bins_kernel
        
        # parameter of the baseline
        self.mu_0 = start_params[0]
        self.mu_1 = start_params[1]
        self.delta = delta
        
        # steps of the grid 
        self.delta_kernel = self.kernel_support / self.n_bins_kernel
        
        # case of a small kernel support (support != 1)
        self.kernel_scaling = (self.kernel_support < 1)
        
        self.kernel_start = kernel_start
        
        if self.kernel_start is None:
            self.kernel = 0.5 * np.random.uniform(size=self.n_bins_kernel)
        else:
            if (self.kernel_start.shape[0] != self.n_bins_kernel ) :
                raise ValueError('kernel_start has wrong shape ')
            self.kernel = self.kernel_start.copy()
        
        #auxiliary computation
        self.dtab=np.zeros(self.n_bins_kernel)
        for k in range(self.n_bins_kernel):
            for r in range(self.n_realizations):
                n_ticks = self.events[r].shape[0]
                for i in range(n_ticks):
                    self.dtab[k] += self.d(self.events[r][i],k)
        
        return
    
    def EMnonparam(self):
        """
        Perform non parametric estimation of the kernel
        one change point in the baseline
        """
        # beginning of the EM loop    
        for i in range(self.max_iter + 1):

            # initialization
            self.next_mu_0 = np.zeros(self.n_realizations)
            self.next_mu_1 = np.zeros(self.n_realizations)
            self.next_kernel = np.zeros((self.n_realizations, self.n_bins_kernel))
  
            for r in range(self.n_realizations):
                # contains for one realization the bins indices
                self.compute_indices(real_index = r)
                
                self.EM_nonparam_update(real_index = r) # update for one realization
                        
            prev_mu_0 = self.mu_0
            prev_mu_1 = self.mu_1
            prev_kernel = self.kernel
            
            # update based on the entire dataset
            self.mu_0 = np.mean(self.next_mu_0)/self.delta
            self.mu_1 = np.mean(self.next_mu_1)/(self.end_time - self.delta)
            self.kernel = np.sum(self.next_kernel, axis =0)/(self.dtab)
        
            # test of convergence
            rel_baseline = relative_distance(np.array([self.mu_0, self.mu_1]) ,\
                                np.array([prev_mu_0, prev_mu_1]))
            rel_kernel = relative_distance(self.kernel, prev_kernel)

            converged = max(rel_baseline, rel_kernel) <= self.tol
            
            if converged:
                break
        
        return 
        return self
    
    def EMparam(self):
        """
        EM parametric algorithm (exponential kernel, change point in the baseline)
        """
        for i in range(self.max_iter + 1):

            # initialization
            self.next_mu_0 = np.zeros(self.n_realizations)
            self.next_mu_1 = np.zeros(self.n_realizations)
            self.next_num = np.zeros(self.n_realizations)
            self.next_alpha_denom = np.zeros(self.n_realizations)
            self.next_beta_denom = np.zeros(self.n_realizations)
  
            for r in range(self.n_realizations):
                
                self.EM_param_update(real_index = r) # update for one realization
            
            prev_mu_0 = self.mu_0
            prev_mu_1 = self.mu_1
            prev_alpha = self.alpha
            prev_beta = self.beta
            
            # update based on the entire dataset
            self.mu_0 = np.mean(self.next_mu_0)/self.delta
            self.mu_1 = np.mean(self.next_mu_1)/(self.end_time - self.delta)
            self.alpha = np.sum(self.next_num) / np.sum(self.next_alpha_denom)
            self.beta = np.sum(self.next_num) / np.sum(self.next_beta_denom)
        
            # test of convergence
            rel_dist_params = relative_distance(np.array([self.mu_0, self.mu_1 , self.alpha,self.beta]) ,\
                                np.array([prev_mu_0, prev_mu_1, prev_alpha, prev_beta]))
           
            if (rel_dist_params <= self.tol):
                break
        
        return
    
    def EM_nonparam_update(self, real_index = 0):
        """
        update the kernel with a given realization
        """
        n_ticks = self.events[real_index].shape[0]
        probs = np.zeros((n_ticks,n_ticks))
        
        # Expectation step
        for i in range(n_ticks):

            if (self.events[real_index][i] < self.delta) :
                probs[i][i] = self.mu_0
            else :
                probs[i][i] = self.mu_1
            
            for j in range(i):
                temp = self.events[real_index][i] - self.events[real_index][j]
                if (temp < self.kernel_support):
                    int_temp_kernel = int(np.floor(temp/self.delta_kernel))
                    probs[i][j] = self.kernel[int_temp_kernel]
                    
            probs[i][:] /= np.sum(probs[i][:])
        
        # Maximization step   
        for i in range(n_ticks):
            #for the baseline
            if (self.events[real_index][i] < self.delta) :
                self.next_mu_0[real_index] += probs[i][i]
            else :
                self.next_mu_1[real_index] += probs[i][i]

        
        for ell in range(self.n_bins_kernel):
            temp = 0.0
            for item in self.bins_kernel[ell]:
                temp += probs[item[0]][item[1]]
            self.next_kernel[real_index][ell] = temp
        return
    
    def EM_param_update(self, real_index=0):
        """
        update the parameters in the EM algorithm for one realization
        """
        n_ticks = self.events[real_index].shape[0]
        probs = np.zeros((n_ticks,n_ticks))
        
        # Expectation step with one change point
        for i in range(n_ticks):
            
            if (self.events[real_index][i] < self.delta) :
                probs[i][i] = self.mu_0
            else :
                probs[i][i] = self.mu_1
            
            for j in range(i):
                probs[i][j] = self.alpha * self.beta * np.exp(- self.beta* \
                     (self.events[real_index][i] - self.events[real_index][j]))
                
            probs[i][:] /= np.sum(probs[i][:])
        
        # Maximization step       
        for i in range(n_ticks):
            
            if (self.events[real_index][i] < self.delta) :
                self.next_mu_0[real_index] += probs[i][i]
            else :
                self.next_mu_1[real_index] += probs[i][i]
            
            for j in range(i):
                self.next_num[real_index] += probs[i][j]
                self.next_beta_denom[real_index] += \
                    (self.events[real_index][i] - self.events[real_index][j]) \
                    * probs[i][j]
            
            temp =  self.end_time - self.events[real_index][i]
            self.next_alpha_denom[real_index] += 1 - np.exp(- self.beta *temp)
            self.next_beta_denom[real_index] += self.alpha \
                    * temp * np.exp(- self.beta *temp)
        return
    
    def compute_indices(self, real_index = 0):
        """
        For a given array of ordered ticks (one realization) 
        compute the index lists
        """
        
        n_ticks = self.events[real_index].shape[0]
        self.bins_kernel = []
        # kernel indices computation
        for k in range(self.n_bins_kernel) :
            self.bins_kernel.append([])
            
        for i in range(1,n_ticks) :
            j = i-1
            for k in range(self.n_bins_kernel): 
                while ((self.events[real_index][i] \
                       - self.events[real_index][j]) < (k+1)*self.delta_kernel) \
                       and (j >= 0):
                           self.bins_kernel[k].append((i,j))       
                           j -= 1
                    
        return
    
    def d(self,t,k):
        """
        Compute the value of an integral involved in the log-likelihood function
        """
        return max(min(self.end_time -t,(k+1)*self.delta_kernel) - k*self.delta_kernel , 0)
 