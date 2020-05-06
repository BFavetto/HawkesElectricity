# -*- coding: utf-8 -*-

import numpy as np
from EMutils import relative_distance

class EM():
    
    def __init__(self, kernel_support=1, n_bins_baseline=10, \
                 n_bins_kernel=10, tol=1e-5, max_iter = 100):
        """
        Univariate Hawkes process with non constant baseline over [0,1]
        Different time scales for the baseline and the kernel 
        """
        self.kernel_support = kernel_support
        self.n_bins_baseline = n_bins_baseline
        self.n_bins_kernel = n_bins_kernel
        self.tol = tol
        self.max_iter = max_iter
        
        self.baseline = None
        self.kernel = None
        
        
    def fit(self, events, baseline_start=None, kernel_start=None):
        """
        Set the corresponding realization(s) of the process.
        Parameters
        ----------
        events :  `list` of `np.ndarray`
            List of Hawkes processes realizations.
            Each realization of the Hawkes process is a list. 
            Namely `events[i]` contains a
            one-dimensional `numpy.array` of the events' timestamps 
            of realization i.
        baseline_start and kernel_start are preliminary estimations
        """
        self.events = events
        self.n_realizations = len(self.events)
        self.baseline_start = baseline_start
        self.kernel_start = kernel_start
        
        # steps of the grid 
        self.delta_baseline = 1 / self.n_bins_baseline
        self.delta_kernel = self.kernel_support / self.n_bins_kernel
        
        # case of a small kernel support (support != 1)
        self.kernel_scaling = (self.kernel_support < 1)
        
        self.solve()
        return self
        
    
    def solve(self):
        """
        Performs nonparametric estimation and stores the result in the
        attributes `kernel` and `baseline`
        Parameters
        ----------
            
        """
        # if the time scale for kernel is different from the standard case 
        # (support != [0,1]) -> an extra interval is added to avoid side effect
        if self.kernel_start is None:
            self.kernel = 0.5 * np.random.uniform(size=self.n_bins_kernel)
        else:
            if (self.kernel_start.shape[0] != self.n_bins_kernel ) :
                raise ValueError('kernel_start has wrong shape ')
            self.kernel = self.kernel_start.copy()

        if self.baseline_start is None:
            self.baseline = np.zeros(self.n_bins_baseline) + 1
        else:
            if (self.baseline_start.shape[0] != self.n_bins_baseline ) :
                raise ValueError('baseline_start has wrong shape ')
            self.baseline = self.baseline_start.copy()

        # computation of the total number of ticks
        self.total_ticks = 0
        for r in range(self.n_realizations):
            self.total_ticks += self.events[r].shape[0]
            
        #auxiliary computation
        self.dtab=np.zeros(self.n_bins_kernel)
        for k in range(self.n_bins_kernel):
            for r in range(self.n_realizations):
                n_ticks = self.events[r].shape[0]
                for i in range(n_ticks):
                    self.dtab[k] += self.d(self.events[r][i],k)

        # beginning of the EM loop    
        for i in range(self.max_iter + 1):

            # initialization
            self.next_baseline = np.zeros((self.n_realizations, self.n_bins_baseline))
            self.next_kernel = np.zeros((self.n_realizations, self.n_bins_kernel))
  
            for r in range(self.n_realizations):
                # contains for one realization the bins indices
                self.bins_baseline = [] # contains for one realization the bins indices
                self.bins_kernel = []
        
                self.compute_indices(real_index = r)
                self.update(real_index = r) # update for one realization
            
            prev_baseline = self.baseline
            prev_kernel = self.kernel
            
            # update based on the entire dataset
            self.baseline = np.mean(self.next_baseline, axis = 0)/self.delta_baseline
            self.kernel = np.sum(self.next_kernel, axis =0)/(self.dtab)
        
            # test of convergence
            rel_baseline = relative_distance(self.baseline, prev_baseline)
            rel_kernel = relative_distance(self.kernel, prev_kernel)

            converged = max(rel_baseline, rel_kernel) <= self.tol
            
            if converged:
                break
        
        return 
    
    def d(self,t,k):
        """
        Compute the value of an integral involved in the log-likelihood function
        """
        if (self.kernel_scaling) and (k+1 == self.n_bins_kernel) :
            return (1 - t - self.kernel_support) 
        else :
            return max( min(1-t,(k+1)*self.delta_kernel) - k*self.delta_kernel , 0)
        
    def compute_indices(self, real_index = 0):
        """
        For a given array of ordered ticks (one realization) 
        compute the index lists
        """
        
        n_ticks = self.events[real_index].shape[0]
        
        # baseline indices computation
        stopindex = 0
        for k in range(self.n_bins_baseline) :
            temp = []
            while (stopindex < n_ticks) and \
                (self.events[real_index][stopindex] < (k+1)*self.delta_baseline) : 
                    temp.append(stopindex)
                    stopindex += 1
            self.bins_baseline.append(temp)
        
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
     
    def update(self, real_index = 0):
        """
        update the baseline and the kernel with a given realization
        """
        n_ticks = self.events[real_index].shape[0]
        probs = np.zeros((n_ticks,n_ticks))
        
        # Expectation step
        for i in range(n_ticks):
            int_temp_baseline = int(np.floor(self.events[real_index][i]) / self.delta_baseline)
            probs[i][i] = self.baseline[int_temp_baseline]
            
            for j in range(i):
                temp = self.events[real_index][i] - self.events[real_index][j]
                if (temp < self.kernel_support):
                    int_temp_kernel = int(np.floor(temp/self.delta_kernel))
                    probs[i][j] = self.kernel[int_temp_kernel]
                    
            probs[i][:] /= np.sum(probs[i][:])
        
        # Maximization step       
        for ell in range(self.n_bins_baseline):            
            temp = 0.0
            for item in self.bins_baseline[ell]:
                temp += probs[item][item]
            self.next_baseline[real_index][ell] = temp
        
        for ell in range(self.n_bins_kernel):
            temp = 0.0
            for item in self.bins_kernel[ell]:
                temp += probs[item[0]][item[1]]
            self.next_kernel[real_index][ell] = temp
        return
        
        
        
            
                
            
        

        
        
        
        
        
        
        
        

