# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from EMutils import relative_distance
import matplotlib.pyplot as plt

class HawkesProcess():
    
    def __init__(self, events, end_time=1):
        """
        Univariate Hawkes process with non constant baseline over [0,1]
        Computation of the log-likelihood with AIC and BIC criteria
        Computation of the MLE estimator with change point
        Computation of the likelihood ratio test statistic
        """
        self.events = events
        self.end_time = end_time
        self.n_realizations = len(self.events)
        self.total_ticks = 0
        for j in range(self.n_realizations):  
            self.total_ticks += self.events[j].shape[0]
    
    def set_kernel_shape(self,kernel_shape="exp"):
        self.kernel_shape = kernel_shape
        return
    
    def compute_mle(self, start_params, delta0 , epsilon=1e-5):
        """
        Compute the MLE estimator with a time varying baseline
        """
        self.delta0 = delta0 # compute MLE for a fixed value of delta
        objective_function = lambda x :  \
            self.loglikelihood(x, self.delta0).ll
        bnds = ((epsilon, None),(epsilon,None),(epsilon,None))
        
        if (self.kernel_shape == "pow") :
            bnds += ((1+epsilon,None),(epsilon, None),) # case of beta_1, c in pow law kernel
        else :
            bnds += ((epsilon,None),) # case of beta_1 in exp kernel
        
        self.resoptim = minimize(fun = objective_function, x0 = start_params, \
            bounds = bnds)
        return
    
    def mle_delta(self, start_params, delta_grid, graphic_option=False):
        """
        Compute the MLE estimator of delta over a grid
        if graphic_option is True, plot the values of the ll for different
        values of delta
        grid is an array with values between 0 and 1
        """
        self.delta_grid = delta_grid
        s = delta_grid.shape[0]
        self.ll_grid = np.zeros(s)
        parameters = []
        for i in tqdm(range(s)):
            self.compute_mle(start_params,self.delta_grid[i])
            self.ll_grid[i] = -self.resoptim.fun
            parameters.append(self.resoptim.x)
        j = np.argmax(self.ll_grid)
        self.deltahat = self.delta_grid[j]
        
        if graphic_option :
            print(parameters[j])
            plt.plot(self.delta_grid,self.ll_grid)
            plt.show()
        self.mle_theta = parameters[j]
        return parameters[j]
    
    def variance_asympt(self):
        """
        Compute an approximation of the asymptotic variance matrix for the 
        smooth parameters
        """
        self.delta = self.deltahat # set delta_0 to its estimated value delta_hat
        self.mu_0 = self.mle_theta[0]
        self.mu_1 = self.mle_theta[1]
        self.beta_0 = self.mle_theta[2]
        self.beta_1 = self.mle_theta[3] # values of the parameters set to MLE values
        if (self.kernel_shape == "exp"):
            self.fischer_mat = np.zeros((4,4)) 
            for j in range(self.n_realizations):
                self.fischer_mat += self.fischer_obs(obs_index=j)
            self.fischer_mat /= self.n_realizations
        else:
            print("Not yet implemented")
        return
    
    def fischer_obs(self, obs_index=0):
        """
        Compute an approximation of the Fischer information matrix for one trajectory
        For the observation j-th
        """
        self.setdeltaindex(obs_index=obs_index)  
        return self.Gfischer() + self.Gfischer(calc_type = 1)
    
    
    def loglikelihood(self, param_array, delta, option_bic=0):
        """
        Computation of the opposite of the log-likelihood
        In the case of power law, assume beta_1 > 0
        option_bic defines the bic computation method
        """
        self.param_array = param_array
        self.delta = delta
        self.option_bic = option_bic
        
        self.ll = 0.0
 
        # parameters of the baseline
        self.mu_0 = self.param_array[0]
        self.mu_1 = self.param_array[1] 
        # parameters for the kernel
        self.beta_0 = self.param_array[2]
        self.beta_1 = self.param_array[3] 
        
        if (self.kernel_shape == "exp"):
            
            for j in range(self.n_realizations):  
                self.ll -= self.llexp(obs_index = j)
                
        elif (self.kernel_shape == "pow"):
            
            if (self.beta_1 <= 1):
                raise ValueError('parameter beta_1 must be greater that 1')
            
            # parameter of the kernel
            self.c = self.param_array[4]
            
            for j in range(self.n_realizations):
                self.ll -= self.llpow(obs_index = j)
        
        self.AIC()
        self.BIC(option=self.option_bic)        
        return self
    
    def llratio(self,param_array, delta):
        """
        Compute the likelihood ratio dP_1/dP_0 when the parameters are known
        """
        self.param_array = param_array
        self.delta = delta
        
        self.lr=0
        
        # parameters of the baseline
        self.mu_0 = self.param_array[0]
        self.mu_1 = self.param_array[1] 
        # parameters for the kernel
        self.beta_0 = self.param_array[2]
        self.beta_1 = self.param_array[3] 
        
        if (self.kernel_shape == "exp"):
            for j in range(self.n_realizations):  
                self.lr += self.lrexp(obs_index = j)
                
        elif (self.kernel_shape == "pow"):
            
            if (self.beta_1 <= 1):
                raise ValueError('parameter beta_1 must be greater that 1')
            
            # parameter of the kernel
            self.c = self.param_array[4]
            
            for j in range(self.n_realizations):
                self.lr += self.lrpow(obs_index = j)
        
        return self
    
    def setEMparam(self, start_params, tol=1e-5, max_iter = 100):
        """
        initialize the EM parametric algorith (constant baseline)
        """
        self.tol = tol
        self.max_iter = max_iter
        
        # parameter of the baseline
        self.mu = start_params[0]

        # parameters for the kernel
        self.alpha = start_params[1]
        self.beta = start_params[2] 
        
        if (self.kernel_shape == "pow"):
            self.c = start_params[3]
         
        return
    
    def EMparam(self):
        """
        EM parametric algorithm (exponential case)
        """
        
        if (self.kernel_shape == "pow"):
            raise ValueError("Not yet implemented")
            
        for i in range(self.max_iter + 1):

            # initialization
            self.next_mu = np.zeros(self.n_realizations)
            self.next_num = np.zeros(self.n_realizations)
            self.next_alpha_denom = np.zeros(self.n_realizations)
            self.next_beta_denom = np.zeros(self.n_realizations)
  
            for r in range(self.n_realizations):
                
                self.update(real_index = r) # update for one realization
            
            prev_mu = self.mu
            prev_alpha = self.alpha
            prev_beta = self.beta
            
            # update based on the entire dataset
            self.mu = np.mean(self.next_mu)/self.end_time
            self.alpha = np.sum(self.next_num) / np.sum(self.next_alpha_denom)
            self.beta = np.sum(self.next_num) / np.sum(self.next_beta_denom)
        
            # test of convergence
            rel_dist_params = relative_distance(np.array([self.mu,self.alpha,self.beta]) ,\
                                            np.array([prev_mu, prev_alpha, prev_beta]))
           
            if (rel_dist_params <= self.tol):
                break
        
        return
    
    def update(self, real_index=0):
        """
        update the parameters in the EM algorithm for one realization
        (constant baseline, exponential kernel)
        """
        n_ticks = self.events[real_index].shape[0]
        probs = np.zeros((n_ticks,n_ticks))
        
        # Expectation step
        for i in range(n_ticks):
            probs[i][i] = self.mu
            for j in range(i):
                probs[i][j] = self.alpha * self.beta * np.exp(- self.beta* \
                     (self.events[real_index][i] - self.events[real_index][j]))
            probs[i][:] /= np.sum(probs[i][:])
        
        # Maximization step       
        for i in range(n_ticks):
            self.next_mu[real_index] += probs[i][i]
            
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
                   
    
    def lrexp(self,obs_index=0):
        """
        Likelihood ratio with exponential kernel for one observation
        """
        self.setdeltaindex(obs_index=obs_index)
        return self.Grexp()
    
    def lrpow(self,obs_index=0):
        """
        Likelihood ratio with power law kernel for one  observation
        """
        self.setdeltaindex(obs_index=obs_index)      
        return self.Grpow()
    
    def AIC(self):
        """
        Compute the AIC value of the model
        """
        if (self.kernel_shape == "exp"):
            if (self.delta ==0 or self.delta == 1) :
                self.aic_value = 2*(3 + self.ll) # ll is the opposite of loglikelihood
            else :
                self.aic_value = 2*(5 + self.ll) # delta and mu_1 as extra parameters
        elif (self.kernel_shape == "pow"):
            if (self.delta ==0 or self.delta == 1) :
                self.aic_value = 2*(4 + self.ll) 
            else :
                self.aic_value = 2*(6 + self.ll) # delta and mu_1 as extra parameters
        else :
            self.aic_value = 0
        return

    def BIC(self, option=0):
        """
        Compute the BIC value of the model
        option: 0 if we consider the number of observed hawkes processes
        option: 1 if we consider the number of ticks
        """
        self.bic_value = 0
        if (option ==0):
            N = self.n_realizations
        elif (option ==1):
            N = self.total_ticks
        
        if (self.kernel_shape == "exp"):
            if (self.delta ==0 or self.delta == 1) :
                self.bic_value = 3*np.log(N) + 2*self.ll # ll is the opposite of loglikelihood
            else :
                self.bic_value = 5*np.log(N) + 2*self.ll # delta and mu_1 as extra parameters
        elif (self.kernel_shape == "pow"):
            if (self.delta ==0 or self.delta == 1) :
                self.bic_value = 4*np.log(N) + 2*self.ll 
            else :
                self.bic_value = 6*np.log(N) + 2*self.ll # delta and mu_1 as extra parameters       
        return

    def llexp(self,obs_index=0):
        """
        Log likelihood with exponential kernel for one observation
        """
        self.setdeltaindex(obs_index=obs_index)            
        return self.Gexp() + self.Gexp(calc_type = 1) - self.Bexp()
    
    def llpow(self, obs_index=0):
        """
        Log likelihood with power law kernel for one observation
        """
        self.setdeltaindex(obs_index=obs_index)
        return self.Gpow() + self.Gpow(calc_type = 1) - self.Bpow()
    
    def Aexp(self,i):
        """
        Compute A(t_i)
        """
        ticks = self.events[self.obs_index][:i]
        inter_time = self.events[self.obs_index][i] - ticks
        return self.beta_0 * np.sum(np.exp(- self.beta_1 * inter_time))
    
    def Abis(self,i):
        """
        Compute Abis(t_i) (in the asymptotic variance)
        """
        ticks = self.events[self.obs_index][:i]
        inter_time = self.events[self.obs_index][i] - ticks
        return -self.beta_0 * np.sum(inter_time*np.exp(- self.beta_1 * inter_time))
    
    def Ater(self,i):
        """
        Compute Ater(t_i) (in the asymptotic variance)
        """
        ticks = self.events[self.obs_index][:i]
        inter_time = self.events[self.obs_index][i] - ticks
        return self.beta_0 * np.sum(np.power(inter_time,2)*np.exp(- self.beta_1 * inter_time))    
    
    def Gexp(self,calc_type=0):
        """
        Compute G(delta)
        """
        if (calc_type == 0):
            temp = 0
            for i in range(self.deltaindex):
                temp += np.log(self.mu_0 + self.Aexp(i) )
            return temp - self.mu_0*self.delta
        elif (calc_type == 1):
            temp = 0
            for i in range(self.deltaindex,self.nticks):
                temp += np.log(self.mu_1 + self.Aexp(i) )
            return temp - self.mu_1*(self.end_time - self.delta)
    
    def Gfischer(self,calc_type=0):
        """
        Compute Gf(delta)
        """
        if (calc_type ==0):
            temp = np.zeros((4,4))
            for i in range(self.deltaindex):
                a = self.Aexp(i)
                b = self.Abis(i)
                #c= self.Ater(i)
                lambada = self.mu_0 + a
                mat_deriv = np.zeros((4,1))
                mat_deriv[0,0] = 1
                mat_deriv[2,0] = a/self.mu_0
                mat_deriv[3,0] = b
                temp += (mat_deriv @ mat_deriv.transpose()) /  lambada**2
            return temp
        elif (calc_type == 1):
            temp = np.zeros((4,4))
            for i in range(self.deltaindex,self.nticks):
                a = self.Aexp(i)
                b = self.Abis(i)
                #c= self.Ater(i)
                lambada = self.mu_1 + a
                mat_deriv = np.zeros((4,1))
                mat_deriv[1,0] = 1
                mat_deriv[2,0] = a/self.mu_0
                mat_deriv[3,0] = b                
                temp += (mat_deriv @ mat_deriv.transpose()) /  lambada**2
            return temp
            
            
            
        
    
    
    def Grexp(self):
        """
        Compute Gr(delta)
        """
        temp = 0
        for i in range(self.deltaindex,self.nticks):
            temp += np.log(1 + (self.mu_1 - self.mu_0)/(self.mu_0 + self.Aexp(i)) )
        return temp - (self.mu_1 - self.mu_0)*(self.end_time - self.delta)
    
    def Grpow(self):
        """
        Compute Gr(delta)
        """
        temp = 0
        for i in range(self.deltaindex,self.nticks):
            temp += np.log(1 + (self.mu_1 - self.mu_0)/(self.mu_0 + self.Apow(i)) )
        return temp - (self.mu_1 - self.mu_0)*(self.end_time - self.delta)
        
    def Bexp(self):
        """
        Compute B
        """
        ticks = self.end_time - self.events[self.obs_index]
        temp = np.sum(np.exp(- self.beta_1 * ticks))
        return self.beta_0 / self.beta_1 * (self.nticks - temp)
    
    def Apow(self,i):
        """
        Compute A(t_i)
        """
        ticks = self.events[self.obs_index][:i]
        inter_time = self.events[self.obs_index][i] - ticks # compute t_i - t_j for j < i
        return self.beta_0 * np.sum(np.power(self.c + inter_time,- self.beta_1))

    def Gpow(self, calc_type=0):
        """
        Compute G(delta)
        """
        if (calc_type == 0):
            temp = 0.0
            for i in range(self.deltaindex):
                temp += np.log(self.mu_0 + self.Apow(i) )
            return temp - self.mu_0*self.delta
        elif (calc_type == 1):
            temp = 0.0
            for i in range(self.deltaindex,self.nticks):
                temp += np.log(self.mu_1 + self.Apow(i) )
            return temp - self.mu_1*(self.end_time - self.delta)
        
    def Bpow(self):
        """
        Compute B
        """
        ticks = self.end_time - self.events[self.obs_index]
        temp = self.c**(1- self.beta_1) - np.power(self.c + ticks , 1 - self.beta_1 )
        return self.beta_0/(self.beta_1 - 1)* np.sum(temp)
    

        
    
    
        
        
        
    
    
