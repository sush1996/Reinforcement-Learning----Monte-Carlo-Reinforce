import numpy as np

class DiscreteSoftmaxPolicy(object):
    def __init__(self, num_states, num_actions, temperature):
        self.num_states = num_states 
        self.num_actions = num_actions
        self.temperature = temperature
        # here are the weights for the policy - you may change this initialization       
        self.weights = np.random.rand(self.num_states, self.num_actions)
    
    def act(self, state):
        policy_softmax = np.zeros(self.num_actions)
        for action_num in range(self.num_actions):
            
            num = np.exp(self.weights[state, action_num]/float(self.temperature))
            den = sum([np.exp(self.weights[state, action_den]/float(self.temperature)) for action_den in range(self.num_actions)])

            policy_softmax[action_num] = num/den

        return np.random.choice(4,1, p = policy_softmax)

    def act_optimally(self, state):
        policy_softmax = np.zeros(self.num_actions)
        for action_num in range(self.num_actions):
            
            num = np.exp(self.weights[state, action_num]/float(self.temperature))
            den = sum([np.exp(self.weights[state, action_den]/float(self.temperature)) for action_den in range(self.num_actions)])

            policy_softmax[action_num] = num/den
            
        return np.argmax(policy_softmax)
    
    # computes the gradient of the discounted return    
    # at a specific state and action    
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, advantage):
        policy_softmax_grad = np.zeros((self.num_states, self.num_actions))
        
        num = np.exp(self.weights[state, action]/float(self.temperature))
        den = sum([np.exp(self.weights[state, action_den]/float(self.temperature)) for action_den in range(self.num_actions)])
        policy_softmax_grad[state, action] = (1/float(self.temperature))*(1 - (num/den))
        
        return advantage*policy_softmax_grad#*discounted_return

    # takes a step of gradient ascent given a gradient (from compute_gradient())    
    # and a step size. adjust self.weights
    def gradient_step(self, grad, w_step_size):
        self.weights = self.weights + w_step_size*grad
