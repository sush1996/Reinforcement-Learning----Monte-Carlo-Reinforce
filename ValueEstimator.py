import numpy as np

#Estimates value

class ValueEstimator(object):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions 
        #initial value estimates or weights of the value estimator are set to zero. 
        self.values = np.zeros((self.num_states))

    #takes in a state and predicts a value for the state
    def predict(self,state):
        value_estimate = self.values[state]
        return value_estimate

    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self, state, advantage, value_step_size):
        value_grad = 1
        self.values[state] = self.values[state] + value_step_size*advantage*value_grad
