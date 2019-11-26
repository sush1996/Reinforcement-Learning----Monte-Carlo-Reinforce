from grid_world import *
from DiscreteSoftmaxPolicy import DiscreteSoftmaxPolicy
from ValueEstimator import ValueEstimator
from get_discounted_returns import get_discounted_returns
from reinforce import reinforce
import numpy as np
import math

def main():

    env = GridWorld(MAP2)
    env.print()
    num_episodes = 2000
    w_step_size = 2**(-3)
    value_step_size = 2**(-2)
    temperature = 1
    gamma = 0.9
    
    policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions(), temperature)
    value_estimator = ValueEstimator(env.get_num_states(), env.get_num_actions())
    new_policy = reinforce(env, policy, value_estimator, num_episodes, w_step_size, value_step_size, gamma)

    size = int(math.sqrt(new_policy.num_states))
    
    table_o_policies = np.zeros((new_policy.num_states, 1))
    table_o_policies = np.argmax(new_policy.weights, axis = 1)
    table_o_policies = table_o_policies.reshape((size, size))
    
    print(table_o_policies)

    state = env.reset()
    env.print()
    done = False
    while not done:
        input("press enter:")
        action = new_policy.act_optimally(state)
        state, reward, done = env.step(action)
        env.print()

if __name__ == "__main__":
    main()