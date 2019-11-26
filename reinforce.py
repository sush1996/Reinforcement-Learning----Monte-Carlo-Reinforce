import numpy as np
from get_discounted_returns import get_discounted_returns
import matplotlib.pyplot as plt
#performs monte carlo reinforce with baseline

def reinforce(env, policy, value_estimator, num_episodes, w_step_size, value_step_size, gamma):
    count = 0
    sum_o_returns = [0]*num_episodes

    for i in range(num_episodes):
        states = []
        actions = []
        rewards = []

        grad = np.zeros((env.get_num_states(), env.get_num_actions()))
        state = env.reset()    
        states = states + [state]
        actions = actions + [policy.act(states[0])]

        done = False
        j = 0
        
        while not done:        
            states_t, rewards_t, done = env.step(actions[j])
            states = states + [states_t]
            rewards = rewards + [rewards_t]
            actions_t = policy.act(states_t)
            actions = actions + [actions_t]
            j = j+1
            
           
        print("Number of iterations in episode {}:". format(i), j, "sum of rewards:", sum(rewards))
        sum_o_returns[i] = sum(rewards)

        discounted_return = get_discounted_returns(rewards, 1)

        #Update weights only when a positive reward is observed (a handy trick in Monte Carlo Approaches)
        if rewards_t>0:
            for k in range(len(rewards)):
                value_estimate = value_estimator.predict(states[k])
                advantage = discounted_return[k] - value_estimate

                grad = policy.compute_gradient(states[k], actions[k], advantage)
                policy.gradient_step(grad, w_step_size)
                
                value_estimator.update(states[k], advantage, value_step_size)
                
        #Q1.2.3
        state = env.reset()
        done = False
        
        while not done:
            action = policy.act(state)
            state, reward, done = env.step(action)
        
        if reward>0 and done==True:
            count = count+1
    
    print("Number of times it reaches to goal:", count)

    plt.plot(range(num_episodes), sum_o_returns)
    plt.xlabel('Number of episodes')
    plt.ylabel('Sum of Returns')
    plt.title('Sum of Returns vs Number of Episodes without baseline')
    plt.show()

    return policy
