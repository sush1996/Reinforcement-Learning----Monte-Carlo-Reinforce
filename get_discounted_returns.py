import math

# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]

def get_discounted_returns(rewards, gamma):
    disc_return = rewards
    
    for  num, reward in enumerate(rewards):
        disc_return[num] = sum([r2*math.pow(gamma, num2) for num2,r2 in enumerate(rewards[num:])])
    
    return disc_return
