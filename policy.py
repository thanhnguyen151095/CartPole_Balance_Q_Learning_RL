import gym
import numpy as np
import matplotlib.pyplot as plt
import random

env = gym.make("CartPole-v1")

gamma = 0.99
alpha = 0.0001
action = 0
w = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
delta_w = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

def log(log_message):
    """
    DESCRIPTION:
    - Adds a log message "log_message" to a log file.
    """
    # open the log file and make sure that it's closed properly at the end of the 
    # block, even if an exception occurs:
    with open("/home/win/Desktop/Cart_balance/log.txt", "a") as log_file:
        # write the log message to logfile:
        log_file.write(log_message)
        log_file.write("\n") # (so the next message is put on a new line)

def normalize_angle(angle):
    """
    3*pi gives -pi, 4*pi gives 0 etc, etc. (returns the negative difference
    from the closest multiple of 2*pi)
    """
    normalized_angle = abs(angle)
    normalized_angle = normalized_angle % (2*np.pi)
    if normalized_angle > np.pi:
        normalized_angle = normalized_angle - 2*np.pi
    normalized_angle = abs(normalized_angle)
    return normalized_angle

def state_action_to_features(state, action):
    x = state.item(0)
    x_dot = state.item(1)
    theta = state.item(2)
    theta_dot = state.item(3)
    X = np.array([x**2,x,
                 theta**2,theta,
                 x_dot**2,x_dot,
                 theta_dot**2,theta_dot,
                 0*action*x,
                 0*action*x_dot,
                 action*theta,
                 action*theta_dot])

    return X

def q_hat(state, action, w):
    X = state_action_to_features(state, action)
    output = np.dot(X,w)
    return output
    

def get_action(state, w):
    actions = [0, 1]
    qs = []
    for action in actions:
        qs.append(q_hat(state, action, w))
    #print(qs)
    max_index = np.argmax(qs)
    action = actions[max_index]

    return action

# epislon-greedy:    
def eps_greedy_action(epsilon,state, w):
    actions = [0, 1]
    if np.random.rand()< 1-epsilon:
        action = get_action(state, w)
    else:
        action = random.choice(actions)

    return action

timesteps = []
for i_episode in range(200):
    state = env.reset()
    
    #action = get_action(state, w)
    for t in range(100000):
        #env.render()
        #w = np.array([ 3.56066643, -0.06260068,  0.3297794,  -0.00658669,  3.76363316, -0.05467598, 0.99698785, -0.04918008,  0.05498753,  0.46581163])
        action = eps_greedy_action(0.01,state, w)

        #print(action)
        observation, reward, done, info = env.step(action)
        # update w
        delta_w = (alpha*(reward + gamma*q_hat(observation, get_action(observation, w), w) - q_hat(state, action, w)))*state_action_to_features(state, action)
        w = np.add(w,delta_w)
        state = observation
        

        if done:
            print("Episode " + str(i_episode) + " finished after " + str(t+1) + " timesteps")
            timesteps += [t+1]
            #print(w)
            break

plt.plot(timesteps)
log(str(timesteps))
log(str(w))
plt.show()
