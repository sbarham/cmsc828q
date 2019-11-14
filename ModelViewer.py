import tensorflow as tf
import gym
import numpy as np
import cv2
import time


loaded = tf.keras.models.load_model("./savedmodels/SpaceInvaders/7/besto/")

game_name = 'SpaceInvaders'

env = gym.make(game_name + '-v0')
num_actions = env.action_space.n

def preprocess(observation):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = observation[34:194]
    #observation = observation[::2,::2]
    
    observation = observation.astype('float32') / 255.0
    return observation.reshape(observation.shape[0],observation.shape[1],1)
    
def getActionFromModel(model, observation):
    return np.random.choice(num_actions,  p=model.predict(np.array([preprocess(observation)]))[0])

frame_skip = 7
def eval_fitness(model, graphic):
    observation = env.reset()
    total_reward = 0
    done = False
    action = 0
    game_step = 0;
    while not done:
        if graphic:
            env.render()
        if game_step % frame_skip == 0:
            action = getActionFromModel(model,observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        game_step += 1
        time.sleep(0.02)
    return total_reward

print(eval_fitness(loaded,True))

#for i in range(10):
#    print(eval_fitness(loaded,False))