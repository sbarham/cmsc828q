from __future__ import absolute_import, division, print_function, unicode_literals


# Install TensorFlow
import numpy as np
import tensorflow as tf
import gym
import cv2
import matplotlib.pyplot as plt
import time
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.backend import clear_session
import os
import random
import multiprocessing
import tqdm
import argparse
from datetime import datetime


#######################
##     CONSTANTS     ##
#######################

### Rendering/Graphics-related constants
# whether or not to render the environment after training
GRAPHIC = False

### Sampling-related constants
# set to None in order to use the model's softmax output
# set to < 0.3 in order to sharpen probability peaks (0.2 is a good default)
# set to < 0.1 in order to make nearly deterministic
temperature = None

### Network initialization constants
# maximum initial number of filters per layer
max_filters = 32
# minimum initial length of the convolutional part of the network
min_length = 2
# maximum initial length of the convolutional part of the network
max_length = 5
# maximum initial length of the last convolutional layer
last_layer_max = 16

### Mutation/Reproduction Constants
# probability a mutation changes the size of the network
prob_change_dim = 0.5
# probability a deletion occurs 
prob_delete = 0.7
# probability that a deletion/insertion of filter occurs in a higher layer
prob_complex = 0.7
# standard deviation of gaussian noise for weight mutation
sigma = 0.05
# percentage of children created through crossover
crossover_percent = 0.0
# percentage of multated children will be 1 - selection - crossover_percent

### Training and evaluation constants
# skip some frames to accelerate training time
frame_skip = 7
# size of training population
population_size = 200
# percentage of population chosen for reproduction/mutation
selection = 0.2
# how many game evaluations to determine fitness
game_evals = 5
# how many evolution iterations to run
iterations = 100

# number of CPU's to parallize over, might need to hardcode this for the server
num_cpu = int(multiprocessing.cpu_count() - 1)

if num_cpu < 2:
  num_cpu = 2

# directory number used to save run results (see end of code for reference)
run_number = datetime.now().strftime("%Y-%m-%d-%H-%M")

curr_fitness = np.zeros(population_size) 


###############
## Technical ##
###############

# code for sharing a single GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

#######################
## Library Functions ##
#######################

# Setting up the openAI gym environment and the preprocess function should occur
# before the generateModel function, since the number of actions could be different
# in different games
game_name = 'SpaceInvaders'

env = gym.make(game_name + '-v0')

def preprocess(observation):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = observation[34:194]
    # uncomment this line to try 2x downsampling
    #observation = observation[::2,::2]
    
    observation = observation.astype('float32') / 255.0
    return observation.reshape(observation.shape[0],observation.shape[1],1)

observation = preprocess(env.reset())
num_actions = env.action_space.n

img_dim1 = observation.shape[0]
img_dim2 = observation.shape[1]

# Image size is 160 by 160 after preprocessing

def generateModel(genome):
    conv_dim = len(genome)
    
    # autoencoder model
    input_img = tf.keras.layers.Input(shape=(img_dim1, img_dim2, 1)) 

    x = input_img

    for i in range(conv_dim):
        x = tf.keras.layers.Conv2D(genome[i], (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(num_actions, activation='softmax')(x)

    ret = tf.keras.models.Model(input_img, x)
    return ret

class Genome:
    def __init__(self, shape, weights):
        self.shape = shape
        self.weights = weights

def generateRandomShape():
    new_conv_shape = []
    new_conv_shape.append(random.randint(1,max_filters))
    curr_length = 1
    while curr_length < min_length:
        new_conv_shape.append(random.randint(1,max_filters))
        curr_length += 1
    while curr_length < max_length and random.random() < prob_complex:
        new_conv_shape.append(random.randint(1,max_filters))
        curr_length += 1
        
    # hardcoded 8 filter layer append for computational reasons
    new_conv_shape.append(random.randint(1,last_layer_max))
    return new_conv_shape
    
def generateFromGenome(genome):
    clear_session()
    model = generateModel(genome.shape)
    model.set_weights(genome.weights)
    return model
    
def getActionFromModel(model, observation, temperature=None):
    """
    temperature = 0.2 is a nice default, giving reasonably sharp
    peaks to the distribution, without totally throwing away
    stochasticity
    """
    p = model.predict(np.array([preprocess(observation)]))[0]
    if temperature is not None:
        tmp_scaled_p = np.exp(p / temperature)
        p = tmp_scaled_p / np.sum(tmp_scaled_p)
    
    return np.random.choice(num_actions, p=p)


def sample(softmax, temperature):
    EPSILON = 10e-16 # to avoid taking the log of zero
    
    (np.array(softmax) + EPSILON).astype('float64')
    preds = np.log(softmax) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    
    return probas[0]
        

def mutate(genome, sigma):
    # weight mutation
    WeightGen = RandomNormal(stddev=sigma)
    new_weights = [np.clip(w + WeightGen(w.shape), 0, 1) for w in genome.weights]
    new_conv_shape = [dim for dim in genome.shape]
    old_layers = len(new_conv_shape)
    
    if random.random() < prob_change_dim:
        layer_index = 0
        # Choose a layer to mutate
        while layer_index < len(new_conv_shape) - 1 and random.random() < prob_complex:
            layer_index += 1
        
        # Choose whether to add or delete a filter
        if random.random() < prob_delete:
            if old_layers == 1:
                if new_conv_shape[layer_index] <= 1:
                    new_conv_shape[layer_index] += 1
            else:
                new_conv_shape[layer_index] -= 1
                if (new_conv_shape[layer_index] == 0):
                    del new_conv_shape[layer_index]
                else:
                    # filter collapse
                    weight_index = layer_index * 2
                    # conv-layers have dimension (3, 3, # input maps, # filters), we want to change axis 3
                    axis = 3
                    num_filters = new_weights[weight_index].shape[axis]
                    filter_to_delete = random.randint(0,num_filters-1)
                    
                    # delete from filter weights
                    np.delete(new_weights[weight_index], filter_to_delete, axis)
                    
                    # delete from bias weights
                    np.delete(new_weights[weight_index + 1], filter_to_delete, 0)
                    
                    if (layer_index < len(new_conv_shape) - 1):
                        # delete from input in next layer
                        np.delete(new_weights[weight_index + 2], filter_to_delete, axis - 1)
                    else:
                        # delete from dense layer
                        dim_dense = new_weights[weight_index + 2].shape[0]
                        # calculate how many connections we need to remove
                        partition_size = int(dim_dense / num_filters)
                        start_delete = partition_size * filter_to_delete
                        end_delete = start_delete + partition_size
                        np.delete(new_weights[weight_index + 2], np.s_[start_delete:end_delete], 0)
        else:
            if layer_index == len(new_conv_shape) - 1 and random.random() < prob_complex:
                # may need to do something different here, layer additions are probably always negative for fitness
                new_conv_shape.append(random.randint(1,last_layer_max))
            else:
                new_conv_shape[layer_index] += 1
                
        new_model = generateModel(new_conv_shape)
        temp_weights = new_model.get_weights()
        
        
        # Try to do weight inheritance for conv layers
        for i in range(min(len(new_conv_shape), old_layers)):
            old_shape = new_weights[i].shape
            new_shape = temp_weights[i].shape
            if (len(old_shape) == len(new_shape)):
                # Create a temporary buffer to hold weight values
                max_size = tuple(np.maximum(new_shape, old_shape))
                new_weight_arr = np.zeros(max_size)
                
                # Copy the new weights first
                new_weight_arr[tuple(map(slice, new_shape))] = temp_weights[i]
                
                # Override the new weights with old weights if possible
                new_weight_arr[tuple(map(slice, old_shape))] = new_weights[i]
                
                # Get the slice containing the new weights
                temp_weights[i] = new_weight_arr[tuple(map(slice, new_shape))]
                
        # Try to do weight inheritance for dense layers (last 4 weight matrices)
        for i in range(4):
            index1 = old_layers * 2 + i
            index2 = len(new_conv_shape) * 2 + i
            
            old_shape = new_weights[index1].shape
            new_shape = temp_weights[index2].shape
            if (len(old_shape) == len(new_shape)):
                max_size = tuple(np.maximum(new_shape, old_shape))
                new_weight_arr = np.zeros(max_size)
                new_weight_arr[tuple(map(slice, new_shape))] = temp_weights[index2]
                new_weight_arr[tuple(map(slice, old_shape))] = new_weights[index1]
                temp_weights[index2] = new_weight_arr[tuple(map(slice, new_shape))]
                
        new_weights = temp_weights
        
        
        clear_session()
    
    new_genome = Genome(new_conv_shape, new_weights)
    return new_genome
    
# filter swapping crossover
def crossover(genome1, genome2):
    p1_shape = len(genome1.shape)
    p2_shape = len(genome2.shape)
    new_conv_shape = genome1.shape
    
    # make a new copy of the weights in the first genome
    new_weights = [np.copy(w) for w in genome1.weights]
    for i in range(p1_shape):
        if i < p2_shape:
            # number of filters to sample from
            num_filters1 = genome1.shape[i]
            num_filters2 = genome2.shape[i]
            
            weightIndex1 = i * 2
            weightIndex2 = i * 2
            
            # we're going to swap half of the amount of filters the smaller conv-net has
            filters_to_swap = int(min(num_filters1, num_filters2)/2)
            
            # sample the indices for swapping
            filters1 = random.sample(range(num_filters1), filters_to_swap)
            filters2 = random.sample(range(num_filters2), filters_to_swap)
            
            for j in range(filters_to_swap):
                filter_size = min(new_weights[weightIndex1].shape[2],genome2.weights[weightIndex2].shape[2])
                # swap filters
                new_weights[weightIndex1][:,:,:filter_size,filters1[j]] = genome2.weights[weightIndex2][:,:,:filter_size,filters2[j]] 
                # swap bias weights as well
                new_weights[weightIndex1 + 1][filters1[j]] = genome2.weights[weightIndex2 + 1][filters2[j]] 
    
    new_genome = Genome(new_conv_shape, new_weights)
    return new_genome

def eval_fitness(model, graphic):
    observation = env.reset()
    total_reward = 0
    done = False
    action = 0
    game_step = 0
    while not done:
        if graphic:
            env.render()
        if game_step % frame_skip == 0:
            action = getActionFromModel(model,observation,temperature)

        observation, reward, done, info = env.step(action)
        
        total_reward += reward
        game_step += 1
        
    return total_reward
    
def evalAgent(agent):    
    model = generateFromGenome(agent)
    ret = 0
        
    evals = 0
    while evals < game_evals:
        curr_fitness = eval_fitness(model, False)
        
        ret += curr_fitness
        evals += 1
        
    ret /= evals

    # print("\t{}".format(ret))
    
    return ret

#####################
## Training Script ##
#####################

# Genetic Programming/Evolution Strategies code
if __name__ == '__main__':
    # initialize our argument parser
    parser = argparse.ArgumentParser(description="Evolve a network for playing Space Invaders.")
    parser.add_argument("-N", "--run_number", help="the distinguishing number to use when saving the training run's results")

    # parse the command-line arguments
    args = parser.parse_args()

    # assign the command-line arguments to the proper global variables
    run_number = run_number if args.run_number is None else args.run_number
    
    # make sure Linux is not forking
    multiprocessing.set_start_method("spawn")
  
    init_conv_shape = [16, 8, 8, 8]
    model = generateModel(init_conv_shape)

    # Begin ES/PSO here
    population = []

    # Add default 
    weights = [glorot_uniform()(w.shape) for w in model.get_weights()]
    population.append(Genome(init_conv_shape, weights))
    clear_session()

    print("Randomizing convolution filter shapes for initial generation ...")
    for i in tqdm.tqdm(range(population_size - 1)):
        new_conv_shape = generateRandomShape()
        # print(new_conv_shape)
        model = generateModel(new_conv_shape)
        weights = [glorot_uniform()(w.shape) for w in model.get_weights()]
        population.append(Genome(new_conv_shape, weights))
        clear_session()
    
    num_parents = int(len(population) * selection)
    
    # 1 slot for elitism
    num_children = len(population) - num_parents - 1
    num_crossover = int(num_children * crossover_percent)

    best_overall = population[0]
    best_fitness = 0.0
    best_member = population[0]
    
if __name__ == '__main__':
    print("Creating a new process pool with {} processes ...".format(num_cpu))
    pool = multiprocessing.Pool(num_cpu)
    fitnesses = np.zeros((iterations, len(population)))
    
    for iter in range(iterations):
        start = time.time()
        fitness = np.array([0.0 for i in population])
        print("(iter {}): Evaluating fitness ... ".format(iter))

        ##############################
        # FITNEESS EVALUATION CODE   #
        
        # multiprocessed version:
        fitness = np.fromiter(
          tqdm.tqdm(
            pool.imap(
              evalAgent,
              population,
              chunksize=1
            ),
            total= len(population)
          ),
          float
        )
        # fitness = np.fromiter(pool.map(evalAgent, population, chunksize=1))
        # fitness = np.array(pool.map(evalAgent, population))
        
        # non-multiprocessed version:
        # fitness = np.fromiter(map(evalAgent, tqdm.tqdm(population)), float)
        
        # END FITNESS EVALUATION CODE #
        ###############################
        
        fitnesses[iter] = fitness
        max_fitness = np.max(fitness)
        print("Mean Fitness:", np.mean(fitness), ", Max Fitness:", max_fitness)
        
        best_member = population[np.argmax(fitness)]
        print("Best Shape: ", best_member.shape)
        if(best_fitness < max_fitness):
            best_fitness = max_fitness
            best_overall = best_member
            
        
        print("Evolving Population....")
        best = np.argpartition(fitness, -num_parents)[-num_parents:]
        parents = [population[i] for i in best]
        
        children = [None for i in range(num_children)]
        # just mutation
        random_select = np.random.randint(num_parents, size=(num_children - num_crossover))
        children[num_crossover:] = [mutate(parents[i], sigma) for i in random_select]
        # children produced from crossover
        random_crossover_select = [random.sample(range(num_parents),2) for i in range(num_crossover)]
        children[:num_crossover] = [mutate(crossover(parents[i],parents[j]), sigma) for (i,j) in random_crossover_select]
        
        population[:num_parents] = parents
        population[num_parents:num_parents + num_children] = children
        population[-1] = best_overall
        end = time.time()
        print("Iteration", str(iter), "finished in", end - start, "seconds")

    model = generateFromGenome(best_overall)
    eval_fitness(model, GRAPHIC)
    
    tf.saved_model.save(model, "./savedmodels/" + game_name + "/" + str(run_number) + "/besto/")
    clear_session()

    model = generateFromGenome(best_member)
    tf.saved_model.save(model, "./savedmodels/" + game_name + "/" + str(run_number) + "/bestf/")

    fig = plt.figure()
    x = range(iterations)
    for xe, ye in zip(x, fitnesses):
        plt.scatter([xe] * len(ye), ye, color='gray')

    # plot
    plt.plot(x, np.mean(fitnesses, axis=1), color='red')
        
    # save everything to file
    name_prefix = './results/' + str(run_number) + "_results"
    np.save(name_prefix, fitnesses)
    plt.savefig(name_prefix + ".png")

    # close the environment
    env.close()
