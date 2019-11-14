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
import random



#######################
## Library Functions ##
#######################

# Image size is 210 by 160

def preprocess(observation):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = observation[34:194]
    
    observation = observation.astype('float32') / 255.0
    return observation.reshape(observation.shape[0],observation.shape[1],1)

def generateModel(genome):
    conv_dim = len(genome)
    
    # autoencoder model
    input_img = tf.keras.layers.Input(shape=(img_dim1, img_dim2, 1)) 

    x = input_img

    layer_sizes = []
    layer_sizes.append((x.shape[1],x.shape[2]))
    for i in range(conv_dim):
        x = tf.keras.layers.Conv2D(genome[i], (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        layer_sizes.append((x.shape[1],x.shape[2]))

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
    
def getActionFromModel(model, observation):
    return np.random.choice(
        num_actions,
        p=model.predict(np.array([preprocess(observation)]))[0]
    )

def mutate(genome, sigma):
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
    
# painless crossover implementation, just stack the nets on top of each other, let deleterous mutations get rid of redundancies
# NOTE: unfinished right now
def crossover(genome1, genome2):
    new_conv_shape = []
    p1_shape = genome1.shape
    p2_shape = genome2.shape
    
    # stack layer shapes
    for i in range(max(len(p1_shape),len(p2_shape))):
        dim1 = 0
        dim2 = 0
        if i < len(p1_shape):
            dim1 = p1_shape[i]
        if i < len(p2_shape):
            dim2 = p2_shape[i]
        new_conv_shape.append(max(dim1,dim2))

    new_model = generateModel(new_conv_shape)
    
    new_weights = [np.zeros(w.shape) for w in new_model.get_weights()]
    
    
    clear_session()
    
    new_genome = Genome(new_conv_shape, new_weights)
    return new_genome

def eval_fitness(model, graphic):
    observation = env.reset()
    total_reward = 0
    done = False
    action = 0
    #for t in range(game_iters):
    game_step = 0
    while not done:
        if graphic:
            env.render()
        if game_step % frame_skip == 0:
            action = getActionFromModel(model,observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        #if done:
        #    if graphic:
        #        print("Episode finished after {} timesteps".format(t+1))
        #    break
        game_step += 1
    return total_reward


#####################
## Training Script ##
#####################

game_name = 'SpaceInvaders'

env = gym.make(game_name + '-v0')

observation = preprocess(env.reset())
num_actions = env.action_space.n

img_dim1 = observation.shape[0]
img_dim2 = observation.shape[1]

prob_complex = 0.7
max_filters = 32
min_length = 2
max_length = 5
last_layer_max = 16

init_conv_shape = [16, 8, 8, 8]
model = generateModel(init_conv_shape)

# Begin ES/PSO here
population_size = 100
population = []

prob_change_dim = 0.5
prob_delete = 0.7

# Add default 
weights = [glorot_uniform()(w.shape) for w in model.get_weights()]
population.append(Genome(init_conv_shape, weights))

clear_session()

for i in range(population_size - 1):
    new_conv_shape = generateRandomShape()
    print(new_conv_shape)
    model = generateModel(new_conv_shape)
    weights = [glorot_uniform()(w.shape) for w in model.get_weights()]
    population.append(Genome(new_conv_shape, weights))
    clear_session()

#game_iters = 10

# skip some frames to accelerate training time
frame_skip = 7

#eval_fitness(model, True)

# Genetic Programming/Evolution Strategies code
selection = 0.2
num_parents = int(len(population) * selection)
sigma = 0.05
# 1 slot for elitism
num_children = len(population) - num_parents - 1
iterations = 100

best_overall = population[0]
best_fitness = 0.0
best_member = population[0]
game_evals = 5
eval_thresh = 0.5

fitnesses = np.zeros((iterations, len(population)))
for iter in range(iterations):

    start = time.time()
    fitness = np.array([0.0 for i in population])
    print("Evaluating fitness: ", end = '', flush=True)
    
    # this is where the evaluation needs to be distributed over processor cores
    for i in range(population_size):
        model = generateFromGenome(population[i])
        
        evals = 0
        while evals < game_evals:
            curr_fitness = eval_fitness(model, False)
            
            fitness[i] += curr_fitness
            evals += 1
            # exploitive eval, we don't want to evaluate agents we know are bad multiple times
            if curr_fitness < best_fitness * eval_thresh:
                # just give the agent the same bad score for the rest
                # we don't want to promote agents that just happen to be lucky in the first run
                fitness[i] += curr_fitness * (game_evals - evals)
                evals = game_evals
                break;
            #endif
        #endwhile
        
        fitness[i] /= evals
        
        if(i % int(len(population)/10) == 0):
            print(str(i) + "...", end = '', flush=True)
        #endif
    #endfor (multiprocessing)
    
    print()
    
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
    
    random_select = np.random.randint(num_parents, size=(num_children))
    children = [mutate(parents[i], sigma) for i in random_select]
    
    population[:num_parents] = parents
    population[num_parents:num_parents + num_children] = children
    population[-1] = best_overall
    end = time.time()
    print("Iteration", str(iter), "finished in", end - start, "seconds")

model = generateFromGenome(best_overall)
eval_fitness(model, True)

run_number = "8"

tf.saved_model.save(model, "./savedmodels/" + game_name + "/" + run_number + "/besto/")
clear_session()

model = generateFromGenome(best_member)
tf.saved_model.save(model, "./savedmodels/" + game_name + "/" + run_number + "/bestf/")

fig = plt.figure()
x = range(iterations)
for xe, ye in zip(x, fitnesses):
    plt.scatter([xe] * len(ye), ye, color='gray')
    
plt.plot(x, np.mean(fitnesses, axis=1), color='red')
plt.savefig('./results/run' + run_number + "results")


env.close()
