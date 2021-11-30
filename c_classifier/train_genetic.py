from my_dataset import MyTfDataset
from genetic import Genetic
from cnn import CNN

import numpy as np
import tensorflow as tf

import time

tf.random.set_seed(12345)
np.random.seed(12345)

train = MyTfDataset()
train.read_data(datset_path='data/train/', augmentation=True)
val = train.validation_split()

print(f'Train {train}')
print(f'Validation {val}')

pop_size = 10
nlayers = 3
max_nfilters = 100
max_sfilters = 7
epochs = 20
num_generations = 15

gen_cnn = Genetic(pop_size, nlayers, max_nfilters, max_sfilters)
pop = gen_cnn.generate_population()

start = time.time()
for i in range(num_generations + 1):
    pop_acc, pop_acc_val = gen_cnn.fitness(pop, train.data, train.labels_oh, val.data, val.labels_oh, epochs)
    print(f'Best Accuracy(Val) at the generation {i}: {gen_cnn.max_acc_val}')
    parents = gen_cnn.select_parents(pop, 5, pop_acc_val.copy())
    child = gen_cnn.crossover(parents)
    child = gen_cnn.mutation(child, max_nfilters, max_sfilters)
    pop = np.concatenate((parents, child), axis=0).astype('int')
print(f'Genetic algorithm took {time.time() - start}[s]')
print(f'Best architecture {gen_cnn.best_arch} - Train: {gen_cnn.max_acc} Val {gen_cnn.max_acc_val}')
gen_cnn.smooth_curve(0.8, num_generations)

model = CNN(gen_cnn.best_arch[:3], gen_cnn.best_arch[3:])
model.compile()
model.train(train.data, train.labels_oh, val.data, val.labels_oh, batch_size=32, epochs=20, save=True, verbose=True)

pred = model.predict_per_class(val.data, val.labels, verbose=True)
