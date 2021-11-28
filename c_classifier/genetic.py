import matplotlib.pyplot as plt
import numpy as np

from cnn import CNN


class Genetic:

    def __init__(self, pop_size, nlayers, max_nfilters, max_sfilters):
        self.pop_size = pop_size

        self.nlayers = nlayers
        self.max_nfilters = max_nfilters
        self.max_sfilters = max_sfilters

        self.best_arch = np.zeros((1, 6))

        self.max_acc = 0
        self.max_acc_val = 0
        self.gen_acc = []
        self.gen_acc_val = []

    def generate_population(self):
        pop_nlayers = np.random.randint(1, self.max_nfilters, (self.pop_size, self.nlayers))
        pop_sfilters = (np.random.randint(1, self.max_sfilters, (self.pop_size, self.nlayers)) * 2) + 1
        pop_total = np.concatenate((pop_nlayers, pop_sfilters), axis=1)
        return pop_total

    @staticmethod
    def select_parents(pop, nparents, fitness):
        parents = np.zeros((nparents, pop.shape[1]))
        for i in range(nparents):
            best = np.argmax(fitness)
            parents[i] = pop[best]
            fitness[best] = -99999
        return parents

    def crossover(self, parents):
        nchild = self.pop_size - parents.shape[0]
        nparents = parents.shape[0]
        child = np.zeros((nchild, parents.shape[1]))
        for i in range(nchild):
            first = i % nparents
            second = (i + 1) % nparents
            child[i, :2] = parents[first][:2]
            child[i, 2] = parents[second][2]
            child[i, 3:5] = parents[first][3:5]
            child[i, 5] = parents[second][5]
        return child

    @staticmethod
    def mutation(child, max_nfilters, max_sfilters):
        for i in range(child.shape[0]):
            val = np.random.randint(1, 6)
            ind = np.random.randint(0, 3)
            if child[i][ind] + val > max_nfilters:
                child[i][ind] -= val
            else:
                child[i][ind] += val
            val = (np.random.randint(0, 3) * 2) + 1
            ind = np.random.randint(3, 6)
            if child[i][ind] + val > (max_sfilters * 2):
                child[i][ind] -= val
            else:
                child[i][ind] += val
        return child

    def fitness(self, pop, X, Y, X_val, Y_val, epochs):
        pop_acc = []
        pop_acc_val = []
        for i in range(pop.shape[0]):
            nfilters = pop[i][0:3]
            sfilters = pop[i][3:]

            model = CNN(nfilters, sfilters)
            model.compile()
            H = model.train(X, Y, X_val, Y_val, batch_size=200, epochs=epochs)

            acc = H.history['accuracy']
            acc_val = H.history['val_accuracy']
            pop_acc.append(max(acc) * 100)
            pop_acc_val.append(max(acc_val) * 100)

        if max(pop_acc_val) > self.max_acc_val:
            self.max_acc = max(pop_acc)
            self.max_acc_val = max(pop_acc_val)
            self.best_arch = pop[np.argmax(pop_acc_val)]

        self.gen_acc.append(max(pop_acc))
        self.gen_acc_val.append(max(pop_acc_val))
        return pop_acc, pop_acc_val

    def smooth_curve(self, factor, gen):
        smoothed_points = []
        smoothed_points_val = []
        for point in self.gen_acc:
            if smoothed_points:
                prev = smoothed_points[-1]
                smoothed_points.append(prev * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        for point in self.gen_acc_val:
            if smoothed_points_val:
                prev = smoothed_points_val[-1]
                smoothed_points_val.append(prev * factor + point * (1 - factor))
            else:
                smoothed_points_val.append(point)
        plt.plot(range(gen + 1), smoothed_points, 'b', label='Smoothed training acc')
        plt.plot(range(gen + 1), smoothed_points_val, 'g', label='Smoothed training acc_val')
        plt.xticks(np.arange(gen + 1))
        plt.legend(loc='lower right')
        plt.title('Fitness Accuracy vs Generations')
        plt.xlabel('Generations')
        plt.ylabel('Fitness (%)')
        plt.show()
