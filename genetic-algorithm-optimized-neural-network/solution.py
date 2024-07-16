import numpy as np
import csv
import argparse
import random

class GeneticAlgorithm:
    def __init__(self, architecture, pop_size, mutation_rate, mutation_scale, elitism):
        self.architecture = architecture
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elitism = elitism
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for i in range(self.pop_size):
            nn = NeuralNetwork(self.architecture)
            population.append(nn)
        return population
    
    def evaluate_fitness(self, X, y):
        fitness_scores = []
        for nn in self.population:
            fitness = 1/nn.calculate_error(X, y) 
            fitness_scores.append(fitness)
        return fitness_scores

    def select_mating_pool(self, fitness_scores, num_mates):
        total_fitness = sum(fitness_scores)
        probabilities = [f / total_fitness for f in fitness_scores]
        selected = np.random.choice(self.population, size=num_mates, p=probabilities)
        return selected
    
    def apply_elitism(self, fitness_scores):
        elite_indices = np.argsort(fitness_scores)[-self.elitism:]
        elites = [self.population[i] for i in elite_indices]
        return elites
    
    def crossover(self, parent1, parent2):
        child_weights = [(w1 + w2) / 2 for w1, w2 in zip(parent1.weights, parent2.weights)]
        child_biases = [(b1 + b2) / 2 for b1, b2 in zip(parent1.biases, parent2.biases)]
        return NeuralNetwork(self.architecture, child_weights, child_biases)
        
    def mutate(self, nn):
        for i in range(len(nn.weights)):
            for j in range(nn.weights[i].shape[0]):
                for k in range(nn.weights[i].shape[1]):
                    if random.uniform(0, 1) < self.mutation_rate:
                        nn.weights[i][j][k] += np.random.normal(0, self.mutation_scale)
            for j in range(nn.biases[i].shape[1]):
                if random.uniform(0, 1) < self.mutation_rate:
                    nn.biases[i][0][j] += np.random.normal(0, self.mutation_scale)

    def evolve_population(self, X, y):
        fitness_scores = self.evaluate_fitness(X, y)
        new_population = self.apply_elitism(fitness_scores)
        
        while len(new_population) < self.pop_size:
            parent1, parent2 = self.select_mating_pool(fitness_scores, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.population = new_population

    def train(self, train_features, train_target, num_iterations, print_interval=2000):
        for iteration in range(num_iterations):
            self.evolve_population(train_features, train_target)
            if (iteration + 1) % print_interval == 0:
                fitness_scores = self.evaluate_fitness(train_features, train_target)
                best_fitness_index = np.argmax(fitness_scores)
                best_individual = self.population[best_fitness_index]
                train_error = best_individual.calculate_error(train_features, train_target)

                print(f"[Train error @{iteration + 1}]: {train_error:.6f}")


                #print(f"Best individual weights: {best_individual.weights}")
                #print(f"Best individual biases: {best_individual.biases}")

    def evaluate(self, test_features, test_target):
        fitness_scores = self.evaluate_fitness(test_features, test_target)
        best_fitness_index = np.argmax(fitness_scores)
        best_individual = self.population[best_fitness_index]
        test_error = best_individual.calculate_error(test_features, test_target)
        print(f"[Test error]: {test_error:.6f}")

class NeuralNetwork:
    def __init__(self, architecture, weights=None, biases=None):
        self.architecture = architecture
        if weights is None or biases is None:
            self.weights, self.biases = self.initialize_weights(architecture)
        else:
            self.weights = weights
            self.biases = biases
    
    def initialize_weights(self, architecture):
        weights = []
        biases = []
        for i in range(len(architecture) - 1):
            weight = np.random.normal(0, 0.01, (architecture[i], architecture[i+1]))
            bias = np.random.normal(0, 0.01, (1, architecture[i+1]))
            weights.append(weight)
            biases.append(bias)
        return weights, biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        activations = X
        for i in range(len(self.weights) - 1):
            biased_linear_combination = np.dot(activations, self.weights[i]) + self.biases[i]
            activations = self.sigmoid(biased_linear_combination)
        output = np.dot(activations, self.weights[-1]) + self.biases[-1]
        return output

    def calculate_error(self, X, y):
        predictions = self.forward(X)
        #print(f"Predictions: {predictions}")
        error =np.mean((y[:, np.newaxis] - predictions) ** 2)
        #print(f"Error: {error}")
        return error
    
    def __repr__(self):
        return f"NeuralNetwork(architecture={self.architecture}, weights={self.weights}, biases={self.biases})"

def load_data(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]
    features = [[float(value) for value in row[:-1]] for row in data]
    target = [float(row[-1]) for row in data]
    return np.array(features), np.array(target)

def main():
    parser = argparse.ArgumentParser(description='Genetic algorithm for neural network.')
    parser.add_argument("--train", type=str, help="Path to training data")
    parser.add_argument("--test", type=str, help="Path to testing data")
    parser.add_argument("--nn", type=str, help="Neural network architecture")
    parser.add_argument("--popsize", type=int, help="Population size for genetic algorithm")
    parser.add_argument("--elitism", type=int, help="Number of best individuals to preserve")
    parser.add_argument("--p", type=float, help="Mutation probability")
    parser.add_argument("--K", type=float, help="Standard deviation for mutation")
    parser.add_argument("--iter", type=int, help="Number of iterations for genetic algorithm")

    args = parser.parse_args()

    train_features, train_target = load_data(args.train)
    test_features, test_target = load_data(args.test)

    architecture_map = {
        '5s': [train_features.shape[1], 5, 1],
        '20s': [train_features.shape[1], 20, 1],
        '5s5s': [train_features.shape[1], 5, 5, 1]
    }
    architecture = architecture_map[args.nn]

    ga = GeneticAlgorithm(architecture, args.popsize, args.p, args.K, args.elitism)

    ga.train(train_features, train_target, args.iter)
    ga.evaluate(test_features, test_target)

if __name__ == "__main__":
    main()