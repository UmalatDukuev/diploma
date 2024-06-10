import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# Define the objective function with multiple local minima and one global minimum
def objective_function(x):
    return np.sin(5 * x) * np.sin(3 * x) * np.sin(x)

# Particle class for PSO
class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1])
        self.velocity = np.random.uniform(-1, 1)
        self.best_position = self.position
        self.best_score = float('inf')

    def update_personal_best(self, objective_function):
        score = objective_function(self.position)
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position

# Particle Swarm Optimizer class
class ParticleSwarmOptimizer:
    def __init__(self, objective_function, pop_size, bounds, max_iter):
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.bounds = bounds
        self.max_iter = max_iter
        self.particles = [Particle(bounds) for _ in range(pop_size)]
        self.global_best_position = self.particles[0].position
        self.global_best_score = float('inf')

    def evaluate(self):
        for particle in self.particles:
            particle.update_personal_best(self.objective_function)
            if particle.best_score < self.global_best_score:
                self.global_best_score = particle.best_score
                self.global_best_position = particle.best_position

    def update_velocities_and_positions(self, inertia=0.5, cognitive=1.5, social=1.5):
        for particle in self.particles:
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_velocity = cognitive * r1 * (particle.best_position - particle.position)
            social_velocity = social * r2 * (self.global_best_position - particle.position)
            particle.velocity = inertia * particle.velocity + cognitive_velocity + social_velocity
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

    def optimize(self):
        plt.ion()
        for iter in range(self.max_iter):
            self.evaluate()
            self.update_velocities_and_positions()
            self.plot(iter)
        plt.ioff()
        plt.show()

    def plot(self, iter):
        plt.clf()
        X = np.linspace(self.bounds[0], self.bounds[1], 400)
        Y = self.objective_function(X)
        plt.plot(X, Y, label="Objective Function")
        for particle in self.particles:
            plt.scatter(particle.position, self.objective_function(particle.position), color='red')
        plt.scatter(self.global_best_position, self.objective_function(self.global_best_position),
                    color='blue', marker='*', s=100)
        plt.title(f'Iteration {iter}')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.pause(0.01)

# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(self, objective_function, pop_size, bounds, max_iter, mutation_rate=0.01, crossover_rate=0.7, elitism=True):
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.bounds = bounds
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.population = np.random.uniform(bounds[0], bounds[1], (pop_size, 1))
        self.best_individual = None
        self.best_score = float('inf')

    def evaluate(self):
        scores = np.array([self.objective_function(ind) for ind in self.population.flatten()])
        best_index = np.argmin(scores)
        if scores[best_index] < self.best_score:
            self.best_score = scores[best_index]
            self.best_individual = self.population[best_index]
        return scores

    def select_parents(self, scores):
        # Convert scores to probabilities, higher scores mean lower probability
        fitness = 1 / (1 + scores - np.min(scores))
        fitness /= np.sum(fitness)
        indices = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=fitness)
        return self.population[indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation = np.random.uniform(-0.1, 0.1)
            individual += mutation
        return np.clip(individual, self.bounds[0], self.bounds[1])

    def optimize(self):
        plt.ion()
        for iter in range(self.max_iter):
            scores = self.evaluate()
            parents = self.select_parents(scores)
            next_population = []
            if self.elitism:
                elite_index = np.argmin(scores)
                elite = self.population[elite_index]
                next_population.append(elite)
            for i in range(0, self.pop_size, 2):
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(child1))
                next_population.append(self.mutate(child2))
            if self.elitism:
                next_population = next_population[:self.pop_size]
            self.population = np.array(next_population)
            self.plot(iter)
        plt.ioff()
        plt.show()

    def plot(self, iter):
        plt.clf()
        X = np.linspace(self.bounds[0], self.bounds[1], 400)
        Y = self.objective_function(X)
        plt.plot(X, Y, label="Objective Function")
        for ind in self.population.flatten():
            plt.scatter(ind, self.objective_function(ind), color='red')
        plt.scatter(self.best_individual, self.objective_function(self.best_individual),
                    color='blue', marker='*', s=100)
        plt.title(f'Iteration {iter}')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.pause(0.01)

# Function to start optimization based on user-selected algorithm
def start_optimization(algorithm, pop_size, max_iter):
    bounds = [-4, 4]
    if algorithm == 'PSO':
        optimizer = ParticleSwarmOptimizer(objective_function, pop_size, bounds, max_iter)
    elif algorithm == 'GA':
        optimizer = GeneticAlgorithm(objective_function, pop_size, bounds, max_iter)
    optimizer.optimize()

# Function to create the user interface
def create_ui():
    root = tk.Tk()
    root.title("Optimization Algorithm Parameters")
    root.geometry("400x400")  # Set window size

    mainframe = ttk.Frame(root, padding="10 10 10 10")
    mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    algorithm_var = tk.StringVar()
    pop_size_var = tk.IntVar()
    max_iter_var = tk.IntVar()

    ttk.Label(mainframe, text="Select Algorithm:").grid(column=1, row=1, sticky=tk.W)

    pso_radio = ttk.Radiobutton(mainframe, text="PSO", variable=algorithm_var, value="PSO")
    pso_radio.grid(column=2, row=1, sticky=tk.W)

    ga_radio = ttk.Radiobutton(mainframe, text="GA", variable=algorithm_var, value="GA")
    ga_radio.grid(column=3, row=1, sticky=tk.W)

    ttk.Label(mainframe, text="Population Size:").grid(column=1, row=2, sticky=tk.W)
    pop_size_entry = ttk.Entry(mainframe, width=7, textvariable=pop_size_var)
    pop_size_entry.grid(column=2, row=2, sticky=(tk.W, tk.E))

    ttk.Label(mainframe, text="Number of Iterations:").grid(column=1, row=3, sticky=tk.W)
    max_iter_entry = ttk.Entry(mainframe, width=7, textvariable=max_iter_var)
    max_iter_entry.grid(column=2, row=3, sticky=(tk.W, tk.E))

    def on_start():
        algorithm = algorithm_var.get()
        pop_size = pop_size_var.get()
        max_iter = max_iter_var.get()
        root.destroy()
        start_optimization(algorithm, pop_size, max_iter)

    ttk.Button(mainframe, text="Start", command=on_start).grid(column=1, row=4, columnspan=3)

    for child in mainframe.winfo_children():
        child.grid_configure(padx=5, pady=5)

    root.mainloop()

# Create the user interface
create_ui()