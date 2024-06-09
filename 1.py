import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk

# Define the objective function with multiple local minima and one global minimum
def objective_function(x, y):
    return (x ** 2 + y ** 2) + 10 * np.sin(x) * np.cos(y)

# Particle class for PSO
class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], 2)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def update_personal_best(self, objective_function):
        score = objective_function(self.position[0], self.position[1])
        if score < self.best_score:
            self.best_score = score
            self.best_position = np.copy(self.position)

# Particle Swarm Optimizer class
class ParticleSwarmOptimizer:
    def __init__(self, objective_function, pop_size, bounds, max_iter):
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.bounds = bounds
        self.max_iter = max_iter
        self.particles = [Particle(bounds) for _ in range(pop_size)]
        self.global_best_position = np.copy(self.particles[0].position)
        self.global_best_score = float('inf')

    def evaluate(self):
        for particle in self.particles:
            particle.update_personal_best(self.objective_function)
            if particle.best_score < self.global_best_score:
                self.global_best_score = particle.best_score
                self.global_best_position = np.copy(particle.best_position)

    def update_velocities_and_positions(self, inertia=0.5, cognitive=1.5, social=1.5):
        for particle in self.particles:
            r1, r2 = np.random.rand(2), np.random.rand(2)
            cognitive_velocity = cognitive * r1 * (particle.best_position - particle.position)
            social_velocity = social * r2 * (self.global_best_position - particle.position)
            particle.velocity = inertia * particle.velocity + cognitive_velocity + social_velocity
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

    def optimize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.get_current_fig_manager().window.state('zoomed')
        plt.ion()
        for iter in range(self.max_iter):
            self.evaluate()
            self.update_velocities_and_positions()
            self.plot(ax, iter)
        plt.ioff()
        plt.show()

    def plot(self, ax, iter):
        ax.cla()
        X, Y = np.meshgrid(np.linspace(self.bounds[0], self.bounds[1], 100),
                           np.linspace(self.bounds[0], self.bounds[1], 100))
        Z = self.objective_function(X, Y)
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        for particle in self.particles:
            ax.scatter(particle.position[0], particle.position[1],
                       self.objective_function(particle.position[0], particle.position[1]), color='red')
        ax.scatter(self.global_best_position[0], self.global_best_position[1],
                   self.objective_function(self.global_best_position[0], self.global_best_position[1]),
                   color='blue', marker='*', s=100)
        ax.set_title(f'Iteration {iter}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.pause(1)

# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(self, objective_function, pop_size, bounds, max_iter, mutation_rate=0.01):
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.bounds = bounds
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.population = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
        self.best_individual = None
        self.best_score = float('inf')

    def evaluate(self):
        scores = np.array([self.objective_function(ind[0], ind[1]) for ind in self.population])
        best_index = np.argmin(scores)
        if scores[best_index] < self.best_score:
            self.best_score = scores[best_index]
            self.best_individual = self.population[best_index]
        return scores

    def select_parents(self, scores):
        prob = 1 - scores / np.sum(scores)
        prob /= np.sum(prob)
        indices = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=prob)
        return self.population[indices]

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(2)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation = np.random.uniform(-1, 1, 2)
            individual += mutation
        return np.clip(individual, self.bounds[0], self.bounds[1])

    def optimize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.get_current_fig_manager().window.state('zoomed')
        plt.ion()
        for iter in range(self.max_iter):
            scores = self.evaluate()
            parents = self.select_parents(scores)
            next_population = []
            for i in range(0, self.pop_size, 2):
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(child1))
                next_population.append(self.mutate(child2))
            self.population = np.array(next_population)
            self.plot(ax, iter)
        plt.ioff()
        plt.show()

    def plot(self, ax, iter):
        ax.cla()
        X, Y = np.meshgrid(np.linspace(self.bounds[0], self.bounds[1], 100),
                           np.linspace(self.bounds[0], self.bounds[1], 100))
        Z = self.objective_function(X, Y)
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        for ind in self.population:
            ax.scatter(ind[0], ind[1],
                       self.objective_function(ind[0], ind[1]), color='red')
        ax.scatter(self.best_individual[0], self.best_individual[1],
                   self.objective_function(self.best_individual[0], self.best_individual[1]),
                   color='blue', marker='*', s=100)
        ax.set_title(f'Iteration {iter}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.pause(1)

# Function to start optimization based on user-selected algorithm
def start_optimization(algorithm, pop_size, max_iter):
    bounds = [-10, 10]
    if algorithm == 'PSO':
        optimizer = ParticleSwarmOptimizer(objective_function, pop_size, bounds, max_iter)
    elif algorithm == 'GA':
        optimizer = GeneticAlgorithm(objective_function, pop_size, bounds, max_iter)
    optimizer.optimize()

# Function to create the user interface
# Function to create the user interface
# Function to create the user interface
def create_ui():
    root = tk.Tk()
    root.title("Параметры алгоритма оптимизации")
    root.geometry("400x400")  # Set window size

    mainframe = ttk.Frame(root, padding="10 10 10 10")
    mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    algorithm_var = tk.StringVar()
    pop_size_var = tk.IntVar()
    max_iter_var = tk.IntVar()

    ttk.Label(mainframe, text="Выберите алгоритм:").grid(column=1, row=1, sticky=tk.W)

    pso_radio = ttk.Radiobutton(mainframe, text="PSO", variable=algorithm_var, value="PSO")
    pso_radio.grid(column=2, row=1, sticky=tk.W)

    ga_radio = ttk.Radiobutton(mainframe, text="GA", variable=algorithm_var, value="GA")
    ga_radio.grid(column=3, row=1, sticky=tk.W)

    ttk.Label(mainframe, text="Размер популяции:").grid(column=1, row=2, sticky=tk.W)
    pop_size_entry = ttk.Entry(mainframe, width=7, textvariable=pop_size_var)
    pop_size_entry.grid(column=2, row=2, sticky=(tk.W, tk.E))

    ttk.Label(mainframe, text="Количество итераций:").grid(column=1, row=3, sticky=tk.W)
    max_iter_entry = ttk.Entry(mainframe, width=7, textvariable=max_iter_var)
    max_iter_entry.grid(column=2, row=3, sticky=(tk.W, tk.E))

    def on_start():
        algorithm = algorithm_var.get()
        pop_size = pop_size_var.get()
        max_iter = max_iter_var.get()
        root.destroy()
        start_optimization(algorithm, pop_size, max_iter)

    ttk.Button(mainframe, text="Запуск", command=on_start).grid(column=1, row=4, columnspan=3)

    for child in mainframe.winfo_children():
        child.grid_configure(padx=5, pady=5)

    root.mainloop()

# Create the user interface
create_ui()
