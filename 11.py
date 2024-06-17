import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from scipy.optimize import differential_evolution

def objective_function(x, y):
    A = 10
    #return x**2 + y**2 # параболоид
    #return (1 - x)**2 + 100 * (y - x**2)**2 # Функция Розенброка
    #return A * 2 + (x ** 2 - A * np.cos(2 * np.pi * x)) + (y ** 2 - A * np.cos(2 * np.pi * y)) # Функция Растригина
    return 0.26*(x**2+y**2)-0.48*(x*y)

class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], 2)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

    def update_personal_best(self, objective_function):
        score = objective_function(*self.position)
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position.copy()

class ParticleSwarmOptimizer:
    def __init__(self, objective_function, pop_size, bounds, max_iter, epsilon=None, target=None):
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.bounds = bounds
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.particles = [Particle(bounds) for _ in range(pop_size)]
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.target = target
        self.particle_paths = [[] for _ in range(pop_size)]  # Добавляем список для хранения путей каждой частицы

    def find_initial_global_minimum(self):
        result = differential_evolution(lambda pos: self.objective_function(pos[0], pos[1]), [self.bounds, self.bounds])
        self.global_best_position = result.x
        self.global_best_score = result.fun

    def evaluate(self):
        for particle in self.particles:
            particle.update_personal_best(self.objective_function)
            if particle.best_score < self.global_best_score:
                self.global_best_score = particle.best_score
                self.global_best_position = particle.best_position.copy()

    def update_velocities_and_positions(self, inertia=0.5, cognitive=1.5, social=1.5):
        for i, particle in enumerate(self.particles):
            r1, r2 = np.random.rand(2), np.random.rand(2)
            cognitive_velocity = cognitive * r1 * (particle.best_position - particle.position)
            social_velocity = social * r2 * (self.global_best_position - particle.position)
            particle.velocity = inertia * particle.velocity + cognitive_velocity + social_velocity
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])
            # Сохраняем текущее положение частицы в пути
            self.particle_paths[i].append(particle.position.copy())

    def optimize(self):
        self.find_initial_global_minimum()  # Find global minimum using differential evolution
        plt.ion()
        for iter in range(self.max_iter):
            self.evaluate()
            self.update_velocities_and_positions()
            self.plot(iter)
            # Вывод координат лучшей и худшей точки и их расстояния до глобального минимума
            distances = [np.linalg.norm(particle.position - self.target) for particle in self.particles]
            best_particle = self.particles[np.argmin(distances)]
            worst_particle = self.particles[np.argmax(distances)]
            best_distance = np.min(distances)
            worst_distance = np.max(distances)
            print(f"Iteration {iter}: Best Position: {best_particle.position}, Distance: {best_distance}")
            print(f"Iteration {iter}: Worst Position: {worst_particle.position}, Distance: {worst_distance}")
            if self.epsilon is not None and all(distance < self.epsilon for distance in distances):
                break
        plt.ioff()
        plt.show()
        print(f"Global Minimum: Position: {self.global_best_position}, Score: {self.global_best_score}")

    def plot_paths(self, iter):
        for i, path in enumerate(self.particle_paths):
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], color='white', alpha=0.3)  # Тонкая белая линия для пути частицы i

    def plot(self, iter):
        plt.clf()
        x = np.linspace(self.bounds[0], self.bounds[1], 100)
        y = np.linspace(self.bounds[0], self.bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.objective_function(X, Y)
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Objective Function Value')
        for particle in self.particles:
            plt.scatter(*particle.position, color='red')
        plt.scatter(*self.target, color='green', marker='x', s=100, label='Target')
        self.plot_paths(iter)  # Добавляем отображение путей частиц
        plt.title(f'Iteration {iter}')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.legend()
        plt.pause(0.003)


class GeneticAlgorithm:
    def __init__(self, objective_function, pop_size, bounds, max_iter, epsilon=None,
                 mutation_rate=0.1, crossover_rate=0.8, elitism=True, target=None):
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.bounds = bounds
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.target = target
        self.population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, 2))
        self.velocities = np.random.uniform(-1, 1, size=(pop_size, 2))
        self.best_individual = self.population[0]
        self.best_score = self.objective_function(*self.best_individual)
        self.paths = [[] for _ in range(pop_size)]
        self.evaluate()

    def evaluate(self):
        scores = np.array([self.objective_function(ind[0], ind[1]) for ind in self.population])
        best_index = np.argmin(scores)
        if scores[best_index] < self.best_score:
            self.best_score = scores[best_index]
            self.best_individual = self.population[best_index].copy()
        print(f'Best Score: {self.best_score}, Best Position: {self.best_individual}')
        return scores

    def select_parents(self, scores):
        scores = scores - np.min(scores) + 1e-10
        selection_probabilities = 1 / scores
        selection_probabilities /= np.sum(selection_probabilities)
        parents_indices = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=selection_probabilities)
        return self.population[parents_indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.uniform(0.3, 0.7)
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        mutated_individual = individual.copy()
        for i in range(2):
            if np.random.rand() < self.mutation_rate:
                mutated_individual[i] += np.random.uniform(-0.5, 0.5)
                mutated_individual[i] = np.clip(mutated_individual[i], self.bounds[0], self.bounds[1])
        return mutated_individual

    def update_velocities(self, c1=1.5, c2=1.5, w=0.5, perturbation=0.1):
        for i in range(self.pop_size):
            cognitive_component = c1 * np.random.rand() * (self.best_individual - self.population[i])
            social_component = c2 * np.random.rand() * (self.target - self.population[i])
            random_perturbation = perturbation * (np.random.rand(2) - 0.5)
            self.velocities[i] = w * self.velocities[i] + cognitive_component + social_component + random_perturbation
            self.population[i] += self.velocities[i]
            self.population[i] = np.clip(self.population[i], self.bounds[0], self.bounds[1])

    def optimize(self):
        plt.ion()
        for iter in range(self.max_iter):
            scores = self.evaluate()
            parents = self.select_parents(scores)
            next_population = []
            if self.elitism:
                elite_index = np.argmin(scores)
                elite = self.population[elite_index].copy()
                next_population.append(elite)
            for i in range(0, self.pop_size - (1 if self.elitism else 0), 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(child1))
                if len(next_population) < self.pop_size:
                    next_population.append(self.mutate(child2))
            self.population = np.array(next_population[:self.pop_size])
            self.update_velocities(c1=1.5 - iter * 1.4 / self.max_iter,
                                   c2=1.5,
                                   perturbation=0.1 - iter * 0.09 / self.max_iter)
            for i, individual in enumerate(self.population):
                self.paths[i].append(individual.copy())
            self.plot(iter)
            distances = np.linalg.norm(self.population - self.target, axis=1)
            best_individual = self.population[np.argmin(distances)]
            worst_individual = self.population[np.argmax(distances)]
            best_distance = np.min(distances)
            worst_distance = np.max(distances)
            print(f"Iteration {iter}: Best Position: {best_individual}, Distance: {best_distance}")
            print(f"Iteration {iter}: Worst Position: {worst_individual}, Distance: {worst_distance}")
            if self.epsilon is not None and all(distance < self.epsilon for distance in distances):
                break
        plt.ioff()
        plt.show()
        print(f"Global Minimum: Position: {self.best_individual}, Score: {self.best_score}")

    def plot_paths(self, iter):
        for i, path in enumerate(self.paths):
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], color='white', alpha=0.3)

    def plot(self, iter):
        plt.clf()
        x = np.linspace(self.bounds[0], self.bounds[1], 100)
        y = np.linspace(self.bounds[0], self.bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.objective_function(X, Y)
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Objective Function Value')
        for ind in self.population:
            plt.scatter(ind[0], ind[1], color='red')
        plt.scatter(*self.target, color='green', marker='x', s=100, label='Target')
        self.plot_paths(iter)
        plt.title(f'Iteration {iter}')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.legend()
        plt.pause(0.003)
def start_optimization(algorithm, pop_size, max_iter, bounds, epsilon, use_epsilon):
    result = differential_evolution(lambda pos: objective_function(pos[0], pos[1]), [bounds, bounds])
    global_minimum = result.x
    print(f"Initial Global Minimum found by Differential Evolution: {global_minimum}")

    if algorithm == 'PSO':
        optimizer = ParticleSwarmOptimizer(objective_function, pop_size, bounds, max_iter,
                                           epsilon if use_epsilon else None, target=global_minimum)
        print('PSO')
    elif algorithm == 'GA':
        optimizer = GeneticAlgorithm(objective_function, pop_size, bounds, max_iter,
                                     epsilon if use_epsilon else None, target=global_minimum)
        print('GA')
    optimizer.optimize()

def create_ui():
    root = tk.Tk()
    root.title("Optimization Algorithm Parameters")
    root.geometry("400x400")

    mainframe = ttk.Frame(root, padding="10 10 10 10")
    mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    algorithm_var = tk.StringVar()
    pop_size_var = tk.IntVar()
    max_iter_var = tk.IntVar()

    lower_bound_var = tk.DoubleVar()
    upper_bound_var = tk.DoubleVar()
    epsilon_var = tk.DoubleVar()
    use_epsilon_var = tk.BooleanVar()

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

    ttk.Label(mainframe, text="Lower Bound:").grid(column=1, row=4, sticky=tk.W)
    lower_bound_entry = ttk.Entry(mainframe, width=7, textvariable=lower_bound_var)
    lower_bound_entry.grid(column=2, row=4, sticky=(tk.W, tk.E))

    ttk.Label(mainframe, text="Upper Bound:").grid(column=1, row=5, sticky=tk.W)
    upper_bound_entry = ttk.Entry(mainframe, width=7, textvariable=upper_bound_var)
    upper_bound_entry.grid(column=2, row=5, sticky=(tk.W, tk.E))

    ttk.Label(mainframe, text="Epsilon:").grid(column=1, row=6, sticky=tk.W)
    epsilon_entry = ttk.Entry(mainframe, width=7, textvariable=epsilon_var)
    epsilon_entry.grid(column=2, row=6, sticky=(tk.W, tk.E))

    use_epsilon_check = ttk.Checkbutton(mainframe, text="Use Epsilon", variable=use_epsilon_var)
    use_epsilon_check.grid(column=3, row=6, sticky=tk.W)

    def on_start():
        algorithm = algorithm_var.get()
        pop_size = pop_size_var.get()
        max_iter = max_iter_var.get()
        lower_bound = lower_bound_var.get()
        upper_bound = upper_bound_var.get()
        epsilon = epsilon_var.get()
        use_epsilon = use_epsilon_var.get()
        bounds = [lower_bound, upper_bound]
        root.destroy()
        start_optimization(algorithm, pop_size, max_iter, bounds, epsilon, use_epsilon)

    ttk.Button(mainframe, text="Start", command=on_start).grid(column=1, row=7, columnspan=3)
    for child in mainframe.winfo_children():
        child.grid_configure(padx=5, pady=5)
    root.mainloop()

create_ui()
