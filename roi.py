import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk

# Определение целевой функции с несколькими локальными минимумами и одним глобальным минимумом
def objective_function(x, y):
    return (x ** 2 + y ** 2) + 10 * np.sin(x) * np.cos(y)

# Инициализация начальной популяции частиц
def initialize_population(pop_size, bounds):
    positions = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
    velocities = np.random.uniform(-1, 1, (pop_size, 2))
    return positions, velocities

# Класс для реализации алгоритма роя частиц
class ParticleSwarmOptimizer:
    def __init__(self, objective_function, pop_size, bounds, max_iter):
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.bounds = bounds
        self.max_iter = max_iter
        self.positions, self.velocities = initialize_population(pop_size, bounds)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = self.evaluate(self.positions)
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)]
        self.global_best_score = np.min(self.personal_best_scores)

    def evaluate(self, positions):
        return np.apply_along_axis(lambda pos: self.objective_function(pos[0], pos[1]), 1, positions)

    def update_velocities_and_positions(self, inertia=0.5, cognitive=1.5, social=1.5):
        r1 = np.random.rand(self.pop_size, 2)
        r2 = np.random.rand(self.pop_size, 2)
        cognitive_velocity = cognitive * r1 * (self.personal_best_positions - self.positions)
        social_velocity = social * r2 * (self.global_best_position - self.positions)
        self.velocities = inertia * self.velocities + cognitive_velocity + social_velocity
        self.positions = self.positions + self.velocities
        # Ensure particles stay within bounds
        self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])

    def optimize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.get_current_fig_manager().window.state('zoomed')
        plt.ion()
        for iter in range(self.max_iter):
            scores = self.evaluate(self.positions)
            for i in range(self.pop_size):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_positions[i] = self.positions[i]
                    self.personal_best_scores[i] = scores[i]
            if np.min(scores) < self.global_best_score:
                self.global_best_position = self.positions[np.argmin(scores)]
                self.global_best_score = np.min(scores)
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
        ax.scatter(self.positions[:, 0], self.positions[:, 1], self.evaluate(self.positions), color='red')
        ax.scatter(self.global_best_position[0], self.global_best_position[1],
                   self.objective_function(self.global_best_position[0], self.global_best_position[1]),
                   color='blue', marker='*', s=100)
        ax.set_title(f'Iteration {iter}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.pause(1)

# Функция для запуска оптимизации с пользовательскими данными
def start_optimization(pop_size, max_iter):
    bounds = [-10, 10]
    pso = ParticleSwarmOptimizer(objective_function, pop_size, bounds, max_iter)
    pso.optimize()

# Функция для создания пользовательского интерфейса
def create_ui():
    root = tk.Tk()
    root.title("Параметры алгоритма роя частиц")
    root.geometry("400x400")  # Устанавливаем размер окна

    mainframe = ttk.Frame(root, padding="10 10 10 10")
    mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    pop_size_var = tk.IntVar()
    max_iter_var = tk.IntVar()

    ttk.Label(mainframe, text="Размер популяции:").grid(column=1, row=1, sticky=tk.W)
    pop_size_entry = ttk.Entry(mainframe, width=7, textvariable=pop_size_var)
    pop_size_entry.grid(column=2, row=1, sticky=(tk.W, tk.E))

    ttk.Label(mainframe, text="Количество итераций:").grid(column=1, row=2, sticky=tk.W)
    max_iter_entry = ttk.Entry(mainframe, width=7, textvariable=max_iter_var)
    max_iter_entry.grid(column=2, row=2, sticky=(tk.W, tk.E))

    def on_start():
        pop_size = pop_size_var.get()
        max_iter = max_iter_var.get()
        root.destroy()
        start_optimization(pop_size, max_iter)

    ttk.Button(mainframe, text="Запуск", command=on_start).grid(column=1, row=3, columnspan=2)

    for child in mainframe.winfo_children():
        child.grid_configure(padx=5, pady=5)

    root.mainloop()

# Создание пользовательского интерфейса
create_ui()
