import numpy as np
import matplotlib.pyplot as plt

def objective_function(x, y):
    A = 10
    #return x**2 + y**2 # параболоид
    #return (1 - x)**2 + 100 * (y - x**2)**2 # Функция Розенброка
    return A * 2 + (x ** 2 - A * np.cos(2 * np.pi * x)) + (y ** 2 - A * np.cos(2 * np.pi * y)) # Функция Растригина
    #return 0.26*(x**2+y**2)-0.48*(x*y)

# Определяем границы области
bounds = [-5, 5]
x = np.linspace(bounds[0], bounds[1], 100)
y = np.linspace(bounds[0], bounds[1], 100)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

# Построение графика
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Objective Function Value')
plt.title('Objective Function')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
