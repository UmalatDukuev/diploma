import numpy as np
from scipy.optimize import minimize

# Определяем целевую функцию
def objective_function(x):
    return np.sin(5 * x) * np.sin(3 * x) * np.sin(x)

# Начальное приближение для оптимизации
initial_guess = 0

# Минимизируем целевую функцию
result = minimize(objective_function, initial_guess)

# Выводим результат
print("Минимум функции:", result.x[0])
print("Значение функции в минимуме:", result.fun)