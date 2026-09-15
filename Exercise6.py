import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def exact_solution(t):
    return 2.0 / (5.0 - 4.0 * np.sin(np.pi * t / 2.0))

total_time = 5.0
m_interval = range(1, 12)
h_values = []
error_backward = []
error_CN = []

for m in m_interval:
    n = 10 * (2**m)
    h = total_time / n
    h_values.append(h)

    t_grid = np.linspace(0, total_time, n + 1)
    initial_backward_Euler = np.zeros(n + 1)
    initial_Crank_Nickolson = np.zeros(n + 1)
    
    initial_backward_Euler[0] = 0.4
    initial_Crank_Nickolson[0] = 0.4
    
    for i in range(n):
        initial_time = t_grid[i]
        final_time = t_grid[i+1]
        # --- Backward Euler Step ---
        a = h * np.pi * np.cos(np.pi * final_time / 2.0)
        b = -1.0
        c = initial_backward_Euler[i]
        if abs(a) < 1e-14:
            initial_backward_Euler[i+1] = initial_backward_Euler[i]
        else:
            discriminant = max(0.0,b**2 - 4 * a * c)
            initial_backward_Euler[i+1] = (-b - np.sqrt(discriminant)) / (2 * a)      
        # --- Crank-Nicolson Step ---
        a1 = 0.5 * h * np.pi * np.cos(np.pi * final_time / 2.0)
        b1 = -1.0
        c1 = initial_Crank_Nickolson[i] + 0.5 * h * np.pi * (initial_Crank_Nickolson[i]**2) * np.cos(np.pi * initial_time / 2.0)
        if abs(a1) < 1e-14:
            initial_Crank_Nickolson[i+1] = c1
        else:
            discriminant1 = max(0.0, b1**2 - 4 * a1 * c1)
            initial_Crank_Nickolson[i+1] = (-b1 - np.sqrt(discriminant1)) / (2 * a1)  
    exact_vals = exact_solution(t_grid)
    error_backward.append(np.max(np.abs(initial_backward_Euler - exact_vals)))
    error_CN.append(np.max(np.abs(initial_Crank_Nickolson - exact_vals)))

value_log_h = np.log(h_values)
beta_backward_Euler, intercept_backward_Euler, _, _, _ = linregress(value_log_h, np.log(error_backward))
beta_CN, intercept_cn, _, _, _ = linregress(value_log_h, np.log(error_CN))
plt.figure(figsize=(9, 5))
plt.loglog(h_values, error_backward, 'o-', label=f'Backward Euler (slope ≈ {beta_backward_Euler:.2f})')
plt.loglog(h_values, error_CN, 's-', label=f'Crank-Nicolson (slope ≈ {beta_CN:.2f})')
plt.xlabel('Step size (h)')
plt.ylabel('Error')
plt.title('Error vs. Step Size')
plt.legend()
plt.grid(True)
plt.show()