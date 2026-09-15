import numpy as np
import matplotlib.pyplot as plt

def exact_solution(t):
    return 2.0 / (5.0 - 4.0 * np.sin(np.pi * t / 2.0))
total_time = 5.0
n = 1000
h = total_time / n
t_grid = np.linspace(0, total_time, n + 1)

initial_backward_Euler = np.zeros(n+1)
initial_Crank_Nickolson = np.zeros(n+1)
initial_backward_Euler[0] = 0.4
initial_Crank_Nickolson[0] = 0.4

for i in range(n):
    initial_time = t_grid[i]
    final_time = t_grid[i+1]
    # we now solve the quadratic formula for the backward euler
    a,b,c= h * np.pi * np.cos(np.pi * final_time / 2.0), -1, initial_backward_Euler[i]
    if abs(a) < 1e-14:
        initial_backward_Euler[i+1]= initial_backward_Euler[i]
    else:
        discriminant = (b)**2 -4*a*c
        initial_backward_Euler[i+1] = (-b-np.sqrt(discriminant)) / (2*a)
    a1,b1,c1 = 0.5 * h * np.pi * np.cos(np.pi * final_time / 2.0), -1 , initial_Crank_Nickolson[i] + 0.5 * h * np.pi * (initial_Crank_Nickolson[i]**2) * np.cos(np.pi * initial_time / 2.0)

    if abs(a1) < 1e-14:
        initial_Crank_Nickolson[i+1] = c1
    else:
        discriminant1 = (b1)**2 - 4*a1*c1

        initial_Crank_Nickolson[i+1] = (-b1-np.sqrt(discriminant1))/(2*a1)  
plt.figure(figsize=(9, 5))
plt.plot(t_grid, exact_solution(t_grid), 'k-', label='Exact Solution', linewidth=2)
plt.plot(t_grid, initial_backward_Euler, 'r--', label='Backward Euler, n=1000')
plt.plot(t_grid, initial_Crank_Nickolson, 'b:', label='Crank-Nicolson, n=1000')
plt.xlabel('Time(t)')
plt.ylabel('y(t)')
plt.title('Numerical vs. Analytical Solution')
plt.legend()
plt.grid(True)
plt.show()

