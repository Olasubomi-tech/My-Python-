import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


def f_1(x):
    return np.sin(np.abs(x))

def f_2(x):
    return np.sign(x) * np.cos(x)

def chebyshev_nodes(n):
    j = np.arange(n)
    return np.cos((2 * j + 1) * np.pi / (2 * n))

def lagrange_interpolation(x_nodes, y_nodes, x_eval):
    n = len(x_nodes)
    j = np.arange(n)
    weights = (-1)**j * np.sin((2 * j + 1) * np.pi / (2 * n))

    x_eval = np.asarray(x_eval)
    P_eval = np.zeros_like(x_eval, dtype=float)

    for i, x in enumerate(x_eval):
        exact = np.isclose(x - x_nodes, 0, atol=1e-15)
        if np.any(exact):
            P_eval[i] = y_nodes[exact][0]
        else:
            P_eval[i] = np.sum((weights / (x - x_nodes)) * y_nodes) / np.sum(weights / (x - x_nodes))
            
    return P_eval
a_k = np.linspace(-1, 1, 1000)

n_values = np.arange(1, 101)
error_inf_function1 = []
error_inf_function2 = []

for n in n_values:
    nodes = chebyshev_nodes(n)
    y1_nodes = f_1(nodes)
    P1 = lagrange_interpolation(nodes, y1_nodes, a_k)
    error_inf_function1.append(np.max(np.abs(f_1(a_k) - P1)))
    y2_nodes = f_2(nodes)
    P2 = lagrange_interpolation(nodes, y2_nodes, a_k)
    error_inf_function2.append(np.max(np.abs(f_2(a_k) - P2)))

print(f"The L^infinity error for the first function is {error_inf_function1[-1]}")
print(f"The L^infinity error for the second function is {error_inf_function2[-1]}")

n_demo = 21
nodes_demo = chebyshev_nodes(n_demo)

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(a_k, f_1(a_k), 'b--', linewidth=1.5, label=r'(i) $f(x) = \sin(|x|)$')
ax1.plot(a_k, lagrange_interpolation(nodes_demo, f_1(nodes_demo), a_k), 'b-', label=r'(i) $P_{19}(x)$')
ax1.plot(a_k, f_2(a_k), 'r--', linewidth=1.5, label=r'(ii) $f(x) = \mathrm{sign}(x)\cos(x)$')
ax1.plot(a_k, lagrange_interpolation(nodes_demo, f_2(nodes_demo), a_k), 'r-', label=r'(ii) $P_{19}(x)$')
ax1.set_title(f"Polynomial Interpolation at Chebyshev Points ($n = {n_demo}$)")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.set_xlim(-1, 1)
ax1.set_ylim(-2, 2)
ax1.grid(True)
ax1.legend()
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.semilogy(n_values, error_inf_function1, 'b-o', markersize=3, label=r'(i) $f(x) = \sin(|x|)$')
ax2.semilogy(n_values, error_inf_function2, 'r-s', markersize=3, label=r'(ii) $f(x) = \mathrm{sign}(x)\cos(x)$')
ax2.set_title(r"Interpolation Error $\|f - P_{n-1}\|_\infty$ vs $n$ (Log Scale)")
ax2.set_xlabel("Number of nodes $n$")
ax2.set_ylabel(r"$\|f - P_{n-1}\|_\infty$")
ax2.set_xlim(1, 30)
ax2.set_ylim(1e-1, 1e4)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()