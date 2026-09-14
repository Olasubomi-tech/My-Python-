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

f1_l2 = np.sqrt((2 / 1000) * np.sum(f_1(a_k)**2))
f2_l2 = np.sqrt((2 / 1000) * np.sum(f_2(a_k)**2))
error_inf_function1 = []
error_inf_function2 = []

Relative_error_inf_function1 = []
Relative_error_inf_function2 = []

for n in n_values:
    nodes = chebyshev_nodes(n)

    y1_nodes = f_1(nodes)
    P1 = lagrange_interpolation(nodes, y1_nodes, a_k)
    err1 = np.sqrt((2/1000)*np.sum((f_1(a_k) - P1)**2))
    error_inf_function1.append( err1)
    Relative_error_inf_function1.append(err1/f1_l2)

    y2_nodes = f_2(nodes)
    P2 = lagrange_interpolation(nodes, y2_nodes, a_k)
    err2 = np.sqrt((2/1000)*np.sum((f_2(a_k) - P2)**2))
    error_inf_function2.append(err2)
    Relative_error_inf_function2.append(err2/f2_l2)

print(f"The L^2 error for the first function is {error_inf_function1[-1]}")
print(f"The L^2 error for the second function is {error_inf_function2[-1]}")


fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.semilogy(n_values, Relative_error_inf_function1, 'b-o', markersize=3, label=r'(i) $f(x) = \sin(|x|)$')
ax2.semilogy(n_values, Relative_error_inf_function2, 'r-s', markersize=3, label=r'(ii) $f(x) = \mathrm{sign}(x)\cos(x)$')
ax2.set_title(r"Interpolation Error $\|f - P_{n-1}\|_2$ vs $n$ (Log Scale)")
ax2.set_xlabel("Number of nodes $n$")
ax2.set_ylabel(r"$\|f - P_{n-1}\|_2$")
ax2.set_xlim(1, 30)
ax2.set_ylim(1e-1, 1e4)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()