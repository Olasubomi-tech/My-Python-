import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


weights = {
    0: ([(0.0, 2.0)]),
    1: ([(-0.5773502691896257, 1.0), (0.5773502691896257, 1.0)]),
    2: ([(-0.7745966692414834, 0.5555555555555556), (0.0, 0.8888888888888888), (0.7745966692414834, 0.5555555555555556)]),
    3: ([(-0.8611363115940526, 0.3478548451374538), (-0.3399810435848563, 0.6521451548625461),
         (0.3399810435848563, 0.6521451548625461), (0.8611363115940526, 0.3478548451374538)]),
    4: ([(-0.9061798459386640, 0.2369268850561891), (-0.5384693101056831, 0.4786286704993665),
         (0.0, 0.5688888888888889), (0.5384693101056831, 0.4786286704993665), (0.9061798459386640, 0.2369268850561891)]),
    5: ([(-0.9324695142031521, 0.1713244923791704), (-0.6612093864662645, 0.3607615730481386),
         (-0.2386191860831969, 0.4679139345726910), (0.2386191860831969, 0.4679139345726910),
         (0.6612093864662645, 0.3607615730481386), (0.9324695142031521, 0.1713244923791704)]),
    6: ([(-0.9491079123427585, 0.1294849661688697), (-0.7415311855993945, 0.2797053914892766),
         (-0.4058451513773972, 0.3818300505051189), (0.0, 0.4179591836734694),
         (0.4058451513773972, 0.3818300505051189), (0.7415311855993945, 0.2797053914892766),
         (0.9491079123427585, 0.1294849661688697)]),
    7: ([(-0.9602898564975363, 0.1012285362903763), (-0.7966664774136267, 0.2223810344533745),
         (-0.5255324099163290, 0.3137066458778873), (-0.1834346424956498, 0.3626837833783620),
         (0.1834346424956498, 0.3626837833783620), (0.5255324099163290, 0.3137066458778873),
         (0.7966664774136267, 0.2223810344533745), (0.9602898564975363, 0.1012285362903763)])}

def function(x):
    return 101.0 * np.exp(x) * np.sin(10.0 * x)

def composite_quad(function, a,b,n,m):
    h = (b - a) / m
    r = np.array([i[0] for i in weights[n]])
    w = np.array([i[1] for i in weights[n]])
    integral_sum = 0.0
    for i in range(m):
        x_previous = a + i * h

        xi = a + (i + 1) * h
        nodes = (h / 2.0) * r + (x_previous + xi) / 2.0
        integral_sum += (h / 2.0) * np.sum(w * function(nodes))
    return integral_sum

m_interval =  [2**k for k in range(1, 11)]
Exact_integral = np.exp(4.0) * (np.sin(40.0) - 10.0 * np.cos(40.0)) + 10.0
h = [4.0 / m for m in m_interval]

plt.figure(figsize=(7, 4.5))

for n in range(3):
    absolute_list = []
    rel_errors = []
    for m in m_interval:
        value = composite_quad(function, 0.0, 4.0, n, m)
        absolute_error = abs(value - Exact_integral)
        absolute_list.append(absolute_error)
        rel_err = abs(value - Exact_integral) / abs(Exact_integral)
        rel_errors.append(rel_err)
        
    beta, intercept, _, _, _ = linregress(np.log(h), np.log(rel_errors))
    alpha = np.exp(intercept)
    print(f"n = {n}: Estimated Alpha = {alpha:.4f}, Estimated beta = {beta:.4f} (Actual = {2*n+2})")
    plt.loglog(h, absolute_list, 'o-', label=f'n={n} (alpha = {alpha:.2f}, beta ≈ {beta:.2f})')
plt.xlabel('Step size h')
plt.ylabel('Absolute Error')
plt.title(' Absolute Error vs. h')
plt.legend()
plt.grid(True)
plt.show()