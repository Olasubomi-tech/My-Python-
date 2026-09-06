import numpy as np

def f(x):
    return np.sin(x)

x0 = np.pi / 3.0
exact_f2 = -np.sin(x0)
alpha_list=[0.2,0.5,1]
prev_error = None
header_approx = "Approx f''(x0)"


for i in alpha_list:
    print(" ")
    print(f"For alpha ={i}, we have the table below")
    print(f"{'n':<3} | {'h':<12} | {header_approx:<20} | {'Error':<16} | {'Order (p)':<8}")
    print("-" * 72)
    for n in range(1, 13):
        h = 2.0**(-n)
        term1 = (2.0 / ((1.0 + i) * h**2)) * f(x0 - h)
        term2 = -(2.0 / (i * h**2)) * f(x0)
        term3 = (2.0 / (i * (1.0 + i) * h**2)) * f(x0 + i * h)
        approx = term1 + term2 + term3
        error = abs(approx - exact_f2)
        if prev_error is not None:
            p = np.log2(prev_error / error)
            order_str = f"{p:.4f}"
        else:
            order_str = "N/A"
        print(f"{n:<3} | {h:<12.6e} | {approx:<20.14f} | {error:<16.10e} | {order_str:<8}")
        prev_error = error