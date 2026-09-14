import numpy as np
import matplotlib.pyplot as plt

def actual_function(x):
    return x-(np.exp(x)-1)

def derivative(x):
    return 1-np.exp(x)

true_root = 0 

iterate = 40
count = 0

def Newton_iterative_scheme(p0, tolerance, iterate):
    List_of_approximate = [p0]
    count = 0
    x = p0
    if abs(derivative(x)) < 1e-8:
        raise ZeroDivisionError("The derivative is zero or near zero at starting point.")
    d =  2* actual_function(x)/derivative(x)
    while abs(d) >= tolerance and count < iterate:
        x = x - d
        List_of_approximate.append(x)
        count +=1
        d =  2*actual_function(x)/derivative(x)

    print(f"The Value of the root is: {x:.4f} with {count} iterations.")
    return x, List_of_approximate

root, p_n_values = Newton_iterative_scheme(0.5,1e-14, 6)
p = 0
p_n_values = np.array(p_n_values)
errors = np.abs(p_n_values - p)
iterate = 2
asymptotic_error_ratio = np.abs(p_n_values[iterate] - p)/(np.abs(p_n_values[iterate-1] - p)**2)
print(f"The asymptotic error is approximately {asymptotic_error_ratio:.1f}")
Arranged_values = np.arange(len(errors))


plt.figure(figsize=(8, 5))
plt.semilogy(Arranged_values, errors, color='b', label=r'Error $|p_n - p|$')
plt.title("Error of Modified Newton's Method vs. Iteration Number $n$")
plt.xlabel("Iteration $n$")
plt.ylabel(r"Error $|p_n - p|$ (Log Scale)")
plt.grid(True)
plt.legend()
plt.show()
