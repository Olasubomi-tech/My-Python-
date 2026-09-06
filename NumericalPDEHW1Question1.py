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
    d =  actual_function(x)/derivative(x)

    while abs(d) >= tolerance and count < iterate:
        x = x - d
        List_of_approximate.append(x)
        count +=1

        d =  actual_function(x)/derivative(x)

    print(f"The Value of the root is: {x:.4f} with {count} iterations.")
    return x, List_of_approximate

root, p_n_values = Newton_iterative_scheme(0.5,0, 30)
p = 0
p_n_values = np.array(p_n_values)
errors = np.abs(p_n_values - p)
iterate = 15
asymptotic_error_ratio = np.abs(p_n_values[iterate] - p)/(np.abs(p_n_values[iterate-1] - p))
print(f"The asymptotic error is approximately {asymptotic_error_ratio:.1f}")
Arranged_values = np.arange(len(errors))

#To use the code below, increase the iterate from 20 to 50, to see at what iterate the method start to diverge
#fail_iteration = np.where(np.diff(errors) > 0)[0][0] + 1
#print(f"The method starts to fail at iteration n = {fail_iteration}")

plt.figure(figsize=(8, 5))
plt.semilogy(Arranged_values, errors, color='b', label=r'Error $|p_n - p|$')
plt.title("Error of Newton's Method vs. Iteration Number $n$")
#plt.axvline(x=fail_iteration, color='r', linestyle='--', linewidth=1.5, label=f'Method diverges (n = {fail_iteration})')
plt.xlabel("Iteration $n$")
plt.ylabel(r"Error $|p_n - p|$ (Log Scale)")
plt.grid(True)
plt.legend()
plt.show()
