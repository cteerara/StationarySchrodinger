from scipy.integrate import quad
import math

def integrand_cos(x):
    return math.cos(x);

result = quad(integrand_cos, 0, 3.14/2)
print(result)
