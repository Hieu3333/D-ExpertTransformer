import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Define ReLU function
def relu(x):
    return np.maximum(0, x)

# Define GELU function (approximation using tanh form for better performance)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Generate x values
x = np.linspace(-5, 5, 1000)

# Compute ReLU and GELU values
y_relu = relu(x)
y_gelu = gelu(x)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y_relu, label="ReLU", color="blue")
plt.plot(x, y_gelu, label="GELU", color="red")
plt.title("ReLU vs GELU Activation Functions")
plt.xlabel("x")
plt.ylabel("Activation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
