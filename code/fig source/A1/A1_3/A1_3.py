import numpy as np
import matplotlib.pyplot as plt

# Create input range
x = np.linspace(-10, 10, 100)

# ReLU function
relu = np.maximum(0, x)

# Tanh function
tanh = np.tanh(x)

# Sigmoid function
sigmoid = 1 / (1 + np.exp(-x))

# Plotting
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, relu, 'b-')
plt.title('ReLU')
plt.xlabel('input')
plt.ylabel('output')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, tanh, 'r-')
plt.title('Tanh')
plt.xlabel('input')
plt.ylabel('output')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, sigmoid, 'g-')
plt.title('Sigmoid')
plt.xlabel('input')
plt.ylabel('output')
plt.grid(True)

plt.tight_layout()
plt.savefig('activation_functions.pdf', dpi=300)