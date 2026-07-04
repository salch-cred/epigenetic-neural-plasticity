import numpy as np
from src.epigenetic_nn import EpigeneticLayer

print("Initializing Epigenetic Neural Plasticity Simulation...")
layer = EpigeneticLayer(in_features=5, out_features=3)
x = np.random.randn(5, 1)

print("Original weights shape:", layer.weights.shape)
print("Initial Methylation state (mean):", np.mean(layer.methylation))

# Forward pass
output = layer.forward(x)
print("Output from forward pass:\n", output)

# Simulate training and epigenetic updates
dummy_gradients = np.random.randn(3, 5) * 1.5
layer.update_epigenetics(dummy_gradients)
print("Post-update Methylation state (mean):", np.mean(layer.methylation))
print("Plasticity adaptation successfully verified.")
