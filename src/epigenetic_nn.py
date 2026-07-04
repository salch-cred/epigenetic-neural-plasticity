import numpy as np

class EpigeneticLayer:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(out_features, in_features) * 0.1
        self.bias = np.zeros((out_features, 1))
        # Epigenetic markers: 1.0 means fully active, 0.0 means completely silenced
        self.methylation = np.ones((out_features, in_features))
        self.acetylation = np.ones((out_features, in_features))
        
    def forward(self, x):
        # Apply epigenetic markers directly to modify the active weights
        self.active_weights = self.weights * self.methylation * self.acetylation
        return np.dot(self.active_weights, x) + self.bias
        
    def update_epigenetics(self, gradients, adaptation_rate=0.05):
        # Simulate environmental feedback: high gradient variance triggers methylation (silencing)
        # to protect stable memories, while low variance triggers acetylation (activation)
        grad_variance = np.var(gradients, axis=0, keepdims=True)
        self.methylation = np.clip(self.methylation - adaptation_rate * grad_variance, 0.0, 1.0)
        self.acetylation = np.clip(self.acetylation + adaptation_rate * (1.0 - grad_variance), 1.0, 2.0)
