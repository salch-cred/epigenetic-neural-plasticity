# Epigenetic Neural Plasticity

A first-of-its-kind machine learning framework inspired by biological epigenetics (DNA methylation and histone modification). 

Traditional neural networks modify their weights directly during learning. This project implements a system where weights (the "DNA") remain stable, while dynamic "epigenetic gates" (representing methylation/acetylation) act as regulatory elements that silence or amplify weight activations based on environmental feedback and training history.

## Architecture
- **Static DNA Weight Layer**: The core, long-term memory weights of the network.
- **Epigenetic State Vectors**: Dynamic gates representing $M(t)$ (methylation) and $A(t)$ (acetylation) that modify layer outputs:
  $$y = \sigma((W \odot M(t) \cdot A(t))x + b)$$
- **Environmental Feedback Loop**: Regulates the epigenetic state based on local loss gradient variance.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the demo to train the model on a dynamically changing task:
```bash
python examples/train_epigenetic_plasticity.py
```
