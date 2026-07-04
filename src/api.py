from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.epigenetic_nn import EpigeneticLayer

app = FastAPI(title="Epigenetic Neural Plasticity API")
# Instantiating the layer globally
layer = EpigeneticLayer(in_features=5, out_features=3)

class InputVector(BaseModel):
    x: List[float]

@app.post("/forward")
def forward_pass(data: InputVector):
    import numpy as np
    if len(data.x) != 5:
        return {"error": "Input vector must have exactly 5 elements."}
    x_arr = np.array(data.x).reshape(5, 1)
    output = layer.forward(x_arr)
    # Simulate update with dummy gradient
    dummy_grad = np.random.randn(3, 5) * 0.1
    layer.update_epigenetics(dummy_grad)
    return {
        "output": output.flatten().tolist(),
        "methylation": layer.methylation.tolist(),
        "acetylation": layer.acetylation.tolist()
    }
