

import torch

from typing import Dict

from linear import Linear

def run_linear(in_features: int,
               out_features: int,
               weights: Dict[str, torch.Tensor],
               input: torch.Tensor):
    model = Linear(in_features=in_features,
                   out_features=out_features)
    model.load_state_dict({"_weights": weights})
    return model(input)
