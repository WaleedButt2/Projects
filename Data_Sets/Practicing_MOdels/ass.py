import os
import deepspeed
import torch
from transformers import pipeline


if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)         
            