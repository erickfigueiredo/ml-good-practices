import os
import torch


# Processing constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_WORKERS = os.cpu_count()

# Training constants
EPOCHS = 30
BATCH_SIZE = 32
