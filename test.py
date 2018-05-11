import numpy as np
import torch 

sum_mask = torch.from_numpy(np.asarray([[8, 8, 8], [8, 0, 8], [8, 8, 8]])).float()
out = torch.from_numpy(np.asarray([[0.2, 0.2, 0.2], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])).float()

out = out / (sum_mask + 1e-20)
out = out * (1 - (out > 1e+10).float())

print(out)