import torch
import numpy as np
import models

a = models.MiniSqueezeNet(1, 10)

b = np.random.random((10, 1, 64, 64))
b = torch.Tensor(b)

c = a(b)
print(c)
print(c.shape)
