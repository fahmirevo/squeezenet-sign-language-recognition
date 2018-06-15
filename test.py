import torch
import numpy as np

inputs = np.load('dataset/X_test.npy')
labels = np.load('dataset/Y_test.npy')

inputs = torch.Tensor(inputs.reshape(-1, 1, 64, 64))
labels = torch.Tensor(labels)

net = torch.load('overnet.pt')
outputs = net(inputs)

acc = (labels.max(1)[1] == outputs.max(1)[1]).float().sum() / outputs.size(0)
print(acc)
