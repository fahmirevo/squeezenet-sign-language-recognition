import torch.optim as optim
import torch.nn as nn
import torch
import models

from data import data_generator

epochs = 400
train_size = 1500
batch_size = 128
steps_per_epoch = round(train_size / batch_size)

net = models.MiniSqueezeNet(1, 10)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())

data_gen = data_generator(batch_size=batch_size)

for epoch in range(epochs):

    for step in range(steps_per_epoch):
        inputs, labels = next(data_gen)
        inputs = torch.Tensor(inputs)
        labels = torch.Tensor(labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss = loss.item()

        acc = (labels.max(1)[1] == outputs.max(1)[1]).float().sum() / outputs.size(0)
        print(f'epoch : {epoch} step : {step} loss : {loss} acc : {acc}')

torch.save(net, 'net.pt')
