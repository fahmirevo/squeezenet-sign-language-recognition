import torch.optim as optim
import torch.nn as nn
import torch
import models

from data import data_generator

epochs = 200
train_size = 1500
batch_size = 128
steps_per_epoch = round(train_size / batch_size)

net = models.MiniSqueezeNet(1, 10)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())

data_gen = data_generator(batch_size=batch_size)

for epoch in range(epochs):

    loss = 0

    for step in range(steps_per_epoch):
        inputs, labels = next(data_gen)
        inputs = torch.Tensor(inputs)
        labels = torch.Tensor(labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss += loss.item()
        # print(f'epoch : {epoch} step : {step} loss : {loss}')

    loss /= steps_per_epoch
    print(f'epoch : {epoch} loss : {loss}')

torch.save(net, 'net.pt')

for epoch in range(epochs):

    loss = 0

    for step in range(steps_per_epoch):
        inputs, labels = next(data_gen)
        inputs = torch.Tensor(inputs)
        labels = torch.Tensor(labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss += loss.item()
        # print(f'epoch : {epoch} step : {step} loss : {loss}')

    print(f'epoch : {epoch} loss : {loss}')

torch.save(net, 'overnet.pt')
