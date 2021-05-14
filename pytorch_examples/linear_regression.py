# First experiment with Pytorch
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
n_samples, n_features = X.shape

# Build the model
model = nn.Linear(in_features=n_features, out_features=1)

# Choose the loss function and the optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train the model

num_epochs = 100
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    # remember to empty the gradient
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch:{epoch+1},loss = {loss.item():.4f}')

#plot
predicted = model(X).detach()
plt.plot(X_numpy,y_numpy,"ro")
plt.plot(X_numpy,predicted,"b")
plt.show()

