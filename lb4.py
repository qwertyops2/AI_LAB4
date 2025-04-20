import torch 
import torch.nn as nn 
import pandas as pd
import numpy as np

df = pd.read_csv('dataset_simple.csv')
X = torch.Tensor(df.iloc[:, [0, 1]].values)
y = torch.Tensor(df.iloc[:, 2].values)

X_np = df.iloc[:, [0, 1]].values
X_np = (X_np - X_np.mean(axis=0)) / X_np.std(axis=0)
X = torch.Tensor(X_np)
y = torch.Tensor(df.iloc[:, 2].values)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

input_size = X.shape[1]
hidden_size = 16
model = Net(input_size, hidden_size)

lossFn = nn.L1Loss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 3000
for i in range(epochs):
    pred = model.forward(X)
    loss = lossFn(pred.squeeze(), y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 300 == 0:
        print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

with torch.no_grad():
    pred= (model(X) > 0.5).float()
    accuracy = (pred.squeeze() == y).float().mean()
    print(f'\nТочность модели: {accuracy.item() * 100:.2f}%')
    for i in range(15):
        print(f"Реальное: {y[i].item():.0f} | Предсказано: {pred[i].item():.0f} | {'Совпало' if y[i] == pred[i] else 'Ошибка'}")
    
