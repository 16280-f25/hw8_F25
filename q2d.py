# Carnegie Mellon University 16280 Machine Learning HW F25
# Created by Julius Arolovitch, jarolovi@andrew.cmu.edu
# Last modified July 2025

import math
import torch

import matplotlib
matplotlib.use("Agg") # This setting is necessary for running on remote machines
import matplotlib.pyplot as plt

# TODO: Copy from Q2b
x = 
y = 

# TODO: Define the model according to the instructions in the handout. 
model = torch.nn.Sequential(
    ...
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # This is the optimizer PyTorch will use to update the model's parameters

num_epochs = 5000

loss_history = [] # Define a list to store the loss values over iterations

# TODO: Define the loss function using PyTorch's MSE Loss
# Docs: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
loss_fn = ...

for epoch in range(num_epochs):
    y_pred = model(x.unsqueeze(1)).squeeze()

    # TODO: Compute the loss using torch.nn.MSELoss()
    # torch.nn.MSELoss() takes in two arguments: the network's prediction and the target value
    # Example: torch.nn.MSELoss(torch.tensor(1), torch.tensor(2)) = 1
    # Hint: Recall that you don't have to convert the prediction and target to tensors because they are already tensors. 
    
    loss = ...

    if epoch % 100 == 99:
        print(f"Iteration {epoch} loss: {loss.item()}")

    # Backpropagation
    optimizer.zero_grad() # Zero the gradients
    loss.backward() # Compute the gradients
    optimizer.step() # Update the model's parameters

    # Record loss
    loss_history.append(loss.item())

x_plot = torch.linspace(-10 * math.pi, 10 * math.pi, 4000)
y_plot = torch.sin(x_plot)
y_pred_plot = model(x_plot.unsqueeze(1)).squeeze().detach()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(x_plot.numpy(), y_plot.numpy(), label='sin(x)')
ax1.plot(x_plot.numpy(), y_pred_plot.numpy(), label='Neural network')
ax1.set_title('Q3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_ylim(-2, 2)

# Set x-axis ticks at multiples of π
xticks = [i * math.pi for i in range(-10, 11, 2)]  # every 2π from -10π to 10π
xlabels = [f"{i}π" if i != 0 else "0" for i in range(-10, 11, 2)]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xlabels)

ax1.legend()

ax2.plot(loss_history)
ax2.set_title('Training Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE')

plt.tight_layout()

plt.savefig('output_3d.png')
plt.close()
