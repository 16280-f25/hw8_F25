# Carnegie Mellon University 16280 Machine Learning HW F25
# Created by Julius Arolovitch, jarolovi@andrew.cmu.edu
# Last modified July 2025

import math
import torch

import matplotlib
matplotlib.use("Agg") # This setting is necessary for running on remote machines
import matplotlib.pyplot as plt

############ START STUDENT CODE ############

# TODO: Copy from Q3a
x = 

# TODO: Copy from Q3a
y = 

# TODO: Randomly initialize the weights using torch.randn. Don't pass a dtype. Set requires_grad=True. 
# Read docs here: https://pytorch.org/docs/stable/generated/torch.randn.html
a = torch.
b = 
c = 
d = 

learning_rate = 1e-6 # Do not change

# Run 2000 iterations of gradient descent 
for t in range(2000):
    # TODO: Copy from Q3a
    y_pred = 

    # TODO: Copy from Q3a
    loss = 
    if t % 100 == 99:
        print(f"Iteration {t} loss: {loss}") # Print loss every 100 iterations

    # Backpropagate to compute gradients w.r.t. the parameters. Woohoo! No more manual gradients :)
    loss.backward()

    with torch.no_grad():
        # TODO: Copy from Q3a
        a -= 
        b -= 
        c -= 
        d -= 

        # TODO: Zero all gradients after updating to prevent them from accumulating over iterations
        a.grad.zero_()
        #...

print(f'Result: y = {a.item()} x^3 + {b.item()} x^2 + {c.item()} x + {d.item()}')

############ END STUDENT CODE ############

x_plot = torch.linspace(-2 * math.pi, 2 * math.pi, 4000)
y_plot = torch.sin(x_plot)
y_pred_plot = a * x_plot ** 3 + b * x_plot ** 2 + c * x_plot + d

plt.figure(figsize=(10, 4))
plt.plot(x_plot.numpy(), y_plot.numpy(), label='sin(x)')
plt.plot(x_plot.numpy(), y_pred_plot.detach().numpy(), label='Learned polynomial')
plt.title('Q3c')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.ylim(-2, 2)

plt.tight_layout()

plt.savefig('output_3c.png')
plt.close()
