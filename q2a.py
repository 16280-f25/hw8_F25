import torch

# TODO 1: Create a tensor x with value 2.0 and set requires_grad=True. This means PyTorch will track gradients of x.
# Read docs here: https://pytorch.org/docs/stable/generated/torch.tensor.html
x = 

# TODO 2: Create a tensor y with value 3.0 and set requires_grad=True. This means PyTorch will track gradients of y.
y = 

# TODO 3: Compute z = x^2 + y^2. Hint: use ** for exponentiation.
z = 

# TODO 4: Compute a value for loss as z squared. Hint: use ** for exponentiation.
# For clarity, this is not a loss function, it's just a value we're computing.
loss = 

# Print the loss value
print(f"Loss: {loss.item()}")

# TODO 5: Call backward() on loss to compute gradients
# Gradients should be: dloss/dx = 4x * z, dloss/dy = 4y * z
loss.backward()

# w.r.t. means with respect to - will come in handy in the future :)
print(f"Gradient of loss w.r.t x: {x.grad}")
print(f"Gradient of loss w.r.t y: {y.grad}")
