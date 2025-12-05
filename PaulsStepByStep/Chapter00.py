import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Display control
#
showfig1 = False
showfig2 = False
showfig3 = False
showfigs = True

# Synthetic Data Generation
#

true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon

# Train-Validation split
#

# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Display the data.
if showfigs and showfig1:
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))
    ax1.scatter(x_train,y_train, color='blue')
    ax1.set_title('Generated Data - Train')
    ax2.scatter(x_val,y_val,color='red')
    ax2.set_title('Generated Data = Validation')
    plt.show()

# Step 0 Random Initialization
#

np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

print(b, w)

# Step 1 - Compute the Model's Predications
#

yhat = b + w * x_train

# Display the predicted and actual data.
if showfigs and showfig2:
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train, color='blue', label="Actual")
    ax.scatter(x_train, yhat, color='red', label="Model")
    ax.set_title('Model and Actual')
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('Salary')
    ax.legend()
    plt.show()

# Step 2 - Compute the loss
#

# We are using ALL data points, so this is BATCH gradient
# descent. How wrong is our model? That's the error!
error = (yhat - y_train)

# It is a regression, so it computes mean squared error (MSE)
loss = (error ** 2).mean()
print(loss)

# Loss Surface

# we have to split the ranges in 100 evenly spaced intervals each
b_range = np.linspace(true_b - 3, true_b + 3, 101)
w_range = np.linspace(true_w - 3, true_w + 3, 101)
# meshgrid is a handy function that generates a grid of b and w
# values for all combinations
bs, ws = np.meshgrid(b_range, w_range)
# Apply the model function to all 
all_predictions = np.apply_along_axis(
    func1d=lambda x: bs + ws * x, 
    axis=1, 
    arr=x_train
)

all_labels = y_train.reshape(-1,1,1)
all_errors = all_predictions - all_labels
all_losses = (all_errors**2).mean(axis=0)

# Plot the loss surface
#
if showfigs and showfig3:
    fig = plt.figure(figsize=(8,6))
    subfigs = fig.subfigures(1,1)
    ax = subfigs.add_subplot(111, projection='3d')
    surf = ax.plot_surface(bs, ws, all_losses, cmap='viridis', edgecolor='none')
    ax.set_xlabel('b')
    ax.set_ylabel('w')
    ax.set_zlabel('Loss')
    ax.set_title('W, B versus Loss')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()

# Step 3 - Compute the Gradients
#

b_grad = 2 * error.mean()
w_grad = 2 * (x_train * error).mean()
print(b_grad, w_grad)

# Step 4 Update the Parameters
#

# Sets learning rate - this is "eta" ~ the "n" like Greek letter
lr = 0.1
print(b, w)

# Step 4 - Updates parameters using gradients and the 
# learning rate
b = b - lr * b_grad
w = w - lr * w_grad

print(b, w)

i = 0
