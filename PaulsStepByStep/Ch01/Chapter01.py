import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors

doNumpyVersion = True
doIntroToPytorchTensors = True
doPytorchManualParameterUpdate = False
doDynamicComputationGraph = False
doOptimizer = False
doLoss = False
doModel = False
doPuttingItTogether = True

showfigs = True
showfig1 = False
showfig2 = False
showfig3 = False

if doNumpyVersion:

    # Synthetic data generation
    #

    true_b = 1
    true_w = 2
    N = 100

    # Data Generation
    np.random.seed(42)
    x = np.random.rand(N, 1)
    epsilon = (.1 * np.random.randn(N, 1))
    y = true_b + true_w * x + epsilon

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

    # Gradient Descent
    #

    # Step 0 - Initializes parameters "b" and "w" randomly
    np.random.seed(42)
    b = np.random.randn(1)
    w = np.random.randn(1)

    print(b, w)

    # Step 1 - Computes our model's predicted output - forward pass
    yhat = b + w * x_train

    # Step 2 - Computing the loss
    # We are using ALL data points, so this is BATCH gradient
    # descent. How wrong is our model? That's the error!
    error = (yhat - y_train)

    # It is a regression, so it computes mean squared error (MSE)
    loss = (error ** 2).mean()

    print(loss)

    # Step 3 - Computes gradients for both "b" and "w" parameters
    b_grad = 2 * error.mean()
    w_grad = 2 * (x_train * error).mean()
    print(b_grad, w_grad)

    # Sets learning rate - this is "eta" ~ the "n" like Greek letter
    lr = 0.1
    print(b, w)

    # Step 4 - Updates parameters using gradients and 
    # the learning rate
    b = b - lr * b_grad
    w = w - lr * w_grad

    print(b, w)

    # Linear regression in Numpy
    #

    # Step 0 - Initializes parameters "b" and "w" randomly
    np.random.seed(42)
    b = np.random.randn(1)
    w = np.random.randn(1)

    print(b, w)

    # Sets learning rate - this is "eta" ~ the "n"-like Greek letter
    lr = 0.1

    # Defines number of epochs
    n_epochs = 1000

    for epoch in range(n_epochs):
        # Step 1 - Computes model's predicted output - forward pass
        yhat = b + w * x_train
        
        # Step 2 - Computes the loss
        # We are using ALL data points, so this is BATCH gradient
        # descent. How wrong is our model? That's the error!   
        error = (yhat - y_train)

        # It is a regression, so it computes mean squared error (MSE)
        loss = (error ** 2).mean()
        
        # Step 3 - Computes gradients for both "b" and "w" parameters
        b_grad = 2 * error.mean()
        w_grad = 2 * (x_train * error).mean()
        
        # Step 4 - Updates parameters using gradients and 
        # the learning rate
        b = b - lr * b_grad
        w = w - lr * w_grad
        
    print(b, w)

    # Sanity Check: do we get the same results as our
    # gradient descent?
    linr = LinearRegression()
    linr.fit(x_train, y_train)

    print(linr.intercept_, linr.coef_[0])

    if showfigs and showfig2:
        fig, ax = plt.subplots()
        ax.scatter(x_train, y_train, color='blue', label="Actual")
        ax.scatter(x_train, yhat, color='red', label="Model")
        ax.set_title('Model and Actual')
        ax.set_xlabel('Years of Experience')
        ax.set_ylabel('Salary')
        ax.legend()
        plt.show()


if doIntroToPytorchTensors:
    # Pytorch Tensors
    #

    scalar = torch.tensor(3.14159)
    vector = torch.tensor([1, 2, 3])
    matrix = torch.ones((2, 3), dtype=torch.float)
    tensor = torch.randn((2, 3, 4), dtype=torch.float)

    print(scalar)
    print(vector)
    print(matrix)
    print(tensor)

    print(tensor.size(), tensor.shape)
    print(scalar.size(), scalar.shape)

    # We get a tensor with a different shape but it still is
    # the SAME tensor
    same_matrix = matrix.view(1, 6)
    # If we change one of its elements...
    same_matrix[0, 1] = 2.
    # It changes both variables: matrix and same_matrix
    print(matrix)
    print(same_matrix)

    # We can use "new_tensor" method to REALLY copy it into a new one
    different_matrix = matrix.new_tensor(matrix.view(1, 6))
    # Now, if we change one of its elements...
    different_matrix[0, 1] = 3.
    # The original tensor (matrix) is left untouched!
    # But we get a "warning" from PyTorch telling us 
    # to use "clone()" instead!
    print(matrix)
    print(different_matrix)

    # Lets follow PyTorch's suggestion and use "clone" method
    another_matrix = matrix.view(1, 6).clone().detach()
    # Again, if we change one of its elements...
    another_matrix[0, 1] = 4.
    # The original tensor (matrix) is left untouched!
    print(matrix)
    print(another_matrix)

    # Loading Data
    #

    x_train_tensor = torch.as_tensor(x_train)
    print(x_train.dtype, x_train_tensor.dtype)

    float_tensor = x_train_tensor.float()
    print(float_tensor.dtype)

    dummy_array = np.array([1, 2, 3])
    dummy_tensor = torch.as_tensor(dummy_array)
    # Modifies the numpy array
    dummy_array[1] = 0
    # Tensor gets modified too...
    print(dummy_tensor)
    print(dummy_tensor.numpy())

    # Devices and CUDA
    #

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_cudas = torch.cuda.device_count()
    for i in range(n_cudas):
        print(torch.cuda.get_device_name(i))
    gpu_tensor = torch.as_tensor(x_train).to(device)
    print(gpu_tensor[0])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Our data was in Numpy arrays, but we need to transform them 
    # into PyTorch's Tensors and then we send them to the 
    # chosen device
    x_train_tensor = torch.as_tensor(x_train).float().to(device)
    y_train_tensor = torch.as_tensor(y_train).float().to(device)

    # Here we can see the difference - notice that .type() is more
    # useful since it also tells us WHERE the tensor is (device)
    print(type(x_train), type(x_train_tensor), x_train_tensor.type())

    # The line below doesn't work because you can't go straight from a CUDA tensor to Numpy, you 
    # need to convert back to a CPU tensor first:
    #back_to_numpy = x_train_tensor.numpy()
    back_to_numpy = x_train_tensor.cpu().numpy()

if doPytorchManualParameterUpdate:
    # Creating parameters
    #

    # FIRST
    # Initializes parameters "b" and "w" randomly, ALMOST as we
    # did in Numpy since we want to apply gradient descent on
    # these parameters we need to set REQUIRES_GRAD = TRUE
    torch.manual_seed(42)
    b = torch.randn(1, requires_grad=True, dtype=torch.float)
    w = torch.randn(1, requires_grad=True, dtype=torch.float)
    print(b, w)

    # SECOND
    # But what if we want to run it on a GPU? We could just
    # send them to device, right?
    torch.manual_seed(42)
    b = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
    w = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
    print(b, w)
    # Sorry, but NO! The to(device) "shadows" the gradient...

    # THIRD
    # We can either create regular tensors and send them to
    # the device (as we did with our data)
    torch.manual_seed(42)
    b = torch.randn(1, dtype=torch.float).to(device)
    w = torch.randn(1, dtype=torch.float).to(device)
    # and THEN set them as requiring gradients...
    b.requires_grad_()
    w.requires_grad_()
    print(b, w)

    # FINAL
    # We can specify the device at the moment of creation
    # RECOMMENDED!

    # Step 0 - Initializes parameters "b" and "w" randomly
    torch.manual_seed(42)
    b = torch.randn(1, requires_grad=True, \
                    dtype=torch.float, device=device)
    w = torch.randn(1, requires_grad=True, \
                    dtype=torch.float, device=device)
    print(b, w)

    # Autograd
    #

    # Step 1 - Computes our model's predicted output - forward pass
    yhat = b + w * x_train_tensor

    # Step 2 - Computes the loss
    # We are using ALL data points, so this is BATCH gradient descent
    # How wrong is our model? That's the error! 
    error = (yhat - y_train_tensor)
    # It is a regression, so it computes mean squared error (MSE)
    loss = (error ** 2).mean()

    # Step 3 - Computes gradients for both "b" and "w" parameters
    # No more manual computation of gradients! 
    # b_grad = 2 * error.mean()
    # w_grad = 2 * (x_tensor * error).mean()
    loss.backward()

    print(error.requires_grad, yhat.requires_grad, b.requires_grad, w.requires_grad)
    print(y_train_tensor.requires_grad, x_train_tensor.requires_grad)
    print(b.grad, w.grad)

    # Do the previous 3 steps again to demonstrate gradient accumulation.
    #

    # Step 1 - Computes our model's predicted output - forward pass
    yhat = b + w * x_train_tensor

    # Step 2 - Computes the loss
    # We are using ALL data points, so this is BATCH gradient descent
    # How wrong is our model? That's the error! 
    error = (yhat - y_train_tensor)
    # It is a regression, so it computes mean squared error (MSE)
    loss = (error ** 2).mean()

    # Step 3 - Computes gradients for both "b" and "w" parameters
    # No more manual computation of gradients! 
    # b_grad = 2 * error.mean()
    # w_grad = 2 * (x_tensor * error).mean()
    loss.backward()

    print(error.requires_grad, yhat.requires_grad, b.requires_grad, w.requires_grad)
    print(y_train_tensor.requires_grad, x_train_tensor.requires_grad)
    print(b.grad, w.grad)

    # Zero the gradient which gets accumulated.

    # This code will be placed *after* Step 4
    # (updating the parameters)
    b.grad.zero_(), w.grad.zero_()
    print(b.grad, w.grad)

    # Updating Parameters
    #

    # Sets learning rate - this is "eta" ~ the "n"-like Greek letter
    lr = 0.1

    # Step 0 - Initializes parameters "b" and "w" randomly
    torch.manual_seed(42)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

    # Defines number of epochs
    n_epochs = 1000

    for epoch in range(n_epochs):
        # Step 1 - Computes model's predicted output - forward pass
        yhat = b + w * x_train_tensor
        
        # Step 2 - Computes the loss
        # We are using ALL data points, so this is BATCH gradient
        # descent. How wrong is our model? That's the error!
        error = (yhat - y_train_tensor)

        # It is a regression, so it computes mean squared error (MSE)
        loss = (error ** 2).mean()

        # Step 3 - Computes gradients for both "b" and "w" parameters
        # No more manual computation of gradients! 
        # b_grad = 2 * error.mean()
        # w_grad = 2 * (x_tensor * error).mean()   
        # We just tell PyTorch to work its way BACKWARDS 
        # from the specified loss!
        loss.backward()
        
        # Step 4 - Updates parameters using gradients and 
        # the learning rate. But not so fast...
        # FIRST ATTEMPT - just using the same code as before
        # AttributeError: 'NoneType' object has no attribute 'zero_'
        # b = b - lr * b.grad
        # w = w - lr * w.grad
        # print(b)

        # SECOND ATTEMPT - using in-place Python assigment
        # RuntimeError: a leaf Variable that requires grad
        # has been used in an in-place operation.
        # b -= lr * b.grad
        # w -= lr * w.grad        
        
        # THIRD ATTEMPT - NO_GRAD for the win!
        # We need to use NO_GRAD to keep the update out of
        # the gradient computation. Why is that? It boils 
        # down to the DYNAMIC GRAPH that PyTorch uses...
        with torch.no_grad():
            b -= lr * b.grad
            w -= lr * w.grad

        #print(b, w)
        
        # PyTorch is "clingy" to its computed gradients, we
        # need to tell it to let it go...
        b.grad.zero_()
        w.grad.zero_()
        
    print(b, w)

if doDynamicComputationGraph:
    # Dynamic Computation Graph
    #

    # Step 0 - Initializes parameters "b" and "w" randomly
    torch.manual_seed(42)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

    # Step 1 - Computes our model's predicted output - forward pass
    yhat = b + w * x_train_tensor

    # Step 2 - Computes the loss
    # We are using ALL data points, so this is BATCH gradient
    # descent. How wrong is our model? That's the error! 
    error = (yhat - y_train_tensor)
    # It is a regression, so it computes mean squared error (MSE)
    loss = (error ** 2).mean()

    if showfig3 and showfigs:
        # We can try plotting the graph for any python variable: 
        # yhat, error, loss...
        dot = make_dot(yhat)
        dot.render("Model", format="pdf", view=True)

if doOptimizer:

    # Optimizer
    #

    # Sets learning rate - this is "eta" ~ the "n"-like Greek letter
    lr = 0.1

    # Step 0 - Initializes parameters "b" and "w" randomly
    torch.manual_seed(42)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

    # Defines a SGD optimizer to update the parameters
    optimizer = optim.SGD([b, w], lr=lr)

    # Defines number of epochs
    n_epochs = 1000

    for epoch in range(n_epochs):
        # Step 1 - Computes model's predicted output - forward pass
        yhat = b + w * x_train_tensor
        
        # Step 2 - Computes the loss
        # We are using ALL data points, so this is BATCH gradient 
        # descent. How wrong is our model? That's the error! 
        error = (yhat - y_train_tensor)

        # It is a regression, so it computes mean squared error (MSE)
        loss = (error ** 2).mean()

        # Step 3 - Computes gradients for both "b" and "w" parameters
        loss.backward()
        
        # Step 4 - Updates parameters using gradients and 
        # the learning rate. No more manual update!
        # with torch.no_grad():
        #     b -= lr * b.grad
        #     w -= lr * w.grad
        optimizer.step()
        
        # No more telling Pytorch to let gradients go!
        # b.grad.zero_()
        # w.grad.zero_()
        optimizer.zero_grad()

        #print(b, w)
        
    print(b, w)

if doLoss:
    # Loss
    #

    # Set up device for this section.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')
    print(loss_fn)

    # This is a random example to illustrate the loss function
    predictions = torch.tensor([0.5, 1.0])
    labels = torch.tensor([2.0, 1.3])
    print(loss_fn(predictions, labels))

    # Sets learning rate - this is "eta" ~ the "n"-like
    # Greek letter
    lr = 0.1

    # Step 0 - Initializes parameters "b" and "w" randomly
    torch.manual_seed(42)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

    # Defines a SGD optimizer to update the parameters
    optimizer = optim.SGD([b, w], lr=lr)

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # Defines number of epochs
    n_epochs = 1000

    for epoch in range(n_epochs):
        # Step 1 - Computes model's predicted output - forward pass
        yhat = b + w * x_train_tensor
        
        # Step 2 - Computes the loss
        # No more manual loss!
        # error = (yhat - y_train_tensor)
        # loss = (error ** 2).mean()
        loss = loss_fn(yhat, y_train_tensor)

        # Step 3 - Computes gradients for both "b" and "w" parameters
        loss.backward()
        
        # Step 4 - Updates parameters using gradients and
        # the learning rate
        optimizer.step()
        optimizer.zero_grad()
        
    print(b, w)

    print(loss)

    # To get any of the tensors that have gradients computed, we need to DETACH prior to moving back to the cpu.
    print(loss.detach().cpu().numpy())

    # Easier way to do the above
    print(loss.item(), loss.tolist())



if doModel:
    # Model
    #

    # Define a PyTorch model class.
    class ManualLinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            # To make "b" and "w" real parameters of the model,
            # we need to wrap them with nn.Parameter
            self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
            self.w = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
            
        def forward(self, x):
            # Computes the outputs / predictions
            return self.b + self.w * x
        
    # Registering the parameters allows you to list them easily, as below.
    torch.manual_seed(42)

    # Creates a "dummy" instance of our ManualLinearRegression model
    dummy = ManualLinearRegression()
    print(list(dummy.parameters()))
    print(dummy.state_dict())

    # Now train the model.

    # Sets learning rate - this is "eta" ~ the "n"-like
    # Greek letter
    lr = 0.1

    # Step 0 - Initializes parameters "b" and "w" randomly
    torch.manual_seed(42)
    # Now we can create a model and send it at once to the device
    model = ManualLinearRegression().to(device)

    # Defines a SGD optimizer to update the parameters 
    # (now retrieved directly from the model)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # Defines number of epochs
    n_epochs = 1000

    for epoch in range(n_epochs):
        model.train() # What is this?!?

        # Step 1 - Computes model's predicted output - forward pass
        # No more manual prediction!
        yhat = model(x_train_tensor)
        
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y_train_tensor)

        # Step 3 - Computes gradients for both "b" and "w" parameters
        loss.backward()
        
        # Step 4 - Updates parameters using gradients and
        # the learning rate
        optimizer.step()
        optimizer.zero_grad()
        
    # We can also inspect its parameters using its state_dict
    print(model.state_dict())

    # PyTorch Sequential Models

    torch.manual_seed(42)

    # Alternatively, you can use a Sequential model
    model = nn.Sequential(nn.Linear(1, 1)).to(device)

    print(model.state_dict())

i = 0
