# Advanced Variations

### Problem 1
- Create two separate tensors: X_large (1,000,000 samples) and Y_small (1000 samples). 
- Only use the first 1000 samples of $\text{X}$ for training.
- Load the first 1000 samples of $\text{X}$ and the 1000 samples of $\text{Y}$.
- Set the num_workers parameter to leverage multiprocessing (assume machine has 4 CPU cores).
- Write the code for the $\text{Dataset}$ and $\text{DataLoader}$ setup.

### Problem 2a
- Build an FNN with three hidden layers.
- After defining the model, freeze the weights of the first linear layer so they are not updated during training.
- Instead of $\text{nn.ReLU}$, define the activation function yourself as a $\text{lambda}$ function $f(z) = \frac{1}{2} (\sin(z) + 1)$ and place it between the layers.
- Write the $\text{SimpleClassifier}$ class definition and the freezing code.

### Problem 2b
- After $\text{loss.backward()}$, implement gradient clipping to ensure the total norm of the gradients doesn't exceed 1.0.
- Modify the $\text{Adam optimizer}$ initialization to include an $\text{L2}$ penalty (weight decay) of $\mathbf{0.005}$.
- After 20 epochs, verify that the $\text{first layer}$ of the model has not changed (due to the freezing in Problem 2).

# Binary Classification

Implement a complete, working PyTorch training loop for **binary classification** on a dataset of your choice (like the synthetic "Moons" data).

### Requirements

#### 1. Data Pipeline (`Dataset` & `DataLoader`)

* **Requirement 1.1: Dataset:** Create a custom $\text{Dataset}$ (or use a simple tensor/NumPy array) that correctly loads two-dimensional features ($X$) and one-dimensional labels ($Y$).
* **Requirement 1.2: Shape and Type:** Ensure the features are cast to $\text{torch.float32}$ and the labels are reshaped using **$\text{.unsqueeze(1)}$** to give them the required $(N, 1)$ shape.
* **Requirement 1.3: DataLoader:** Instantiate a $\text{DataLoader}$ with a defined **`batch_size`** and **$\text{shuffle=True}$**.

#### 2. Model Architecture (`nn.Module`)

* **Requirement 2.1: nn.Module:** Define a simple Feed-Forward Neural Network (FNN) by subclassing **$\text{nn.Module}$**.
* **Requirement 2.2: Layers:** The network must have at least one hidden $\text{nn.Linear}$ layer and one non-linear activation function (like $\text{nn.ReLU}$).
* **Requirement 2.3: Output:** The final $\text{nn.Linear}$ layer must have an **`out_features`** of **1** for binary classification (outputting a logit).

#### 3. Training Loop (`Autograd` & `optim`)

* **Requirement 3.1: Setup:** Define the $\text{criterion}$ (loss function, specifically **$\text{nn.BCEWithLogitsLoss}$**) and the **$\text{optimizer}$** (e.g., $\text{optim.Adam}$), passing the model's parameters to the optimizer.
* **Requirement 3.2: Co-location (If GPU used):** Inside the loop, move both the **$\text{model}$** and the **$\text{batch data}$** (**`batch_X, batch_Y`**) to the same device ($\text{cuda}$ or $\text{cpu}$) using $\text{.to(device)}$.
* **Requirement 3.3: Optimization Steps:** Implement the three core optimization steps in the correct order:
    1.  **Zero Gradients:** **`optimizer.zero_grad()`**
    2.  **Backward Pass:** $\text{loss.backward()}$
    3.  **Update Parameters:** $\text{optimizer.step()}$
 
















