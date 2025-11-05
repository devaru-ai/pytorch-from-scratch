### Goal

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
 
















