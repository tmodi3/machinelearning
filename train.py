import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import ReceiptsPredictor

# Load data
df = pd.read_csv("data.csv")
# Assuming the data is already summed up by months and it's in a column named "receipts"
data = df["receipts"].values

# Convert to PyTorch tensors
data_tensor = torch.FloatTensor(data).view(-1, 1)

# Create DataLoader for training
train_dataset = TensorDataset(data_tensor[:-1], data_tensor[1:])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Initialize the model and optimizer
model = ReceiptsPredictor()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_function = torch.nn.MSELoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the trained model state
torch.save(model.state_dict(), "model_state.pth")
