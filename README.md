# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS
STEP 1: Import necessary libraries and load the dataset.

STEP 2: Encode categorical variables and normalize numerical features.

STEP 3: Split the dataset into training and testing subsets.

STEP 4: Design a multi-layer neural network with appropriate activation functions.

STEP 5: Train the model using an optimizer and loss function.

STEP 6: Evaluate the model and generate a confusion matrix.

STEP 7: Use the trained model to classify new data samples.

STEP 8: Display the confusion matrix, classification report, and predictions.

## PROGRAM

### Name: YAZHINI R R
### Register Number: 212224100063

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Load Dataset
# -----------------------------
dataset = pd.read_csv("customers.csv")
print("Dataset Preview:\n", dataset.head())

# -----------------------------
# Separate features & target
# -----------------------------
X = dataset.drop("Segmentation", axis=1)
y = dataset["Segmentation"]

# -----------------------------
# Handle missing values
# -----------------------------
X["Work_Experience"].fillna(X["Work_Experience"].median(), inplace=True)
X["Family_Size"].fillna(X["Family_Size"].median(), inplace=True)

# -----------------------------
# Encode categorical columns
# -----------------------------
cat_cols = X.select_dtypes(include="object").columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# -----------------------------
# Encode target
# -----------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Convert to tensors
# -----------------------------
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# -----------------------------
# DataLoader (faster settings)
# -----------------------------
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# -----------------------------
# Neural Network Model
# -----------------------------
class PeopleClassifier(nn.Module):
    def __init__(self, input_size, classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = PeopleClassifier(X.shape[1], len(label_encoder.classes_))

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Training (reduced epochs)
# -----------------------------
epochs = 20
for epoch in range(epochs):
    for xb, yb in loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs} completed")

print("\nTraining Completed")

# -----------------------------
# Evaluation (same data)
# -----------------------------
model.eval()
with torch.no_grad():
    predictions = torch.argmax(model(X), dim=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions))

print("\nClassification Report:")
print(classification_report(
    y,
    predictions,
    target_names=label_encoder.classes_,
    zero_division=0
))

# -----------------------------
# Sample Prediction
# -----------------------------
sample = X[0].unsqueeze(0)
with torch.no_grad():
    pred = model(sample)
    result = label_encoder.inverse_transform([torch.argmax(pred).item()])

print("\nSample Prediction:", result[0])
        
```
```
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
```


## OUTPUT
## Dataset Information

<img width="853" height="332" alt="image" src="https://github.com/user-attachments/assets/ef47c4ba-2e72-486d-a656-f8fb42b0be64" />


### Confusion Matrix

<img width="730" height="496" alt="image" src="https://github.com/user-attachments/assets/a2d7b590-b787-4316-be06-cba7a93534bd" />

### Classification Report

<img width="701" height="452" alt="image" src="https://github.com/user-attachments/assets/01a5183b-4a83-4523-a26d-4eaa078e0bfb" />



### New Sample Data Prediction
<img width="807" height="263" alt="image" src="https://github.com/user-attachments/assets/388ce2ea-13d4-4ef5-828f-b89f48908432" />


## RESULT
Successfully done 
