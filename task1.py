import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
										   train=True,
										   transform=transforms.ToTensor(),
										   download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
										  train=False,
										  transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size,
										   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size,
										  shuffle=False)

images, labels = next(iter(train_loader))

fig, axs = plt.subplots(2, 5)
for ii in range(2):
	for jj in range(5):
		idx = 5 * ii + jj
		axs[ii, jj].imshow(images[idx].squeeze())
		axs[ii, jj].set_title(labels[idx].item())
		axs[ii, jj].axis('off')
plt.show()

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store errors for plotting
train_errors = []
test_errors = []

total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
				.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    # Calculate training error for the epoch
    train_error = running_loss / len(train_loader)
    train_errors.append(train_error)

    # Calculate test error for the epoch
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()

    test_error = test_loss / len(test_loader)
    test_errors.append(test_error)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Error: {train_error:.4f}, Test Error: {test_error:.4f}')


# Plot train and test error curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_errors, label='Training Error')
plt.plot(range(1, num_epochs+1), test_errors, label='Test Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training and Test Error')
plt.legend()
plt.grid(True)
plt.savefig('error_curves.png')
plt.close()


# Final model evaluation
with torch.no_grad():
    correct = 0
    total = 0
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Store misclassified examples
        mask = (predicted != labels)
        if mask.any():
            misclassified_idx = mask.nonzero(as_tuple=True)[0]
            misclassified_images.append(images[misclassified_idx].cpu())
            misclassified_labels.append(labels[misclassified_idx].cpu())
            misclassified_preds.append(predicted[misclassified_idx].cpu())

        final_accuracy = 100 * correct / total
    print(f'Final Test Accuracy: {final_accuracy:.2f}%')
    print(f'Final Test Error: {100 - final_accuracy:.2f}%')

    # Plot misclassified examples
num_to_show = min(10, len(misclassified_images))
if num_to_show > 0:
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    # Combine misclassified examples from all batches
    all_images = torch.cat(misclassified_images)
    all_labels = torch.cat(misclassified_labels)
    all_preds = torch.cat(misclassified_preds)
    
    for i in range(num_to_show):
        img = all_images[i].reshape(28, 28)
        true_label = all_labels[i].item()
        pred_label = all_preds[i].item()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    plt.close()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))