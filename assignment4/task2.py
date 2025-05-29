import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PATH = './cifar_deconv_net.pth'

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_cifar10_data(batch_size=4, num_workers=2):
    """Load CIFAR10 dataset and create data loaders."""
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

def train_deconv_network(net, trainloader, epochs=10, lr=0.001, momentum=0.9, lambda_rec=1.0, save_path=PATH):
    """Train the deconvolutional neural network with combined loss."""
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_ce_loss = 0.0
        running_rec_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            class_outputs, reconstructed = net(inputs)
            
            # compute losses
            ce_loss = criterion_ce(class_outputs, labels)
            rec_loss = criterion_mse(reconstructed, inputs)
            
            # combined loss
            total_loss = ce_loss + lambda_rec * rec_loss
            
            # backward + optimize
            total_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += total_loss.item()
            running_ce_loss += ce_loss.item()
            running_rec_loss += rec_loss.item()
            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] total loss: {running_loss / 2000:.3f}, '
                      f'ce loss: {running_ce_loss / 2000:.3f}, rec loss: {running_rec_loss / 2000:.3f}')
                running_loss = 0.0
                running_ce_loss = 0.0
                running_rec_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), save_path)

def evaluate_deconv_network(net_class, testloader, classes, model_path=PATH):
    """Evaluate the trained deconvolutional neural network."""
    net = net_class()
    net.load_state_dict(torch.load(model_path, weights_only=True))
    net.eval()

    # Show some reconstructed examples
    print("=== Reconstruction Examples ===")
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        class_outputs, reconstructed = net(images)
        
    # Show original vs reconstructed images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        # Original images
        orig_img = images[i] / 2 + 0.5  # unnormalize
        axes[0, i].imshow(np.transpose(orig_img.numpy(), (1, 2, 0)))
        axes[0, i].set_title(f'Original: {classes[labels[i]]}')
        axes[0, i].axis('off')
        
        # Reconstructed images
        recon_img = reconstructed[i] / 2 + 0.5  # unnormalize
        recon_img = torch.clamp(recon_img, 0, 1)  # clamp to valid range
        axes[1, i].imshow(np.transpose(recon_img.numpy(), (1, 2, 0)))
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    
    _, predicted = torch.max(class_outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

    # Evaluate classification accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            class_outputs, _ = net(images)
            _, predicted = torch.max(class_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Classification Accuracy on test set: {100 * correct / total:.2f}%')

    # Per-class accuracy
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            class_outputs, _ = net(images)
            _, predictions = torch.max(class_outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print("\n=== Per-class Accuracy ===")
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

class DeconvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder (same as original network)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Classification layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # Decoder layers
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(16, 6, 5)
        self.deconv2 = nn.ConvTranspose2d(6, 3, 5)

    def forward(self, x):
        # Encoder path
        x1 = F.relu(self.conv1(x))  # 6 x 28 x 28
        x1_pooled, indices1 = self.pool(x1)  # 6 x 14 x 14
        
        x2 = F.relu(self.conv2(x1_pooled))  # 16 x 10 x 10
        x2_pooled, indices2 = self.pool(x2)  # 16 x 5 x 5
        
        # Classification path
        x_flat = torch.flatten(x2_pooled, 1)
        x_fc = F.relu(self.fc1(x_flat))
        x_fc = F.relu(self.fc2(x_fc))
        class_output = self.fc3(x_fc)
        
        # Decoder path
        x_dec = self.unpool(x2_pooled, indices2)  # 16 x 10 x 10
        x_dec = F.relu(self.deconv1(x_dec))  # 6 x 14 x 14
        x_dec = self.unpool(x_dec, indices1)  # 6 x 28 x 28
        reconstructed = self.deconv2(x_dec)  # 3 x 32 x 32
        
        return class_output, reconstructed

if __name__ == '__main__':
    batch_size = 4
    
    # Load data
    trainloader, testloader, classes = load_cifar10_data(batch_size=batch_size, num_workers=2)

    # Create deconvolutional network
    net = DeconvNet()
    
    print("=== Training Deconvolutional Network ===")
    # Train the network with reconstruction loss
    # train_deconv_network(net, trainloader, epochs=10, lr=0.001, momentum=0.9, lambda_rec=1.0, save_path=PATH)

    print("\n=== Evaluating Deconvolutional Network ===")
    # Evaluate the network
    evaluate_deconv_network(DeconvNet, testloader, classes, model_path=PATH)