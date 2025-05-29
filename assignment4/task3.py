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
    
    def forward_with_features(self, x):
        """Forward pass that returns intermediate features for analysis."""
        # Encoder path
        x1 = F.relu(self.conv1(x))  # z(1): 6 x 28 x 28
        x1_pooled, indices1 = self.pool(x1)  # 6 x 14 x 14
        
        x2 = F.relu(self.conv2(x1_pooled))  # z(2): 16 x 10 x 10
        x2_pooled, indices2 = self.pool(x2)  # 16 x 5 x 5
        
        # Classification path
        x_flat = torch.flatten(x2_pooled, 1)
        x_fc = F.relu(self.fc1(x_flat))
        x_fc = F.relu(self.fc2(x_fc))
        class_output = self.fc3(x_fc)
        
        return class_output, x1_pooled, x2_pooled, indices1, indices2
    
    def reconstruct_from_features(self, x2_pooled, x1_pooled, indices1, indices2):
        """Reconstruct image from given feature representations."""
        # Decoder path
        x_dec = self.unpool(x2_pooled, indices2)  # 16 x 10 x 10
        x_dec = F.relu(self.deconv1(x_dec))  # 6 x 14 x 14
        x_dec = self.unpool(x_dec, indices1)  # 6 x 28 x 28
        reconstructed = self.deconv2(x_dec)  # 3 x 32 x 32
        
        return reconstructed

def analyze_single_channel_z1(net, image, channel_idx, z1_features, z2_features, indices1, indices2, classes, label):
    """Analyze single channel in z(1) by zeroing out other channels."""
    # Create modified z1 features with only one channel active
    z1_modified = torch.zeros_like(z1_features)
    z1_modified[0, channel_idx] = z1_features[0, channel_idx]
    
    # Reconstruct from modified features
    with torch.no_grad():
        reconstructed = net.reconstruct_from_features(z2_features, z1_modified, indices1, indices2)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original image
    orig_img = image[0] / 2 + 0.5  # unnormalize
    axes[0].imshow(np.transpose(orig_img.numpy(), (1, 2, 0)))
    axes[0].set_title(f'Original: {classes[label[0]]}')
    axes[0].axis('off')
    
    # Reconstructed from single channel
    recon_img = reconstructed[0] / 2 + 0.5  # unnormalize
    recon_img = torch.clamp(recon_img, 0, 1)
    axes[1].imshow(np.transpose(recon_img.numpy(), (1, 2, 0)))
    axes[1].set_title(f'Reconstruction from z(1) channel {channel_idx}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_single_channel_z2(net, image, channel_idx, z1_features, z2_features, indices1, indices2, classes, label):
    """Analyze single channel in z(2) by zeroing out other channels."""
    # Create modified z2 features with only one channel active
    z2_modified = torch.zeros_like(z2_features)
    z2_modified[0, channel_idx] = z2_features[0, channel_idx]
    
    # Reconstruct from modified features
    with torch.no_grad():
        reconstructed = net.reconstruct_from_features(z2_modified, z1_features, indices1, indices2)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original image
    orig_img = image[0] / 2 + 0.5  # unnormalize
    axes[0].imshow(np.transpose(orig_img.numpy(), (1, 2, 0)))
    axes[0].set_title(f'Original: {classes[label[0]]}')
    axes[0].axis('off')
    
    # Reconstructed from single channel
    recon_img = reconstructed[0] / 2 + 0.5  # unnormalize
    recon_img = torch.clamp(recon_img, 0, 1)
    axes[1].imshow(np.transpose(recon_img.numpy(), (1, 2, 0)))
    axes[1].set_title(f'Reconstruction from z(2) channel {channel_idx}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_latent_representations(net_class, trainloader, testloader, classes, model_path=PATH):
    """Analyze latent representations by visualizing individual channel contributions."""
    
    # Load trained model
    net = net_class()
    net.load_state_dict(torch.load(model_path, weights_only=True))
    net.eval()
    
    print("=== Latent Representation Analysis ===")
    
    # Get one image from train set
    train_iter = iter(trainloader)
    train_image, train_label = next(train_iter)
    train_image = train_image[:1]  # Take only first image
    train_label = train_label[:1]
    
    # Get one image from test set
    test_iter = iter(testloader)
    test_image, test_label = next(test_iter)
    test_image = test_image[:1]  # Take only first image
    test_label = test_label[:1]
    
    # Process both images
    for img_type, image, label in [("Train", train_image, train_label), ("Test", test_image, test_label)]:
        print(f"\n=== Analysis for {img_type} Image ===")
        
        with torch.no_grad():
            # Get features and indices
            class_output, z1_features, z2_features, indices1, indices2 = net.forward_with_features(image)
            
            # Get full reconstruction for comparison
            full_reconstruction = net.reconstruct_from_features(z2_features, z1_features, indices1, indices2)
        
        print(f"Image class: {classes[label[0]]}")
        print(f"z(1) shape: {z1_features.shape}")  # Should be [1, 6, 14, 14]
        print(f"z(2) shape: {z2_features.shape}")  # Should be [1, 16, 5, 5]
        
        # Show original and full reconstruction
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Original
        orig_img = image[0] / 2 + 0.5
        axes[0].imshow(np.transpose(orig_img.numpy(), (1, 2, 0)))
        axes[0].set_title(f'{img_type} Original: {classes[label[0]]}')
        axes[0].axis('off')
        
        # Full reconstruction
        full_recon_img = full_reconstruction[0] / 2 + 0.5
        full_recon_img = torch.clamp(full_recon_img, 0, 1)
        axes[1].imshow(np.transpose(full_recon_img.numpy(), (1, 2, 0)))
        axes[1].set_title('Full Reconstruction')
        axes[1].axis('off')
        
        # Difference
        diff = torch.abs(orig_img - full_recon_img)
        axes[2].imshow(np.transpose(diff.numpy(), (1, 2, 0)))
        axes[2].set_title('Reconstruction Error')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Analyze all channels in z(1) - first convolutional layer (6 channels)
        print(f"\n--- Analyzing z(1) channels for {img_type} image ---")
        for channel in range(6):
            print(f"Channel {channel}:")
            analyze_single_channel_z1(net, image, channel, z1_features, z2_features, 
                                     indices1, indices2, classes, label)
        
        # Analyze 3 channels in z(2) - second convolutional layer (showing 3 out of 16)
        print(f"\n--- Analyzing z(2) channels for {img_type} image (showing 3 representative channels) ---")
        z2_channels_to_show = [0, 8, 15]  # Show first, middle, and last channels
        for channel in z2_channels_to_show:
            print(f"Channel {channel}:")
            analyze_single_channel_z2(net, image, channel, z1_features, z2_features, 
                                     indices1, indices2, classes, label)

if __name__ == '__main__':
    batch_size = 4
    
    # Load data
    trainloader, testloader, classes = load_cifar10_data(batch_size=batch_size, num_workers=2)

    print("=== Task 3: Latent Representation Analysis ===")
    print("This task analyzes what individual feature channels contribute to image reconstruction.")
    print("We will show reconstructions when only one channel is active in z(1) and z(2).")
    
    # Perform latent representation analysis
    analyze_latent_representations(DeconvNet, trainloader, testloader, classes, model_path=PATH)


    #TODO: 
    # 1. What about the second Relu in the deconvnet?
    # 2. The recostruction function seems incorrect. (x1_pooled is not used).
    # 3. We need to understand what is the reconstruction omri is talking about.
