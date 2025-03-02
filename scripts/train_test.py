import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import convnets
import cv2 as cv
import torch.onnx  # Import ONNX support from PyTorch

# Pass in command line arguments for data directory name
if len(sys.argv) != 2:
    print('Training script needs data!!!')
    sys.exit(1)  # Exit with an error code
else:
    data_datetime = sys.argv[1]

# Designate processing unit for CNN training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")


class BearCartDataset(Dataset):
    """
    Customized dataset for RGB data only.
    """
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = v2.ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load RGB image from column 0 in labels.csv
        img_name = self.img_labels.iloc[idx, 0]  # Image name from the labels.csv
        img_path = os.path.join(self.img_dir, img_name)
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Error: Could not read RGB image at {img_path}")
        image = cv.resize(image, (160, 120))  # Ensure consistent resolution for RGB image (120, 160)

        # Convert RGB image to tensor
        image_tensor = self.transform(image)

        # Steering and throttle values
        steering = self.img_labels.iloc[idx, 1].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32)

        return image_tensor.float(), steering, throttle


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    num_used_samples = 0
    ep_loss = 0.
    for b, (im, st, th) in enumerate(dataloader):
        target = torch.stack((st, th), dim=-1)
        feature, target = im.to(DEVICE), target.to(DEVICE)
        pred = model(feature)
        batch_loss = loss_fn(pred, target)  # Loss function
        optimizer.zero_grad()  # Zero previous gradient
        batch_loss.backward()  # Back propagation
        optimizer.step()  # Update params
        num_used_samples += target.shape[0]
        print(f"batch loss: {batch_loss.item()} [{num_used_samples}/{len(dataloader.dataset)}]")
        ep_loss = (ep_loss * b + batch_loss.item()) / (b + 1)
    return ep_loss


def test(dataloader, model, loss_fn):
    model.eval()
    ep_loss = 0.
    with torch.no_grad():
        for b, (im, st, th) in enumerate(dataloader):
            target = torch.stack((st, th), dim=-1)
            feature, target = im.to(DEVICE), target.to(DEVICE)
            pred = model(feature)
            batch_loss = loss_fn(pred, target)  # Loss function
            ep_loss = (ep_loss * b + batch_loss.item()) / (b + 1)
    return ep_loss


# Custom loss function (standard MSE)
def standard_loss(output, target):
    loss = ((output - target) ** 2).mean()  # Mean Squared Error loss
    return loss


# MAIN
# Create a dataset
data_dir = os.path.join(os.path.dirname(sys.path[0]), 'data', data_datetime)
annotations_file = os.path.join(data_dir, 'labels.csv')
img_dir = os.path.join(data_dir, 'rgb_images')  # RGB images directory
bearcart_dataset = BearCartDataset(annotations_file, img_dir)
print(f"data length: {len(bearcart_dataset)}")

# Create training and test dataloaders
train_size = round(len(bearcart_dataset) * 0.9)
test_size = len(bearcart_dataset) - train_size
print(f"train size: {train_size}, test size: {test_size}")
train_data, test_data = random_split(bearcart_dataset, [train_size, test_size])
train_dataloader = DataLoader(train_data, batch_size=125)
test_dataloader = DataLoader(test_data, batch_size=125)

# Create model
model = convnets.DonkeyNet().to(DEVICE)  # Adjust input channels to 3 (RGB only)
# Hyper-parameters
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = standard_loss
epochs = 20
#patience = 10 #Used for early stopping
best_loss = float('inf')  # Best loss on test data
best_counter = 0
train_losses = []
test_losses = []

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    ep_train_loss = train(train_dataloader, model, loss_fn, optimizer)
    ep_test_loss = test(test_dataloader, model, loss_fn)
    print(f"epoch {t + 1} training loss: {ep_train_loss}, testing loss: {ep_test_loss}")
    train_losses.append(ep_train_loss)
    test_losses.append(ep_test_loss)
    # Early stopping
    if ep_test_loss < best_loss:
        best_loss = ep_test_loss
        best_counter = 0  # Reset counter if validation loss improved
        try:
            if 'model_name' in locals():
                os.remove(os.path.join(data_dir, f'{model_name}.pth'))
                print(f"Last best model file has been deleted successfully.")
        except FileNotFoundError:
            print(f"File '{os.path.join(data_dir, f'{model_name}.pth')}' not found.")
        except Exception as e:
            print(f"Error occurred while deleting the file: {e}")

        model_name = f'{model._get_name()}-{t+1}ep-{lr}lr-{ep_test_loss:.4f}mse'
        torch.save(model.state_dict(), os.path.join(data_dir, f'{model_name}.pth'))
        print(f"Best model saved as '{os.path.join(data_dir, f'{model_name}.pth')}'")
    else:
        best_counter += 1
        print(f"{best_counter} epochs since best model")
        # Commenting out early stopping mechanism
        # if best_counter >= patience:
        #     print("Early stopping triggered!")
        #     break

print("Optimization Done!")

# Graph training process
pilot_title = f'{model._get_name()}-{epochs}epochs-{lr}lr'
plt.plot(range(epochs), train_losses, 'b--', label='Training')
plt.plot(range(epochs), test_losses, 'orange', label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.ylim(0.0, 0.1)
plt.yticks(np.arange(0, 0.11, 0.01))  # Set y-axis ticks from 0 to 0.1 in steps of 0.01
plt.grid(True)
plt.legend()
plt.title(pilot_title)
plt.savefig(os.path.join(data_dir, f'{model_name}.png'))

# Save the model (weights only)
torch.save(model.state_dict(), os.path.join(data_dir, f'{pilot_title}.pth'))
print("Model weights saved")

# ONNX export
dummy_input = torch.randn(1, 3, 120, 160).to(DEVICE)  # Adjust shape for 120x160 RGB
onnx_model_path = os.path.join(data_dir, f'{model_name}.onnx')
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)
print(f"Model exported to ONNX format at: {onnx_model_path}")
