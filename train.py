import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from model import UNet, reconstruction_loss, consistency_loss, hazy_reconstruction_loss, perceptual_loss, total_variation_loss
from dataset import HazeLowlightDataset_1
from torchvision.models import vgg16

# Initialize dataset
dataset = HazeLowlightDataset_1(
    hazy_dir="data_ssl2/data_train",
    low_light_dir="data_ssl2/data_train_lowlight",
    transform=None,
    resize_to=(256, 256)
)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNet().to(device)
vgg = vgg16(pretrained=True).features[:9].to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

optimizer = optim.Adam(unet.parameters(), lr=1e-4)

# Training Loop
for epoch in range(1, 201):  # Example: 100 epochs
    print(f"Starting Epoch {epoch}")
    unet.train()
    for batch_idx, batch in enumerate(train_dataloader):
        train_images, low_light_images, t_maps, a_maps, l_maps = [x.to(device) for x in batch]

        # Forward pass
        dehazed_images = unet(train_images)

        # Loss computation
        loss_reconstruction = reconstruction_loss(dehazed_images, low_light_images)
        loss_consistency = consistency_loss(t_maps, l_maps)
        loss_hazy_reconstruction = hazy_reconstruction_loss(train_images, dehazed_images, t_maps, a_maps)
        loss_perceptual = perceptual_loss(vgg, dehazed_images, low_light_images)
        loss_TV = total_variation_loss(dehazed_images)

        total_loss = (
            1.0 * loss_reconstruction +
            0.5 * loss_consistency +
            1.0 * loss_hazy_reconstruction +
            0.1 * loss_perceptual + 
            0.01 * loss_TV
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch}/100]:")
    print(f"Total Training Loss: {total_loss.item():.4f}")

    # Validation loop
    unet.eval()
    val_loss = 0.0

    with torch.no_grad():
        for val_batch_idx, batch in enumerate(val_dataloader):
            train_images, low_light_images, t_maps, a_maps, l_maps = [x.to(device) for x in batch]
            dehazed_images = unet(train_images)
            loss_reconstruction = reconstruction_loss(dehazed_images, low_light_images)
            loss_consistency = consistency_loss(t_maps, l_maps)
            loss_hazy_reconstruction = hazy_reconstruction_loss(train_images, dehazed_images, t_maps, a_maps)
            loss_perceptual = perceptual_loss(vgg, dehazed_images, low_light_images)
            loss_TV = total_variation_loss(dehazed_images)
            total_loss = (
                1.0 * loss_reconstruction +
                0.5 * loss_consistency +
                1.0 * loss_hazy_reconstruction +
                0.1 * loss_perceptual +
                0.01 * loss_TV
            )
            val_loss += total_loss.item()

    val_loss /= len(val_dataloader)
    print(f"Validation Loss: {val_loss:.4f}")

# Save the trained model
torch.save(unet.state_dict(), "saved_models/unet_ssl2.pth")
