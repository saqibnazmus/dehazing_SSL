import torch
from model import UNet
from PIL import Image
import torchvision.transforms as transforms

def test_unet(model, test_image_path, device):
    image = Image.open(test_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    noisy_image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_image = model(noisy_image)
    return denoised_image.squeeze(0).cpu()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
pretrained_weights_path = "saved_models/unet_ssl2.pth"
model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
model.eval()

test_image_path = "data/SOTS-indoor/outdoor/hazy/0001_0.8_0.2.jpg"
dehazed_image = test_unet(model, test_image_path, device)
