import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def to_fft(img_pil):
    """Converts spatial images into 2D frequency magnitude spectrums."""
    gray = np.array(img_pil.convert('L')).astype(np.float32)
    mag  = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    mag  = ((mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(mag).convert('RGB')

# Extreme Equalizer / Spatial Bottleneck
SPATIAL_TFM = transforms.Compose([
    transforms.CenterCrop(150), 
    transforms.Resize((224, 224)),
    transforms.GaussianBlur(kernel_size=5, sigma=1.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

FFT_TFM = transforms.Compose([
    transforms.CenterCrop(150), 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class DualStreamDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
        
    def __len__(self): 
        return len(self.paths)
        
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        sp_tensor = SPATIAL_TFM(img)
        fq_tensor = FFT_TFM(to_fft(img))
        y = torch.tensor(self.labels[i], dtype=torch.float32)
        return sp_tensor, fq_tensor, y
