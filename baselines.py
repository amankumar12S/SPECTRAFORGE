import glob, random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

def get_flat_fft(img_path):
    img_pil = Image.open(img_path).convert('L').resize((128, 128))
    gray = np.array(img_pil).astype(np.float32)
    mag  = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    mag  = ((mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255).astype(np.uint8)
    return mag.flatten()

# Add code here to load paths from your dataset directories and train the RF
