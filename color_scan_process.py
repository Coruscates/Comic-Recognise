# %%
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Image Processing: Color & Scan

# %%
# Gray scale block
def gray_world(img):
    # Gray world
    b, g, r = cv2.split(img)
    
    b_avg = np.mean(b)
    g_avg = np.mean(g)
    r_avg = np.mean(r)
    
    avg = (b_avg + g_avg + r_avg) / 3
    
    b = np.clip(b * (avg / b_avg), 0, 255).astype(np.uint8)
    g = np.clip(g * (avg / g_avg), 0, 255).astype(np.uint8)
    r = np.clip(r * (avg / r_avg), 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

    # return cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2GRAY)
def grayscale_transfer(img):
    """
    Convert a colored comic page to grayscale
    """
    img = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    return gray

# %%
# Retinex algorithms block
def single_scale_retinex(img, sigma=60):
    img = img.astype(np.float32) + 1.0
    log_img = np.log(img)

    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    log_blur = np.log(blur)

    retinex = log_img - log_blur
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)
def multi_scale_retinex(img, sigmas=[15, 80, 250]):
    retinex = np.zeros_like(img, dtype=np.float32)

    for sigma in sigmas:
        retinex += single_scale_retinex(img, sigma)
    
    retinex /= len(sigmas)
    retinex = grayscale_transfer(retinex)
    return retinex.astype(np.uint8)

# %%
def mean_threshold(img):
    mean_val = np.mean(img)
    print("Mean value: {}".format(mean_val))
    if mean_val < 128:
        mean_val = 128
    out = np.where(img > mean_val,
                   255,
                   0)
    return out.astype(np.uint8)

def adaptive_gaussian(img):
    blk_size = img.shape[0] // 10
    if blk_size % 2 == 0:
        blk_size += 1
    out = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blk_size,   # block size
        5    # subtract constant
    )
    return out

# %%
def remove_small_components(img, min_size=5):
    img = cv2.bitwise_not(img)
    
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img_processed = np.zeros(img.shape, dtype=np.uint8)

    for i in range(nb_components):
        if sizes[i] >= min_size:
            img_processed[output == i + 1] = 255
            
    img_cleaned = cv2.bitwise_not(img_processed)
    return img_cleaned

# %%
def extract_line_art_morphological(img):
    """
    Extract line art from comic images using morphological operations.
    
    """
    
    # Step 1: Load and convert to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
  
    # Step 2: Enhance contrast
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    contrast = np.std(gray)
    # print("Contrast (std dev): {}".format(contrast))
    clipLimit = 5.0 if contrast < 50 else 1.0  # adjust denominator for sensitivity
    clipLimit = np.clip(clipLimit, 1.0, 5.0)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Step 3: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Step 4: Threshold to get binary line art
    # Otsu's method automatically finds optimal threshold
    _, line_art = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5: Clean up with morphological operations
    # Remove small noise
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    line_art = cv2.morphologyEx(line_art, cv2.MORPH_OPEN, clean_kernel)
    # Close gaps in lines
    line_art = cv2.morphologyEx(line_art, cv2.MORPH_CLOSE, clean_kernel)
    return line_art

# %%
def Color_Scan_Processing_A(img):
    '''
    Total Color&Scan Processing Function A
    '''
    img_multi_scale_retinex = multi_scale_retinex(img)
    img_adaptive_gauss = adaptive_gaussian(img_multi_scale_retinex)
    img_cleaned = remove_small_components(img_adaptive_gauss)
    
    return img_cleaned
    

# %%
def Color_Scan_Processing_B(img):
    '''
    Total Color&Scan Processing Function B
    '''
    img_multi_scale_retinex = multi_scale_retinex(img)
    img_lined = extract_line_art_morphological(img_multi_scale_retinex)

    return img_lined

# # %%
# img_color = cv2.imread("Manga109_released_2023_12_07/images/ARMS/005.jpg")
# Image.fromarray(img_color)

# # %%
# out1 = Color_Scan_Processing_A(img_color)
# Image.fromarray(out1)

# # %%
# out2 = Color_Scan_Processing_B(img_color)
# Image.fromarray(out2)


