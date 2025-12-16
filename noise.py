import numpy as np
import cv2

# ============================================================================
# Synthetic Noise Application
# ============================================================================

def Synthetic_Noise_Application(img_bgr, noise_level=0.1, jpeg_quality=70):    
    img_float = img_bgr.astype(np.float32)
    
    # --- 1. Gaussian ---
    sigma = noise_level * 255
    gaussian_noise = np.random.normal(0, sigma, img_float.shape).astype(np.float32)
    img_noisy = img_float + gaussian_noise
    
    # --- 2. Simulated uneven brightness/smudges ---
    # Determine blur kernel size based on image dimensions
    
    h, w = img_bgr.shape[:2]
    blur_kernel_size = min(h, w) // 5 
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    # random low-frequency noise
    low_freq_noise = np.random.normal(0, 15, img_float.shape).astype(np.float32)
    low_freq_noise = cv2.GaussianBlur(low_freq_noise, (blur_kernel_size, blur_kernel_size), 0)
    
    img_noisy += low_freq_noise * 0.5 
    
    # --- 3. Simulated JPEG compression artifacts (Blocking artifacts) ---
    
    img_compressed_input = np.clip(img_noisy, 0, 255).astype(np.uint8)
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encimg = cv2.imencode('.jpg', img_compressed_input, encode_param)
    img_compressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)


    return img_compressed.astype(np.uint8)
