import numpy as np
import cv2
from scipy.fftpack import dct, idct

# ============================================================================
# JPEG Artifact Mitigation - Processing Function
# ============================================================================


def _dct2(block):
    """Compute 2D Discrete Cosine Transform."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def _idct2(block):
    """Compute 2D Inverse Discrete Cosine Transform."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# --- DCT Denoise ---
def _frequency_domain_artifact_suppression(img, alpha=0.68):
    result = np.zeros_like(img, dtype=np.float32)
    
    h, w = img.shape[:2]
    
    h_block = h - (h % 8)
    w_block = w - (w % 8)

    for c in range(3):
        channel = img[:, :, c]
        channel_result = np.zeros_like(channel, dtype=np.float32)
        
        for i in range(0, h_block, 8):
            for j in range(0, w_block, 8):
                block = channel[i:i+8, j:j+8].astype(np.float32)
                
                # to frequency domain
                dct_block = _dct2(block)
                
                # Create frequency attenuation mask
                mask = np.ones((8, 8))
                for u in range(8):
                    for v in range(8):
                        freq_magnitude = u + v
                        if freq_magnitude > 6:
                            mask[u, v] = alpha * 0.5
                        elif freq_magnitude > 4:
                            mask[u, v] = alpha * 0.75
                        elif freq_magnitude > 2:
                            mask[u, v] = alpha
                
                # Apply mask
                dct_block_filtered = dct_block * mask
                
                # Back to spatial domain
                reconstructed = _idct2(dct_block_filtered)
                channel_result[i:i+8, j:j+8] = reconstructed

        # Handle remaining pixels if image size not multiple of 8
        if h % 8 != 0:
            channel_result[h_block:, :] = channel[h_block:, :]
        if w % 8 != 0:
            channel_result[:, w_block:] = channel[:, w_block:]

        result[:, :, c] = channel_result
    
    return np.clip(result, 0, 255).astype(np.uint8)


# --- Main Process Function ---
def JPEG_Artifact_Mitigation_C(img, alpha=0.68):
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input image must be a BGR color image.")

    denoised_img = _frequency_domain_artifact_suppression(img, alpha=alpha)
    
    return denoised_img

# --- Non-Local Means (NLM) (if needed) ---
def JPEG_Artifact_Mitigation_NLM(img, h=10, template_window_size=7, search_window_size=21):
    """
    input: BGR image (NumPy array)
    output: denoised BGR image (NumPy array)
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        img, None, h, h, template_window_size, search_window_size
    )
    return denoised
