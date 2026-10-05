import cv2
import numpy as np
from PIL import Image
from configs.logging_config import logger

class PerceptualHashDetector:
    """Perceptual Hashing (pHash / dHash) for image similarity & reuse detection."""

    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size

    def compute_phash(self, image_path: str) -> str:
        """Computes 64-bit DCT perceptual hash string of an image."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Pillow fallback
                img_pil = Image.open(image_path).convert('L')
                img = np.array(img_pil)

            img = cv2.resize(img, (32, 32))
            img_float = np.float32(img)
            dct = cv2.dct(img_float)
            dct_low = dct[:self.hash_size, :self.hash_size]
            avg = np.mean(dct_low)
            hash_bits = dct_low > avg
            
            # Convert boolean array to hex string
            hex_str = ''.join(['1' if b else '0' for b in hash_bits.flatten()])
            return hex(int(hex_str, 2))[2:].zfill(16)
        except Exception as e:
            logger.error(f"Error computing pHash for {image_path}: {e}")
            return "0000000000000000"

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Computes Hamming Distance between two 64-bit hex hash strings."""
        try:
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            xor_val = val1 ^ val2
            return bin(xor_val).count('1')
        except Exception as e:
            logger.error(f"Error calculating Hamming distance: {e}")
            return 64

    def is_reused(self, hash1: str, hash2: str, threshold: int = 8) -> bool:
        """Returns True if Hamming distance <= threshold (indicating reused or repurposed image)."""
        dist = self.hamming_distance(hash1, hash2)
        return dist <= threshold
