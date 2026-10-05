import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from configs.logging_config import logger

class ELADetector:
    """Error Level Analysis (ELA) for image tampering & splicing detection.

    NOTE: ELA is a JPEG-recompression-artifact technique. It is most
    meaningful for images that were originally JPEG-compressed (where
    tampered regions re-compress differently from untouched regions).
    Uncompressed/lossless source formats (BMP, PNG, some WEBP) have no
    prior JPEG quantization to compare against, so ELA will naturally read
    "high difference" everywhere -- not because they were tampered with,
    but because the whole image is compressing for the first time. This is
    a genuine, documented limitation of the ELA method itself, not a defect
    in this implementation; `source_format_caveat` surfaces it to the UI.
    """

    def __init__(self, quality: int = 90, scale: int = 15):
        self.quality = quality
        self.scale = scale

    def compute_ela(self, image_path: str, temp_out_path: str = None) -> tuple:
        """
        Computes Error Level Analysis (ELA) map and average compression anomaly score.
        Returns (ela_image_pil, ela_score, is_manipulated_flag, source_format_caveat).
        On failure, returns (None, None, None, error_message) -- callers must
        handle this explicitly rather than silently displaying a fabricated
        "not manipulated" result.
        """
        import uuid
        temp_out_path = temp_out_path or f"temp_ela_{uuid.uuid4().hex[:8]}.jpg"
        try:
            with Image.open(image_path) as src:
                source_format = (src.format or "").upper()
                original = src.convert('RGB')

            # Re-compress image at target quality
            original.save(temp_out_path, 'JPEG', quality=self.quality)
            compressed = Image.open(temp_out_path)

            # Compute pixel difference between original and re-compressed version
            ela_img = ImageChops.difference(original, compressed)

            # Scale difference for visual inspection
            extrema = ela_img.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1

            scale_factor = 255.0 / max_diff
            ela_img = ImageEnhance.Brightness(ela_img).enhance(scale_factor)

            # Calculate average ELA intensity score
            ela_np = np.array(ela_img)
            mean_intensity = float(np.mean(ela_np))
            std_intensity = float(np.std(ela_np))

            # Anomaly heuristic: manipulated/spliced images exhibit high ELA variance
            ela_score = min(mean_intensity / 100.0, 1.0)
            is_manipulated = std_intensity > 35.0 or mean_intensity > 40.0

            caveat = None
            if source_format not in ("JPEG", "JPG"):
                caveat = (
                    f"Source image is {source_format or 'an unrecognized format'}, not JPEG. "
                    "ELA is calibrated for JPEG recompression artifacts, so results on "
                    "non-JPEG sources are less reliable than the pHash/ResNet-50 signals."
                )

            if os.path.exists(temp_out_path):
                os.remove(temp_out_path)

            return ela_img, ela_score, is_manipulated, caveat

        except Exception as e:
            logger.error(f"Error computing ELA for {image_path}: {e}")
            if os.path.exists(temp_out_path):
                try:
                    os.remove(temp_out_path)
                except OSError:
                    pass
            # Honest failure signal -- never fabricate a "not manipulated"
            # verdict when the computation itself did not succeed.
            return None, None, None, f"ELA computation failed: {e}"
