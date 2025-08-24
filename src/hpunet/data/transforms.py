from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def _maybe_flip(img: np.ndarray, masks: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Horizontal flip
    if rng.random() < 0.5:
        img = np.ascontiguousarray(np.fliplr(img))
        masks = np.ascontiguousarray(np.fliplr(masks))
    # Vertical flip
    if rng.random() < 0.5:
        img = np.ascontiguousarray(np.flipud(img))
        masks = np.ascontiguousarray(np.flipud(masks))
    return img, masks


class RandomAffine2D:
    """
    Minimal affine augmenter for 2D grayscale images + 4 binary masks.

    Applies flips, rotation, scale, and translation (no shear), then returns:
      - transformed image (float32 in [0,1])
      - transformed masks (uint8 {0,1})
      - pad_mask (bool) indicating pixels that remained inside the original field of view
    """
    def __init__(
        self,
        max_rotate_deg: float = 15.0,
        max_translate_px: float = 8.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        p: float = 1.0,
    ):
        self.max_rotate_deg = float(max_rotate_deg)
        self.max_translate_px = float(max_translate_px)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.p = float(p)

    def __call__(self, img: np.ndarray, masks: np.ndarray, rng: np.random.Generator):
        assert img.ndim == 2 and masks.ndim == 3 and masks.shape[0] == 4, "Expect image [H,W] and masks [4,H,W]"

        if rng.random() > self.p:
            H, W = img.shape
            return img, masks, np.ones((H, W), dtype=bool)

        # Flips (no padding introduced)
        img, masks = _maybe_flip(img, masks, rng)

        H, W = img.shape
        cx, cy = (W / 2.0, H / 2.0)

        angle = float(rng.uniform(-self.max_rotate_deg, self.max_rotate_deg))
        scale = float(rng.uniform(self.scale_range[0], self.scale_range[1]))
        tx = float(rng.uniform(-self.max_translate_px, self.max_translate_px))
        ty = float(rng.uniform(-self.max_translate_px, self.max_translate_px))

        # 2x3 affine: rotation + scale about center, then add translation
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)  # returns 2x3
        M[0, 2] += tx
        M[1, 2] += ty

        img_t = cv2.warpAffine(
            img,
            M,
            dsize=(W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        masks_t = np.stack(
            [
                cv2.warpAffine(
                    m.astype(np.uint8) * 255,
                    M,
                    dsize=(W, H),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ) > 127
                for m in masks
            ],
            axis=0,
        ).astype(np.uint8)

        # Validity map: warp an all-ones mask and see which pixels stayed in-bounds
        valid = np.ones((H, W), dtype=np.uint8)
        valid_t = cv2.warpAffine(
            valid,
            M,
            dsize=(W, H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        pad_mask = valid_t > 0  # bool

        return img_t.astype(np.float32), masks_t, pad_mask
