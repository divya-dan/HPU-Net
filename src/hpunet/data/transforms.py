
from __future__ import annotations
from typing import Tuple, Sequence
import numpy as np
import cv2


# basic utility functions
def _maybe_flip(img: np.ndarray, masks: np.ndarray, rng: np.random.Generator):
    # flip horizontally 50% of the time
    if rng.random() < 0.5:
        img = np.ascontiguousarray(np.fliplr(img))
        masks = np.ascontiguousarray(np.fliplr(masks))
    
    # flip vertically 50% of the time
    if rng.random() < 0.5:
        img = np.ascontiguousarray(np.flipud(img))
        masks = np.ascontiguousarray(np.flipud(masks))
    return img, masks


def _warp_affine_nn(m: np.ndarray, M: np.ndarray, W: int, H: int) -> np.ndarray:
    # nearest neighbor warp for masks
    return cv2.warpAffine(
        m.astype(np.uint8) * 255, M, dsize=(W, H),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ) > 127


def _warp_affine_lin(x: np.ndarray, M: np.ndarray, W: int, H: int) -> np.ndarray:
    # linear interpolation warp for images
    return cv2.warpAffine(
        x, M, dsize=(W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )



class RandomIntensity:
    """
    Random brightness and contrast adjustments for medical imgs
    """
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.3,
    ):
        self.brightness_range = (float(brightness_range[0]), float(brightness_range[1]))
        self.contrast_range = (float(contrast_range[0]), float(contrast_range[1]))
        self.p = float(p)

    def __call__(self, img: np.ndarray, masks: np.ndarray, rng: np.random.Generator):
        if rng.random() > self.p:
            return img, masks, np.ones(img.shape, dtype=bool)

        # random brightness multiplier
        brightness = float(rng.uniform(self.brightness_range[0], self.brightness_range[1]))
        
        # contrast adjustment around mean
        contrast = float(rng.uniform(self.contrast_range[0], self.contrast_range[1]))
        img_mean = img.mean()
        
        img_t = (img - img_mean) * contrast + img_mean
        img_t *= brightness
        img_t = np.clip(img_t, 0.0, 1.0)

        pad_mask = np.ones(img.shape, dtype=bool)
        return img_t.astype(np.float32), masks, pad_mask



class RandomGaussianNoise:
    """
    Add some gaussian noise to images
    """
    def __init__(
        self,
        noise_std: float = 0.005,  # pretty small for normalized imgs
        p: float = 0.2,
    ):
        self.noise_std = float(noise_std)
        self.p = float(p)

    def __call__(self, img: np.ndarray, masks: np.ndarray, rng: np.random.Generator):
        if rng.random() > self.p:
            return img, masks, np.ones(img.shape, dtype=bool)

        # add gaussian noise
        noise = rng.normal(0, self.noise_std, img.shape).astype(np.float32)
        img_t = img + noise
        img_t = np.clip(img_t, 0.0, 1.0)

        pad_mask = np.ones(img.shape, dtype=bool)
        return img_t.astype(np.float32), masks, pad_mask



class RandomAffine2D:
    """
    Does flips + rotation + scale + translation + shear around image center
    params are scaled for 180x180 input size
    """
    def __init__(
        self,
        max_rotate_deg: float = 15.0,
        max_translate_px: float = 11.0,  # scaled for 180x180
        scale_range: Tuple[float, float] = (0.95, 1.05),
        max_shear_deg: float = 10.0,
        p: float = 1.0,
    ):
        self.max_rotate_deg = float(max_rotate_deg)
        self.max_translate_px = float(max_translate_px)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.max_shear_deg = float(max_shear_deg)
        self.p = float(p)

    def __call__(self, img: np.ndarray, masks: np.ndarray, rng: np.random.Generator):
        assert img.ndim == 2 and masks.ndim == 3 and masks.shape[0] == 4
        H, W = img.shape

        if rng.random() > self.p:
            return img, masks, np.ones((H, W), dtype=bool)

        # do the flips first
        img, masks = _maybe_flip(img, masks, rng)

        # sample transform parameters
        angle = np.deg2rad(float(rng.uniform(-self.max_rotate_deg, self.max_rotate_deg)))
        scale = float(rng.uniform(self.scale_range[0], self.scale_range[1]))
        shear = np.deg2rad(float(rng.uniform(-self.max_shear_deg, self.max_shear_deg)))  # x-shear
        tx = float(rng.uniform(-self.max_translate_px, self.max_translate_px))
        ty = float(rng.uniform(-self.max_translate_px, self.max_translate_px))

        # build the 3x3 transform matrix: T * C * R * Sh * S * C^-1
        cx, cy = (W / 2.0, H / 2.0)
        C = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        Cinv = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle),  np.cos(angle), 0],
                      [0, 0, 1]], dtype=np.float32)
        Sh = np.array([[1, np.tan(shear), 0],
                       [0, 1, 0],
                       [0, 0, 1]], dtype=np.float32)
        S = np.array([[scale, 0, 0],
                      [0, scale, 0],
                      [0, 0, 1]], dtype=np.float32)
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

        A = T @ Cinv @ R @ Sh @ S @ C
        M = A[:2, :]  # cv2 wants 2x3 matrix

        img_t = _warp_affine_lin(img, M, W, H)
        masks_t = np.stack([_warp_affine_nn(m, M, W, H) for m in masks], axis=0).astype(np.uint8)

        # figure out which pixels stayed in bounds
        valid = np.ones((H, W), dtype=np.uint8)
        pad_mask = cv2.warpAffine(valid, M, dsize=(W, H),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        return img_t.astype(np.float32), masks_t, pad_mask



class RandomElastic2D:
    """
    Elastic deformation like in Simard et al paper
    params scaled for 180x180 input size
    """
    def __init__(
        self,
        alpha_range: Tuple[float, float] = (11.0, 17.0),  # scaled for 180x180
        sigma: float = 8.5,                               # scaled for 180x180
        p: float = 0.5,
    ):
        self.alpha_range = (float(alpha_range[0]), float(alpha_range[1]))
        self.sigma = float(sigma)
        self.p = float(p)

    def __call__(self, img: np.ndarray, masks: np.ndarray, rng: np.random.Generator):
        H, W = img.shape
        if rng.random() > self.p:
            return img, masks, np.ones((H, W), dtype=bool)

        alpha = float(rng.uniform(self.alpha_range[0], self.alpha_range[1]))

        # create random displacement fields
        dx = rng.standard_normal((H, W)).astype(np.float32)
        dy = rng.standard_normal((H, W)).astype(np.float32)
        ksize = int(2 * round(3 * self.sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1  # make sure kernel size is odd
        dx = cv2.GaussianBlur(dx, (ksize, ksize), self.sigma) * (alpha / dx.std())
        dy = cv2.GaussianBlur(dy, (ksize, ksize), self.sigma) * (alpha / dy.std())

        # create coordinate grid
        x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        img_t = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        # remap each mask
        masks_t = []
        for m in masks:
            mt = cv2.remap(m.astype(np.uint8) * 255, map_x, map_y,
                           interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
            masks_t.append(mt)
        masks_t = np.stack(masks_t, axis=0).astype(np.uint8)

        # track validity by remapping an all-ones mask
        valid = np.ones((H, W), dtype=np.uint8)
        pad_mask = cv2.remap(valid, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        return img_t.astype(np.float32), masks_t, pad_mask


class ComposeAug2D:
    """chains multiple transforms together and combines the pad masks"""
    def __init__(self, transforms: Sequence[object]):
        self.transforms = list(transforms)

    def __call__(self, img: np.ndarray, masks: np.ndarray, rng: np.random.Generator):
        H, W = img.shape
        pad = np.ones((H, W), dtype=bool)
        for t in self.transforms:
            img, masks, p = t(img, masks, rng)
            pad &= p
        return img, masks, pad


def build_default_transform() -> ComposeAug2D:
    """
    Our main augmentation pipeline - matches DeepMind's approach
    Applied to 180x180 imgs before cropping to 128x128
    """
    return ComposeAug2D([
        RandomIntensity(brightness_range=(0.95, 1.05), contrast_range=(0.95, 1.05), p=0.3),
        RandomGaussianNoise(noise_std=0.005, p=0.2),
        RandomAffine2D(
            max_rotate_deg=15.0,
            max_translate_px=11.0,  # scaled for 180x180
            scale_range=(0.95, 1.05),
            max_shear_deg=10.0,
            p=1.0
        ),
        RandomElastic2D(
            alpha_range=(11.0, 17.0),  # scaled for 180x180
            sigma=8.5,                 # scaled for 180x180
            p=0.5
        ),
    ])