import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class OCT5kDataset(Dataset):
    """
    OCT5k dataset loader for a root directory containing six subfolders:

        root/
          train_img/
          train_msk/
          valid_img/
          valid_msk/
          test_img/
          test_msk/

    Images and masks are matched by filename stem (name without extension).
    For example, `train_img/scan_001.png` pairs with `train_msk/scan_001.png`
    (mask may have a different extension; only the stem must match).

    Parameters
    ----------
    root : str or Path
        Path to the dataset root containing the six split directories.
    split : {'train','valid','test'}
        Which split to load.
    img_transform : Optional[Callable]
        Transform applied to the PIL image (e.g., torchvision transforms).
        Should return a tensor (C×H×W). If None, image is converted to a
        float32 tensor in [0,1] with shape (1,H,W).
    msk_transform : Optional[Callable]
        Transform applied to the mask (PIL image). If None, mask is loaded
        as a LongTensor with shape (H,W) containing class IDs (no scaling).
        If you supply a transform, ensure it preserves integer class labels.
    return_filename : bool
        If True, __getitem__ returns (image, mask, filename_stem).
    img_extensions : Optional[Sequence[str]]
        Allowed image file extensions. Defaults to common formats.
    strict : bool
        If True, raises if any image is missing a mask (or vice versa).
        If False, silently ignores unpaired files and keeps only pairs.
    """

    SPLIT_DIRS = {
        "train": ("train_img", "train_msk"),
        "valid": ("valid_img", "valid_msk"),
        "test":  ("test_img",  "test_msk"),
    }

    def __init__(
        self,
        root: Path,
        split: str = "train",
        img_transform: Optional[Callable] = None,
        msk_transform: Optional[Callable] = None,
        return_filename: bool = False,
        img_extensions: Optional[Sequence[str]] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
        strict: bool = True,
    ):
        super().__init__()
        root = Path(root)
        if split not in self.SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(self.SPLIT_DIRS.keys())}, got {split}")

        img_dirname, msk_dirname = self.SPLIT_DIRS[split]
        self.img_dir = root / img_dirname
        self.msk_dir = root / msk_dirname

        if not self.img_dir.is_dir() or not self.msk_dir.is_dir():
            raise FileNotFoundError(
                f"Expected directories '{self.img_dir}' and '{self.msk_dir}' to exist."
            )

        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.return_filename = return_filename
        self.strict = strict

        # Index files
        exts = tuple(e.lower() for e in img_extensions) if img_extensions else None
        img_files = [p for p in self.img_dir.iterdir() if p.is_file() and (exts is None or p.suffix.lower() in exts)]
        msk_files = [p for p in self.msk_dir.iterdir() if p.is_file() and (exts is None or p.suffix.lower() in exts)]

        # Build dicts by stem
        img_by_stem: Dict[str, Path] = {p.stem: p for p in img_files}
        msk_by_stem: Dict[str, Path] = {p.stem: p for p in msk_files}

        # Intersect stems to form pairs
        common_stems = sorted(set(img_by_stem.keys()) & set(msk_by_stem.keys()))
        missing_img = sorted(set(msk_by_stem.keys()) - set(img_by_stem.keys()))
        missing_msk = sorted(set(img_by_stem.keys()) - set(msk_by_stem.keys()))

        if self.strict:
            errors = []
            if missing_img:
                errors.append(f"Masks with no image: {missing_img[:5]}{' ...' if len(missing_img) > 5 else ''}")
            if missing_msk:
                errors.append(f"Images with no mask: {missing_msk[:5]}{' ...' if len(missing_msk) > 5 else ''}")
            if errors:
                raise RuntimeError("Unpaired files detected:\n" + "\n".join(errors))

        self.samples: List[Tuple[Path, Path, str]] = [(img_by_stem[s], msk_by_stem[s], s) for s in common_stems]

        if len(self.samples) == 0:
            raise RuntimeError(f"No paired samples found in {self.img_dir} and {self.msk_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_image_as_tensor(path: Path) -> torch.Tensor:
        # OCT scans are typically grayscale; convert to 'L' to be consistent.
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0  # H×W in [0,1]
        # Add channel dimension -> (1,H,W)
        return torch.from_numpy(arr).unsqueeze(0)

    @staticmethod
    def _load_mask_as_long_tensor(path: Path) -> torch.Tensor:
        # Load mask as single-channel; pixel values are class IDs (integers).
        msk = Image.open(path).convert("L")
        arr = np.array(msk, dtype=np.int64)  # H×W with class ids
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        img_path, msk_path, stem = self.samples[idx]

        # Load
        img = Image.open(img_path).convert("L")  # keep PIL for optional transform
        msk = Image.open(msk_path).convert("L")

        # Apply transforms if provided
        if self.img_transform is not None:
            img_t = self.img_transform(img)
        else:
            # default: to float tensor (1,H,W) in [0,1]
            img_t = self._pil_to_norm_tensor(img)

        if self.msk_transform is not None:
            # IMPORTANT: ensure your mask transform preserves labels
            msk_t = self.msk_transform(msk)
        else:
            msk_t = torch.from_numpy(np.array(msk, dtype=np.int64))  # (H,W)

        if self.return_filename:
            return img_t, msk_t, stem
        return img_t, msk_t

    @staticmethod
    def _pil_to_norm_tensor(pil_img: Image.Image) -> torch.Tensor:
        """Convert single-channel PIL to (1,H,W) float tensor in [0,1]."""
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
