import os
import cv2
import sys
import numpy as np
import albumentations as A

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from albumentations.pytorch import ToTensorV2
from transformers import SegformerImageProcessor
from preprocessing import utils
from datasets.satellite_data import SatelliteDataset
from utils.config_loader import load_config
cfg = load_config("config.yaml")

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
patch_size, overlap = 512, 0.5
from utils.augmentation import build_aug_from_cfg

aug = cfg.augmentation

# Model name → appropriate augmentation set
model_name = cfg.model.name.lower()

if model_name == "mask2former":
    train_tf = build_aug_from_cfg(
        aug.mask2former.train, aug.normalize, aug.ade_normalize
    )
    val_tf = build_aug_from_cfg(
        aug.mask2former.val, aug.normalize, aug.ade_normalize
    )

elif model_name in ["segformer", "upernet"]:
    train_tf = build_aug_from_cfg(
        aug.segformer.train, aug.normalize, aug.ade_normalize
    )
    val_tf = build_aug_from_cfg(
        aug.segformer.val, aug.normalize, aug.ade_normalize
    )

else:  # UNet, DeepLab, ResNet-SMP models
    train_tf = build_aug_from_cfg(
        aug.smp.train, aug.normalize, aug.ade_normalize
    )
    val_tf = build_aug_from_cfg(
        aug.smp.val, aug.normalize, aug.ade_normalize
    )
sg_val_tf =None

# =====================================
# patchify and load data DLR skyscapes
# =====================================

def patchify_image_or_masks(image_mask: np.ndarray, patch_size, overlap: float):
    patch_size = patch_size
    step = int(patch_size * (1 - overlap))
    H, W = image_mask.shape[:2]

    pad_bottom = (patch_size - H % patch_size) % patch_size
    pad_right = (patch_size - W % patch_size) % patch_size


    if pad_bottom or pad_right:
        constant_values = 255 if image_mask.ndim == 2 else 0
        pad_shape = ((0, pad_bottom), (0, pad_right), (0, 0)) if image_mask.ndim == 3 else ((0, pad_bottom),
                                                                                                (0, pad_right))
        image = np.pad(image_mask, pad_shape, mode="constant", constant_values=constant_values)
        H, W = image.shape[:2]
    else:
        image = image_mask

    patches = []
    for top in range(0, H - patch_size + 1, step):
        for left in range(0, W - patch_size + 1, step):
            patch = image[top:top + patch_size, left:left + patch_size]
            patches.append(patch)

    return patches, (H, W)


def load_folder(image_dir, mask_dir=None, patchify_enabled:bool=True):
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    X, y = [], []
    if mask_dir:
        masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
        for img_path, mask_path in zip(images, masks):
            img = cv2.imread(img_path).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

            unique_classes = np.unique(mask)
            #print(f"{os.path.basename(mask_path)} → Classes present: {unique_classes}")

            if patchify_enabled:
                image_p, _ = patchify_image_or_masks(img, patch_size, overlap)
                mask_p, _ = patchify_image_or_masks(mask, patch_size, overlap)
                X.extend(image_p)
                y.extend(mask_p)


            else:
                X.append(img)
                y.append(mask)

        return X, y
    else:
        for img_path in images:
            img = cv2.imread(img_path)
            if patchify_enabled:
                image_p, _ = patchify_image_or_masks(img, patch_size, overlap)
                X.extend(image_p)
            else:
                X.append(img)

        return X

def load_data_dlr(base_dir, dataset_type="SS_Dense", model_name="Mask2former"):
    base = os.path.join(base_dir, dataset_type)

    X_train, y_train = load_folder(
        os.path.join(base, "train/images"),
        os.path.join(base, "train/labels/grayscale")
    )
    X_val, y_val = load_folder(
        os.path.join(base, "val/images"),
        os.path.join(base, "val/labels/grayscale")
    )
    X_test = load_folder(
        os.path.join(base, "test/images"),
        mask_dir=None
    )

    color_map_rgb = {k: utils.hex_to_rgb(v[1]) for k, v in utils.COLOR_MAP_dense.items()}

    """visualisation.visualize_sample(
        X_train,
        [utils.class_to_rgb(mask, color_map_rgb) for mask in y_train],
        y_train
    )"""
    ## relabel 1–20 → 0–19, and 0 → 255 (ignored)
    def relabel_fn(mask):
        relabeled = np.full_like(mask, 255, dtype=np.int64)  # default to ignore index
        valid = (mask >= 1) & (mask <= 20)
        relabeled[valid] = mask[valid] - 1  # remap 1–20 → 0–19
        return relabeled

    if model_name.lower() == "mask2former":
        train_dataset = SatelliteDataset(X_train, y_train, transform=train_tf, relabel_fn=relabel_fn, is_hf_model=True)
        val_dataset = SatelliteDataset(X_val, y_val, relabel_fn=relabel_fn, is_hf_model=True)
        test_dataset = SatelliteDataset(X_test, masks=None, is_hf_model=True)


    elif model_name.lower() in ["segformer", "upernet"]:
        train_dataset = SatelliteDataset(
            X_train, y_train, transform=sg_val_tf, relabel_fn=relabel_fn, is_hf_model=True
        )
        val_dataset = SatelliteDataset(
            X_val, y_val, transform=sg_val_tf, relabel_fn=relabel_fn, is_hf_model=True
        )
        test_dataset = SatelliteDataset(X_test, masks=None, transform=sg_val_tf, is_hf_model=True)


    else:
        train_dataset = SatelliteDataset(X_train, y_train, transform=train_tf, relabel_fn=relabel_fn_ignore, use_processor=cfg.data.dlr.use_processor, is_hf_model=cfg.data.dlr.is_hf_model)
        val_dataset = SatelliteDataset(X_val, y_val, transform=val_tf, relabel_fn=relabel_fn_ignore, is_hf_model=cfg.data.dlr.is_hf_model)
        test_dataset = SatelliteDataset(X_test, masks=None, transform=val_tf, is_test=True, is_hf_model=cfg.data.dlr.is_hf_model)

    """num_classes = 20  # or whatever you set in the model
    for i, (_, mask) in enumerate(train_dataset):
        uniques = torch.unique(mask)
        print(f"Sample {i}: Unique labels:", uniques.tolist())
        valid = (uniques == 255) | ((uniques >= 0) & (uniques < num_classes))
        assert torch.all(valid), f"Invalid label(s) in sample {i}: {uniques.tolist()}"""

    return train_dataset, val_dataset, test_dataset


def relabel_fn_ignore(mask):
    """
    Remaps original class labels 1–20 into contiguous 0–15 labels,
    ignoring classes 3, 5, 13, and 16 (set to 255).
    """
    to_ignore = {3, 5, 13, 16}
    valid_classes = [i for i in range(1, 21) if i not in to_ignore]  # [1,2,4,6,...,20]
    mapping = {orig: new_id for new_id, orig in enumerate(valid_classes)}

    relabeled = np.full_like(mask, 255, dtype=np.int64)  # Default = ignore

    for orig, new in mapping.items():
        relabeled[mask == orig] = new

    return relabeled