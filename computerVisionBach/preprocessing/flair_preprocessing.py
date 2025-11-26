import os
import csv
import sys
import numpy as np
import albumentations as A

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.augmentation import build_aug_from_cfg
from typing import Tuple, List, Optional
from datasets.flair_dataset import FlairDataset
from utils.config_loader import load_config
cfg = load_config("config.yaml")
aug = cfg.augmentation
model_name = cfg.model.name.lower()


if model_name == "mask2former":
    train_tf = build_aug_from_cfg(aug.mask2former.train, aug.normalize, aug.ade_normalize)
    val_tf   = build_aug_from_cfg(aug.mask2former.val,   aug.normalize, aug.ade_normalize)

elif model_name in ["segformer", "upernet"]:
    train_tf = build_aug_from_cfg(aug.segformer.train, aug.normalize, aug.ade_normalize)
    val_tf   = build_aug_from_cfg(aug.segformer.val,   aug.normalize, aug.ade_normalize)

else:
    train_tf = build_aug_from_cfg(aug.smp.train, aug.normalize, aug.ade_normalize)
    val_tf   = build_aug_from_cfg(aug.smp.val,   aug.normalize, aug.ade_normalize)



# =====================================
# patchify and load data FLAIR
# =====================================
def prepare_datasets_from_csvs(
    train_csv_path: str,
    val_csv_path: str,
    test_csv_path: Optional[str] = None,
    base_dir: str = None
) -> Tuple[FlairDataset, FlairDataset, FlairDataset]:
    def load_csv(csv_path: str) -> List[Tuple[str, str]]:
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            return [(row[0], row[1]) for row in reader if len(row) == 2]

    def resolve_path(p: str) -> str:
        return os.path.normpath(os.path.join(base_dir, p)) if base_dir and not os.path.isabs(p) else p

    # Load CSVs
    train_pairs = load_csv(train_csv_path)
    val_pairs = load_csv(val_csv_path)


    train_pairs = [(resolve_path(img), resolve_path(mask)) for img, mask in train_pairs]
    val_pairs = [(resolve_path(img), resolve_path(mask)) for img, mask in val_pairs]

    train_imgs, train_masks = zip(*train_pairs)
    val_imgs, val_masks = zip(*val_pairs)
    #test_images, test_masks = train_imgs[:40], train_masks[:40]
    #t_val_images, t_val_masks = val_imgs[:40], val_masks[:40]
    def relabel_fn(mask):
        relabeled = np.full_like(mask, 255, dtype=np.int64)
        valid = (mask >= 1) & (mask <= 19)
        relabeled[valid] = mask[valid] - 1
        return relabeled

    def relabel_fn_12(mask):
        """
        Keeps only class labels 1 to 12 from the original FLAIR dataset,
        and remaps them to 0 to 11. All other pixels are mapped to 255 (ignore index).
        """
        relabeled = np.full_like(mask, 255, dtype=np.uint8)
        for i in range(1, 13):
            relabeled[mask == i] = i - 1
        return relabeled

    train_dataset = FlairDataset(train_imgs, train_masks, transform=train_tf, relabel_fn=relabel_fn_12, allowed_labels=tuple(range(12)), use_processor=cfg.data.flair.use_processor, is_hf_model=cfg.data.flair.is_hf_model)
    val_dataset = FlairDataset(val_imgs, val_masks,transform=val_tf, relabel_fn=relabel_fn_12, allowed_labels=tuple(range(12)), use_processor=cfg.data.flair.use_processor, is_hf_model=cfg.data.flair.is_hf_model)

    if test_csv_path is not None:
        test_pairs = load_csv(test_csv_path)
        test_pairs = [(resolve_path(img), resolve_path(mask)) for img, mask in test_pairs]
        test_imgs, test_masks = zip(*test_pairs)
        test_dataset = FlairDataset(test_imgs, test_masks, transform=None)

    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset

def relabel_mask(mask: np.ndarray, original_labels: list) -> np.ndarray:
    """
    Remaps original sparse label values (e.g., [1, 2, 6, 18]) to contiguous [0, 1, 2, ..., N-1]
    so CrossEntropyLoss works without index errors.
    """
    label_map = {orig: new for new, orig in enumerate(sorted(original_labels))}
    remapped = np.zeros_like(mask)
    for orig_label, new_label in label_map.items():
        remapped[mask == orig_label] = new_label
    return remapped