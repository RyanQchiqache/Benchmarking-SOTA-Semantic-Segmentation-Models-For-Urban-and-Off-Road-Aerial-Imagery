import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_aug_from_cfg(aug_cfg, normalize_cfg=None, ade_cfg=None):
    aug_list = []

    # flips & rotations
    if aug_cfg.get("horizontal_flip"):
        aug_list.append(A.HorizontalFlip(p=aug_cfg.horizontal_flip))

    if aug_cfg.get("vertical_flip"):
        aug_list.append(A.VerticalFlip(p=aug_cfg.vertical_flip))

    if aug_cfg.get("rotate90"):
        aug_list.append(A.RandomRotate90(p=aug_cfg.rotate90))

    # Normalization
    if aug_cfg.get("use_normalize") and normalize_cfg:
        aug_list.append(
            A.Normalize(mean=normalize_cfg.mean, std=normalize_cfg.std)
        )

    if aug_cfg.get("use_ade_normalize") and ade_cfg:
        aug_list.append(
            A.Normalize(mean=ade_cfg.mean, std=ade_cfg.std)
        )

    aug_list.append(ToTensorV2())

    return A.Compose(aug_list)
