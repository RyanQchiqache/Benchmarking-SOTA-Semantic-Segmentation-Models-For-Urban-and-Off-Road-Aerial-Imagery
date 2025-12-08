# Benchmarking State-of-the-Art Semantic Segmentation Models for Urban and Off-Road Aerial Imagery

## Introduction

This project benchmarks state-of-the-art semantic segmentation models on aerial imagery from both urban and off-road environments. It includes implementations and evaluations of:

* **U-Net**
  [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597) — a convolutional network architecture for biomedical image segmentation, widely adapted for remote sensing.

* **DeepLabV3+**
  [Chen et al., 2018](https://arxiv.org/abs/1802.02611) — combines spatial pyramid pooling and encoder-decoder structure for improved context capture and edge refinement.

* **Mask2Former**
  [Cheng et al., 2022](https://arxiv.org/abs/2112.01527) — a unified transformer-based architecture for instance, semantic, and panoptic segmentation.

* **UPerNet**
  [Xiao et al., 2018](https://arxiv.org/abs/1807.10221) — Unified Perceptual Parsing Network, incorporating Pyramid Pooling Module and Feature Pyramid Network.

### Datasets

* **DLR SkyScapes**:
  **Access**: [DLR SkyScapes Official Dataset](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes)

* **FLAIR (French Land Cover from Aerial Image Reference)**:
  **Access**: [FLAIR Dataset](https://ignf.github.io/FLAIR/FLAIR2/flair_2.html)

These models are tested on their performance, generalization, and usability for aerial segmentation tasks across varied terrains.


## Repository Overview

```
computerVisionBach/
├── configs/
│   └── config.yaml
├── datasets/
├── models/
│   └── model_pipleline.py
├── preprocessing/
├── utilities/
├── visualisation_app/
└── requirements.txt
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/RyanQchiqache/Benchmarking-SOTA-Semantic-Segmentation-Models-For-Urban-and-Off-Road-Aerial-Imagery.git
cd computerVisionBach
```

### 2. Create and Activate a Python Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Project Root

If needed, export the root for module imports:

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## Configuration (`configs/config.yaml`)

All parameters (dataset, model, training, paths) are defined here.

### Modify for Your Use

* **Change dataset paths**:

```yaml
data:
  dlr:
    root_dir: "/your/path/to/DLR"
  flair:
    base_dir: "/your/path/to/FLAIR"
    train_csv: "/your/path/to/FLAIR/train.csv"
    val_csv:   "/your/path/to/FLAIR/test.csv"
    test_csv:  "/your/path/to/FLAIR/test.csv"
```

* **Select Model**:

```yaml
model:
  name: "Unet"  # or DeepLabV3+, Mask2Former, UPerNet
```

* **Training Parameters**:

```yaml
training:
  batch_size: # choose batchsize that corresponds to your model and GPU capacity
  num_epochs: # number of epochs depending on your need
  learning_rate: # LR depending on the model
```

* **Logging and Outputs**:

```yaml
paths:
  artifacts_root: "/your/output/dir"
  tensorboard:
    dir: "${paths.artifacts_root}/runs"
```

## Training

To train your model:

```bash
python models/model_pipeline.py
```

Monitor training using TensorBoard:

```bash
tensorboard --logdir runs/tensorboard --port # choose a free port
```

## Inference and Visualization

Launch Streamlit app for image segmentation:

```bash
streamlit run visualisation_app/vis_app.py --server.maxUploadSize=1024 # for bigger images [png, jpeg, jpg, tif..etc]
```

* Choose a trained model checkpoint
* Upload or select a test image
* View segmentation results

## Model Summaries

### U-Net

Encoder-decoder with skip connections for fine-grained segmentation. Lightweight and fast.

### DeepLabV3+

Uses atrous convolutions and ASPP for multiscale feature extraction. Good boundary performance.

### Mask2Former

Transformer-based model with unified mask prediction. State-of-the-art results.

### UPerNet

Pyramid-based multiscale segmentation with strong context fusion.

## Notes

* Current dataset support: **DLR SkyScapes**, **FLAIR**
* Custom datasets: coming in future update
* Supports Python 3.12 and `venv`

## Citation

```
@misc{BenchmarkingAerialSeg2025,
  title   = {Benchmarking State-of-the-Art Semantic Segmentation Models for Urban and Off-Road Aerial Imagery},
  author  = {Ryan Qchiqache},
  year    = {2025},
  howpublished = {https://github.com/RyanQchiqache/Benchmarking-SOTA-Semantic-Segmentation-Models-For-Urban-and-Off-Road-Aerial-Imagery}}
}
```

---

For questions or issues, please open an issue on the GitHub repository.
