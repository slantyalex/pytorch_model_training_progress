# animalvision

Modular PyTorch computer vision project for image classification using ResNet18 and CIFAR-10.

## Setup

1. Create and activate virtual environment (Python 3.12):

```
py -3.12 -m venv venv
source venv/Scripts/activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

Train the model:

```
python train.py
```

Run inference on an image:

```
python predict.py --image path/to/image.jpg
```

## Structure

* `animalvision/config.py` - Hyperparameters and paths
* `animalvision/data.py` - Dataloaders and transforms
* `animalvision/model.py` - ResNet18 model setup
* `animalvision/utils.py` - Metrics and plotting utilities
* `train.py` - Main training loop and checkpoint saving
* `predict.py` - Single image inference script