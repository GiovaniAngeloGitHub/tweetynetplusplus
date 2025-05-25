# Tweetynet++

Tweetynet++ is a deep learning framework for birdsong analysis and segmentation. This project extends the capabilities of the original Tweetynet model, providing enhanced tools for analyzing and processing birdsong data.

## About the Original Tweetynet

This project is based on the original Tweetynet implementation by David Nicholson and colleagues. Tweetynet is a deep neural network architecture specifically designed for segmenting and classifying birdsong syllables. The original implementation can be found in the following paper:

Nicholson, D. P., & Cohen, Y. (2019). Tweetynet: A neural network for segmenting and classifying birdsong syllables. Journal of the Acoustical Society of America, 145(3), 1827-1838.

Key features of the original Tweetynet:
- Convolutional neural network architecture optimized for birdsong analysis
- Efficient syllable segmentation and classification
- Real-time processing capabilities
- Integration with the VAK framework for automated annotation

## Features

- Deep learning-based birdsong analysis
- Audio segmentation capabilities
- Integration with the VAK framework
- Support for various audio processing tasks

## Requirements

- Python >= 3.10
- CUDA-compatible GPU (recommended for training)

## Installation

### Using UV (Recommended)

1. First, make sure you have UV installed. If not, install it using:
```bash
pip install uv
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/tweetynetplusplus.git
cd tweetynetplusplus
```

3. Create and activate a virtual environment using UV:
```bash
uv venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

4. Install the package and its dependencies:
```bash
uv pip install -e .
```

### Alternative Installation (using pip)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tweetynetplusplus.git
cd tweetynetplusplus
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install the package:
```bash
pip install -e .
```

## Dataset Preparation

1. Download your birdsong dataset and organize it in the following structure:
```
data/
├── raw/
│   └── your_dataset/
│       ├── audio_files/
│       └── annotations/
└── processed/
```

2. Process your dataset using the provided scripts in the `src/tweetynetplusplus` directory.


## Usage

### Configuration

All training and evaluation parameters are defined using `.toml` configuration files inside the `configs/` directory.

- `default.toml`: defines base parameters for training, evaluation, data and logging.
- `experimento_base.toml`: overrides specific parameters such as learning rate, epochs, etc.

#### Example: `configs/default.toml`

```toml
[model]
name = "resnet18"
pretrained = true

[data]
processed_dir = "data/processed/llb11"
annotation_file = "data/raw/llb11/llb11_annot.csv"
batch_size = 8

[training]
epochs = 30
learning_rate = 1e-4
weight_decay = 1e-4
patience = 5
use_saved_model = false

[logging]
model_dir = "models_checkpoints"
log_dir = "logs"
```

### Training

To train a model using the combined configuration:

```bash
python src/tweetynetplusplus/scripts/train_model.py
```

This script will:

- Load `default.toml`
- Merge it with `experimento_base.toml`
- Run training with logging, checkpoint saving, and early stopping

### Evaluation

To evaluate a trained model and save metrics:

```bash
python src/tweetynetplusplus/scripts/evaluate_model.py
```

This will:

- Load a saved model (edit the script to specify the `.pt` file)
- Evaluate on the validation/test sets
- Save classification report to `logs/classification_report.csv`
- Save confusion matrix image to `logs/eval_confusion_matrix.png`
- Export raw predictions to `logs/y_true.npy` and `logs/y_pred.npy`

### Report Generation

Once evaluation is complete, generate a CSV/JSON report with:

```bash
python src/tweetynetplusplus/scripts/generate_report.py
```

---

## 🧪 Running Tests

Unit tests are located in `tests/` and can be run via:

```bash
pytest
```