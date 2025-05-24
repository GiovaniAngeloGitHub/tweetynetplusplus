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

1. Configure your training parameters in the `configs` directory.

2. Run the training script:
```bash
python main.py --config configs/your_config.yaml
```

## Development

To install development dependencies:
```bash
uv pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

## Project Structure

```
tweetynetplusplus/
├── configs/           # Configuration files
├── data/             # Dataset directory
├── logs/             # Training logs
├── models_checkpoints/ # Saved model checkpoints
├── src/              # Source code
│   └── tweetynetplusplus/
├── tests/            # Test files
├── main.py           # Main entry point
└── pyproject.toml    # Project dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Original Tweetynet implementation by David Nicholson and colleagues
  - Paper: Nicholson, D. P., & Cohen, Y. (2019). Tweetynet: A neural network for segmenting and classifying birdsong syllables. Journal of the Acoustical Society of America, 145(3), 1827-1838.
  - GitHub: [Original Tweetynet Repository](https://github.com/NickleDave/tweetynet)
- VAK framework for automated annotation of vocalizations
- All contributors and users of this project

## Citation

If you use this software in your research, please cite both the original Tweetynet paper and this implementation:

```bibtex
@article{nicholson2019tweetynet,
  title={Tweetynet: A neural network for segmenting and classifying birdsong syllables},
  author={Nicholson, David P and Cohen, Yarden},
  journal={Journal of the Acoustical Society of America},
  volume={145},
  number={3},
  pages={1827--1838},
  year={2019},
  publisher={Acoustical Society of America}
}
```
