# StyleGAN Implementation

![StyleGAN Architecture](gan.png)

## Features

- Full implementation of the StyleGAN architecture
- Support for custom datasets
- Progressive growing of GAN layers for stable training on high-resolution images
- Pre-trained weights for 64x64 resolution
- Tensorboard integration for comprehensive logging

## Prerequisites

Ensure you have the following installed on your system:

- Python 3.7+
- PyTorch
- NVIDIA CUDA (for GPU acceleration)
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/stylegan-implementation.git
   cd stylegan-implementation
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Training

1. Prepare your dataset and update the dataset path in `train.py`
2. Adjust hyperparameters in the train.py as needed
3. Start training:
   ```
   python train.py
   ```

## Results

![Sample Generated Images](test_images.png)

These images are generated at 64x64 resolution using local training. Image quality can be further improved with extended training periods and higher-resolution datasets.

## Roadmap

- [x] Initial implementation
- [ ] Style mixing
- [ ] Style transfer

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- The original StyleGAN paper: [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
- NVIDIA for their work on progressive growing of GANs

