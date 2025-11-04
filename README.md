# MNIST Digit Classifier with WebGPU

![WebGPU](https://img.shields.io/badge/WebGPU-Enabled-brightgreen) ![Tinygrad](https://img.shields.io/badge/Framework-Tinygrad-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Live Demo

[Access the live application here](https://f0rtin.github.io/Web_App_Mnist/)

Draw a digit on the canvas and experience real-time predictions powered by GPU acceleration.

---

## Overview

This project demonstrates a production-ready digit classification system that runs entirely in modern web browsers. Built with **Tinygrad** for efficient model training and **WebGPU** for browser-based GPU acceleration, it showcases how advanced machine learning can be seamlessly integrated into web applications.

The application features two distinct neural network architectures—a lightweight MLP and a high-performance CNN—both optimized for inference speed and accuracy. Users can interactively draw digits and receive instant predictions with confidence scores, all without requiring server-side computation.

---

## Key Features

- Real-time digit prediction with sub-100ms inference latency
- GPU-accelerated inference through WebGPU API
- Dual model architecture with model selection
- Probability visualization for all 10 digit classes
- Responsive, intuitive drawing interface
- Advanced training with geometric data augmentation
- Smart early stopping with adaptive learning rate decay
- Models exported to WebGPU-compatible format

---

## Model Architecture

| Model | Layers | Parameters | Test Accuracy |
|-------|--------|---|---|
| MLP | Dense(784→128→64→10) | ~109K | 97.5% |
| CNN | Conv→Conv→Pool→Conv→Conv→Pool→Dense | ~44K | 99.2% |

### Convolutional Neural Network

The CNN architecture provides superior accuracy through hierarchical feature extraction:

```
Input Layer: 28x28x1
├── Conv2d(1→32, kernel=5) + SiLU activation
├── Conv2d(32→32, kernel=5) + SiLU + BatchNorm
├── MaxPooling2d
├── Conv2d(32→64, kernel=3) + SiLU activation
├── Conv2d(64→64, kernel=3) + SiLU + BatchNorm
├── MaxPooling2d
├── Flatten layer
└── Dense(576→10)
Output: 10-class softmax
```

---

## Getting Started

### Requirements

- Python 3.8 or higher
- Git
- Modern browser with WebGPU support (Chrome 113+, Edge 113+)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Web_App_Mnist.git
   cd Web_App_Mnist
   ```

2. Install dependencies:
   ```bash
   pip install tinygrad numpy
   ```

3. Train models (optional):
   ```bash
   cd training
   python mnist_convnet.py
   python mnist_mlp.py
   ```

4. Start a local development server:
   ```bash
   cd docs
   python -m http.server 8000
   ```

5. Open http://localhost:8000 in your browser

---

## Project Structure

```
Web_App_Mnist/
├── docs/
│   ├── index.html
│   └── models/
│       ├── mnist_mlp/
│       │   ├── mnist_mlp.js
│       │   └── mnist_mlp.webgpu.safetensors
│       └── mnist_convnet/
│           ├── mnist_convnet.js
│           └── mnist_convnet.webgpu.safetensors
├── training/
│   ├── mnist_convnet.py
│   ├── mnist_mlp.py
│   ├── export_model.py
│   └── tmp_save/
├── README.md
├── HYPERPARAMETERS.md
└── .gitignore
```

---

## Training Configuration

Detailed hyperparameter specifications and training methodology are documented in [HYPERPARAMETERS.md](./HYPERPARAMETERS.md).

Primary training settings:
- Batch Size: 512
- Initial Learning Rate: 0.02 (exponential decay: 0.9x)
- Optimizer: Muon
- Early Stopping Patience: 50 epochs
- Data Augmentation: Rotation (±15°), Scale (±10%), Translation (±10%)

---

## Usage

1. Select your preferred model from the dropdown menu
2. Allow 2-5 seconds for model compilation to WebGPU
3. Draw a digit (0-9) on the canvas using your mouse
4. View the predicted digit and confidence distribution
5. Click "Clear" to draw another digit

---

## Technical Stack

- **Tinygrad**: Minimalist deep learning framework with automatic differentiation
- **WebGPU**: Hardware-accelerated compute API for modern browsers
- **HTML5 Canvas**: Graphics rendering for drawing interface
- **Tailwind CSS**: Responsive UI framework
- **ES6 Modules**: Dynamic JavaScript module loading

---

## Data Augmentation Strategy

Training incorporates geometric transformations to enhance model generalization:

- Rotation: Random angles between -15 and +15 degrees
- Scaling: Scale factors between 0.9 and 1.1
- Translation: Pixel shifts between -10% and +10% in both axes

Augmentation uses configurable interpolation (nearest-neighbor or bilinear) to maintain training stability.

---

## Performance Metrics

- Average Inference Time: 50-100ms (first run includes JIT compilation)
- Subsequent Predictions: 5-20ms
- Model Size: CNN (~200KB) + MLP (~100KB)
- Peak Test Accuracy: 99.2% (CNN)

---

## Troubleshooting

**Model Loading Issues**
- Verify WebGPU support in your browser (try Chrome Canary or Edge Beta)
- Check browser console (F12) for specific error details
- Clear browser cache or use incognito mode

**File Not Found Errors**
- Confirm `/docs/models/` directory structure matches documentation
- Regenerate models by running training scripts
- Verify relative paths in index.html match actual file locations

**Performance Concerns**
- Initial model loading includes WebGPU shader compilation (expected delay)
- Subsequent inferences benefit from compiled shader cache
- Check browser GPU utilization in DevTools Performance tab

---

## License

MIT License - See LICENSE file for details.

---

## About

This project was developed as part of a Creative Technologies program at ESILV, demonstrating practical applications of machine learning in modern web environments.

For questions or contributions, please open an issue or submit a pull request.
