# cmpt419-project

## Table of Contents

- [Project Info](#project-info)
- [Contributers](#contributers)
- [Folder Structure](#folder-structure)
- [Installation](#installation)

## Project Info
**RageRadar**: A binary classifier using a multimodal deep learning architecture, used to identify rage using audio and image sequences.

The dataset includes video and audio samples labeled as Rage, Fear, Frustration, or Excitement.
Each sample is relabeled as Rage (0) or Not Rage (1) in the model.

Audio preprocessing:
.wav files are loaded and converted to Mel spectrograms of size [1, 64, 64]. Audio is normalized to a consistent length of 160,000 samples.

Video preprocessing:
20 evenly spaced grayscale frames. Each frame is resized to 112×112 and stacked into tensors of shape [20, 1, 112, 112].

Model Architecture
- VideoCNN: a modified ResNet18 processes grayscale image sequences by extracting per-frame features and averaging them temporally.
- AudioCNN: a custom CNN processes 2D Mel spectrograms into a feature vector.
- LateFusion: combines predictions from both models using an MLP that fuses audio and video outputs into a final binary decision.

Training: trained using CrossEntropyLoss with the Adam optimizer (lr = 1e-4) for 20 epochs.
Metrics tracked include loss and binary accuracy. Training data is augmented with robust error handling to account for missing or corrupted media.

## Contributers:
- Eric 
- Sepher
- Trevor

## Folder Structure
- .
- **process_data.ipynb**: Run this file first, splits data into their respective directories
- **train_test_models**: Run this file second, this is our ML model.
- **rage_detection_fusion_train.pt**: Pre-saved training model
- **rage_detection_fusion_test.pt**: Pre-saved testing model
- **data**: Directory containing data files.
  - **train**: Directory for training data.
    - **trainRage**: Directory for rage data for training
    - **trainFear**: Directory for fear data for training
    - **trainExcitement**: Directory for excitement data for training
    - **trainFrustration**: Directory for frustration data for training
    - **trainAudio**: Directory for audio data for training
  - **test**: Directory for testing data.
    - **testRage**: Directory for rage data for testing
    - **testFear**: Directory for fear data for testing
    - **testExcitement**: Directory for excitement data for testing
    - **testFrustration**: Directory for frustration data for testing
    - **testAudio**: Directory for audio data for testing

## Installation

[Download our dataset here](https://drive.google.com/file/d/1EtoDBcclW2dRN0Sa25p2T6bWJ2Pughg6/view?usp=sharing)

Via https
```bash
git clone https://github.com/ericw5134/cmpt419-project.git
cd cmpt419-project
```