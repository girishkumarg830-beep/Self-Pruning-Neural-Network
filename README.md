# Self-Pruning Neural Network: Dynamic Weight Sparsification

### Project Overview
This repository contains an implementation of a neural network that learns to prune its own weights during training. Instead of traditional post-training pruning, this model uses learnable "gate" parameters and L1 regularization to identify and remove redundant connections on the fly.

### 📁 Project Structure
- `self_pruning_network.ipynb`: Main notebook with training logs and visual analysis.
- `self_pruning_network.py`: Modular Python script containing the `PrunableLinear` class.
- `gate_distribution.png`: Histogram showing the polarization of active vs. pruned weights.
- `report.md`: Detailed technical analysis of the experiment.

### 🚀 How to Run
1. Open the `self_pruning_network.ipynb` in **Google Colab**.
2. Go to `Runtime` > `Change runtime type` and select **T4 GPU**.
3. Run all cells to execute the hyperparameter sweep ($10^{-6}, 5 \times 10^{-5}, 10^{-4}$).

### 🛠️ Key Technologies
- **PyTorch**: Custom `nn.Module` development.
- **CIFAR-10**: Image classification dataset.
- **Regularization**: L1 Sparsity penalty.
