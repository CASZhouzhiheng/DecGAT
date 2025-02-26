# DecGAT


## Overview
This repository contains the implementation of the DecGAT model using PyTorch. The model is specifically designed for signed link prediction and is applied in the paper titled "**[Decoupled Signed Link Prediction Method Based on Graph Neural Network]**" For detailed information, please refer to the paper.

## Paper Reference
If you use or refer to this DecGAT model in your work, please cite the following paper:
"**[Decoupled Signed Link Prediction Method Based on Graph Neural Network]**"

## Requirements
PyTorch,NumPy

## Example Usage
```python
# Importing the DecGAT model
from DecGAT import DecGAT

# Creating an instance of the DecGAT model
model = DecGAT()

...

# Forward pass
output = model(feature)
