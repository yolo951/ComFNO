

# Component Fourier Neural Operator for Singularly Perturbed Differential Equations

## Overview

This repository contains the implementation and experimental code for the paper "Component Fourier Neural Operator for Singularly Perturbed Differential Equations". Our work presents a novel approach to solving singularly perturbed differential equations using machine learning techniques.

## Repository Structure

The repository is organized into several directories, each corresponding to a specific experimental scenario:

-  `1d/`: One-dimensional convection-diffusion problem
- `2d/`: Two-dimensional convection-diffusion problem
- `fewshot/`: One-dimensional problem in few-shot learning scenarios
- `initial_boundary/`: One-dimensional initial boundary problem
- `multiple_eps/`: One-dimensional problem with multiple distribution
- `turning_point/`: One-dimensional problem with turning point

Each experiment folder contains the corresponding code, data files.

## Requirements

[List of required libraries, e.g.,]
- Python
- PyTorch
- NumPy
- Matplotlib

## Usage

For example:
```bash
cd 1d
python spno_1.py