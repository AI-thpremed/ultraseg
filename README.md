# UltraSeg

&gt; **Ultra-lightweight real-time polyp segmentation for CPU-only deployment**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


## Overview

UltraSeg establishes the first strong baseline for **extreme-compression** polyp segmentation (&lt;0.3M parameters), delivering **&gt;90 FPS on single-core CPU** with clinically viable accuracy.

- **UltraSeg-108K** (0.108M params): Optimized for single-center/single-modality data
- **UltraSeg-130K** (0.13M params): Enhanced for multi-center/multi-modal generalization

## Key Results

| Model | Params | Avg Dice | Single-Core FPS |
|-------|--------|----------|-----------------|
| UNet-Base | 31.0M | 0.839 | 1.6 |
| **UltraSeg-108K** | **0.108M** | **0.784** | **92.1** |
| **UltraSeg-130K** | **0.130M** | **0.793** | **90.3** |

*Evaluated on CVC-ClinicDB, Kvasir, PolypGen, PolypDB, and Kvasir-Instrument datasets.*

## Quick Start

```bash
git clone https://github.com/AI-thpremed/ultraseg.git
cd ultraseg


