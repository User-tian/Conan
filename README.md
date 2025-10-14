# Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This is the official implementation of our ASRU 2025 paper "Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion"
![Architecture](figs/arch.png)
Zero-shot online voice conversion (VC) holds significant promise for real-time communications and entertainment. 
However, current VC models struggle to preserve semantic fidelity under real-time constraints, deliver natural-sounding conversions, and adapt effectively to unseen speaker characteristics.
To address these challenges, we introduce Conan, a chunkwise online zero-shot voice conversion model that preserves the content of the source while matching the speaker representation of reference speech.
Conan comprises three core components: 
1) A Stream Content Extractor that leverages Emformer for low-latency streaming content encoding; 
2) An Adaptive Style Encoder that extracts fine-grained stylistic features from reference speech for enhanced style adaptation; 
3) A Causal Shuffle Vocoder that implements a fully causal HiFiGAN using a pixel-shuffle mechanism. 
Experimental evaluations demonstrate that Conan outperforms baseline models in subjective and objective metrics.

## 🌟 Features

- **Streaming Voice Conversion**: Real-time voice conversion with low latency (~80ms)
- **Emformer Integration**: Efficient transformer-based content encoding
- **High-Quality Vocoding**: Pixel-shuffle  causal HiFi-GAN vocoder for natural-sounding audio output

## Workflow
Our workflow (inference procedure) is shown in the following figure.
![Workflow](figs/intro.png)
we first feed the entire reference speech into the model to provide timbre
and stylistic information. During chunkwise online inference,
we wait until the input reaches a predefined chunk size before
passing it to the model. Because our generation speed for each
chunk is faster than the chunk’s duration, online generation
becomes possible. To ensure temporal continuity, we employ
a sliding context window strategy. At each generation step,
we not only input the source speech of the current chunk but
also include the preceding context. From the model’s output,
we extract only the segment for this chunk. As the context
covers the receptive field, consistent overlapping segments can
be generated, ensuring smooth transitions at chunk boundaries.

## 📋 Requirements

### System Requirements
- Python 3.10+

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/User-tian/Conan.git
cd Conan
```

2. **Create a virtual environment**:
```bash
conda create -n conan python=3.10
conda activate conan
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```
## 📊 Data Preparation

### Dataset Structure
You only need to prepare the metadata.json file in the `data/processed/` directory.
```
data/
└── processed/
    ├── metadata.json
    └── spker_set.json
```
### Metadata Format
There is an example "example_metadata.json" file in the `data/processed/vc/` directory.
The `metadata.json` file should contain entries like:
```json
[
  {
    "item_name": "speaker1_audio1",
    "wav_fn": "data/raw/speaker1/audio1.wav", // Path to the raw audio file
    "spk_embed": "0.1 0.2 0.3 ...", // Speaker embedding vector
    "duration": 3.5, // Duration in seconds
    "hubert": "12 34 56 ..." // HuBERT features as space-separated string
  }
]
```

### Data Preprocessing Steps

1. **Extract F0 features using RMVPE (needed only for main model training)**:
```bash
export PYTHONPATH=/storage/baotong/workspace/Conan:$PYTHONPATH # (optional) you may need to set the PYTHONPATH for import dependencies
python trials/extract_f0_rmvpe.py \
    --config  /u/usertian/workspace/streamvc/egs/conan.yaml \
    --batch-size 80 \
    --save-dir /path/to/audio  
```
F0 will be saved to the same level folder as the audio folder.
File structure: (an example below)
```data/
└── audio/
    ├── p225_001.wav
    ├── ...
└── audio_f0/
    ├── p225_001.npy
    ├── ...
```
2. **Binarize the dataset**:
```bash
python data_gen/tts/runs/binarize.py --config egs/conan.yaml
```
(You can use this config for all 3-stage training binarization)
### Configuration
Update the configuration files in `egs/` directory to match your dataset:
- `egs/conan_emformer.yaml`: Main training configuration
- `egs/emformer.yaml`: Emformer training configuration
- `egs/hifi_16k320_shuffle.yaml`: Vocoder training configuration

Key parameters to adjust:
```yaml
# Dataset paths
binary_data_dir: 'data/binary/vc'
processed_data_dir: 'data/processed/vc'
```
## 🎯 Training

### Stage 1: Train Emformer
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/emformer.yaml \
    --exp_name emformer_training \
    --reset
```

### Stage 2: Train Main Conan Model
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/conan_emformer.yaml \
    --exp_name conan_training \
    --reset
```

### Stage 3: Train HiFi-GAN Vocoder
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/hifi_16k320_shuffle.yaml \
    --exp_name hifigan_training \
    --reset
```

## 🔮 Inference

### Streaming Voice Conversion
```bash
CUDA_VISIBLE_DEVICES=0 python inference/Conan.py \
    --config egs/conan_emformer.yaml \
    --exp_name conan
```
Use the exp_name that contains the trained main model checkpoints, and update your config with the trained Emformer checkpoint and HifiGAN checkpoint.

## Checkpoints
You can download pre-trained model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1QhnECo2L4xfXDgdrnM6L1xpsH7u3iRvj?usp=sharing).

Note: As we previous developed the Emformer training branch on another codebase, we provided another inference script for it `inference/Conan_previous.py`.
## 📁 Project Structure

```
Conan/
├── modules/                    # Core model implementations
│   ├── Conan/                 # Main Conan model
│   ├── Emformer/              # Emformer feature extractor
│   ├── vocoder/               # HiFi-GAN vocoder
│   └── ...
├── tasks/                     # Training and evaluation tasks
│   ├── Conan/                 # Conan training task
│   └── ...
├── inference/                 # Inference scripts
│   ├── Conan.py              # Main inference script
│   ├── run_voice_conversion.py
│   └── ...
├── data_gen/                  # Data preprocessing
│   ├── conan_binarizer.py    # Data binarization
│   └── ...
├── egs/                       # Configuration files
│   ├── conan.yaml           # Main training config
│   ├── emformer.yaml         # Emformer config
│   └── ...
├── utils/                     # Utility functions
└── checkpoints/              # Model checkpoints
```
## 📈 Performance

The Conan system achieves state-of-the-art performance on voice conversion tasks:

- **Latency**: ~80ms streaming latency
- **Quality**: High-quality voice conversion with natural prosody
- **Robustness**: Robust to different speaking styles and content

## 📄 Citation

If you use Conan in your research, please cite our work:

```bibtex
@article{conan2024,
  title={Conan: Streaming Voice Conversion with Diffusion Models},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [FastSpeech2](https://github.com/ming024/FastSpeech2) for the codebase and base TTS architectures
- [HiFi-GAN](https://github.com/jik876/hifi-gan) for the neural vocoder
- [Emformer](https://github.com/pytorch/audio) for efficient transformer implementation
