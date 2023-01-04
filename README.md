<p align="center">
    <a target="_blank" href="https://colab.research.google.com/github/bshall/soft-vc/blob/main/soft-vc-demo.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</p>

# Acoustic-Model

Training and inference scripts for the acoustic models in [A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion](https://ieeexplore.ieee.org/abstract/document/9746484). For more details see [soft-vc](https://github.com/bshall/soft-vc). Audio samples can be found [here](https://bshall.github.io/soft-vc/). Colab demo can be found [here](https://colab.research.google.com/github/bshall/soft-vc/blob/main/soft-vc-demo.ipynb).

<div align="center">
    <img width="100%" alt="Soft-VC"
      src="https://raw.githubusercontent.com/bshall/acoustic-model/main/acoustic-model.png">
</div>
<div>
  <sup>
    <strong>Fig 1:</strong> Architecture of the voice conversion system. a) The <strong>discrete</strong> content encoder clusters audio features to produce a sequence of discrete speech units. b) The <strong>soft</strong> content encoder is trained to predict the discrete units. The acoustic model transforms the discrete/soft speech units into a target spectrogram. The vocoder converts the spectrogram into an audio waveform.
  </sup>
</div>

## Example Usage

### Programmatic Usage

```python
import torch
import numpy as np

# Load checkpoint (either hubert_soft or hubert_discrete)
acoustic = torch.hub.load("bshall/acoustic-model:main", "hubert_soft").cuda()

# Load speech units
units = torch.from_numpy(np.load("path/to/units"))

# Generate mel-spectrogram
mel = acoustic.generate(units)
```

### Script-Based Usage

```
usage: generate.py [-h] {soft,discrete} in-dir out-dir

Generate spectrograms from input speech units (discrete or soft).

positional arguments:
  {soft,discrete}  available models (HuBERT-Soft or HuBERT-Discrete)
  in-dir           path to the dataset directory.
  out-dir          path to the output directory.

optional arguments:
  -h, --help       show this help message and exit
```

## Training

### Step 1: Dataset Preparation

Download and extract the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset. The training script expects the following tree structure for the dataset directory:

```
└───wavs
    ├───dev
    │   ├───LJ001-0001.wav
    │   ├───...
    │   └───LJ050-0278.wav
    └───train
        ├───LJ002-0332.wav
        ├───...
        └───LJ047-0007.wav
```

The `train` and `dev` directories should contain the training and validation splits respectively. The splits used for the paper can be found [here](https://github.com/bshall/acoustic-model/releases/tag/v0.1).

### Step 2: Extract Spectrograms

Extract mel-spectrograms using the `mel.py` script:

```
usage: mels.py [-h] in-dir out-dir

Extract mel-spectrograms for an audio dataset.

positional arguments:
  in-dir      path to the dataset directory.
  out-dir     path to the output directory.

optional arguments:
  -h, --help  show this help message and exit
```

for example:

```
python mel.py path/to/LJSpeech-1.1/wavs path/to/LJSpeech-1.1/mels
```

At this point the directory tree should look like:

```
├───mels
│   ├───...
└───wavs
    ├───...
```

### Step 3: Extract Discrete or Soft Speech Units

Use the HuBERT-Soft or HuBERT-Discrete content encoders to extract speech units. First clone the [content encoder repo](https://github.com/bshall/hubert) and then run `encode.py` (see the repo for details):

```
usage: encode.py [-h] [--extension EXTENSION] {soft,discrete} in-dir out-dir

Encode an audio dataset.

positional arguments:
  {soft,discrete}       available models (HuBERT-Soft or HuBERT-Discrete)
  in-dir                path to the dataset directory.
  out-dir               path to the output directory.

optional arguments:
  -h, --help            show this help message and exit
  --extension EXTENSION
                        extension of the audio files (defaults to .flac).
```

for example:

```
python encode.py soft path/to/LJSpeech-1.1/wavs path/to/LJSpeech-1.1/soft --extension .wav
```

At this point the directory tree should look like:

```
├───mels
│   ├───...
├───soft/discrete
│   ├───...
└───wavs
    ├───...
```

### Step 4: Train the Acoustic-Model

```
usage: train.py [-h] [--resume RESUME] [--discrete] dataset-dir checkpoint-dir

Train the acoustic model.

positional arguments:
  dataset-dir      path to the data directory.
  checkpoint-dir   path to the checkpoint directory.

optional arguments:
  -h, --help       show this help message and exit
  --resume RESUME  path to the checkpoint to resume from.
  --discrete       Use discrete units.
```

## Links

- [Soft-VC repo](https://github.com/bshall/soft-vc)
- [Soft-VC paper](https://ieeexplore.ieee.org/abstract/document/9746484)
- [HuBERT content encoders](https://github.com/bshall/hubert)
- [HiFiGAN vocoder](https://github.com/bshall/hifigan)

## Citation

If you found this work helpful please consider citing our paper:

```
@inproceedings{
    soft-vc-2022,
    author={van Niekerk, Benjamin and Carbonneau, Marc-André and Zaïdi, Julian and Baas, Matthew and Seuté, Hugo and Kamper, Herman},
    booktitle={ICASSP}, 
    title={A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion}, 
    year={2022}
}
```



[] Hirokazu Kameoka, Takuhiro Kaneko, Kou Tanaka, and Nobukatsu Hojo, “Stargan-vc: Non-parallel many-tomany voice conversion using star generative adversarial networks,” in SLT, 2018.

idea ====> By using the parallel method, it is possible to identify each person's voice completely separately in a conversation two by two.


[] Adam Polyak et al., “Speech resynthesis from discrete disentangled self-supervised representations,” in Interspeech, 2021.

idea ====> The system automatically combines multiple conversations and uses artificial intelligence to obtain a unique output from the combination of multiple conversations.


[] Aaron van den Oord, Yazhe Li, and Oriol Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.
[] Tu Anh Nguyen et al., “The zero resource speech benchmark 2021: Metrics and baselines for unsupervised spoken language modeling,” in NeurIPS SAS Workshop,
2020.

idea ====> An artificial intelligence that has the ability to convert words, images and 3D text to each other.


[20] Julian Za¨ıdi, Hugo Seute, Benjamin van Niekerk, and ´Marc-Andre Carbonneau, “Daft-exprt: Robust prosody ´transfer across speakers for expressive speech synthesis,” arXiv preprint arXiv:2108.02271, 2021.

idea ====> The speaker (tone) can distinguish between strong and weak sound. Make the louder text bigger and the weaker text thinner.
Recognize different sounds that have different wavelengths in a speech.

[] Tomoki Toda, Alan W Black, and Keiichi Tokuda, “Voice conversion based on maximum-likelihood estimation of spectral parameter trajectory,” TASLP, vol. 15, no. 8,
2007.

[] Kou Tanaka, Hirokazu Kameoka, Takuhiro Kaneko, and Nobukatsu Hojo, “Atts2s-VC: Sequence-to-sequence voice conversion with attention and context preservation mechanisms,” in ICASSP, 2019.


[] Lifa Sun, Kun Li, Hao Wang, Shiyin Kang, and Helen Meng, “Phonetic posteriorgrams for many-to-one voice conversion without parallel data training,” in ICME, 2016.


[] Wen-Chin Huang, Tomoki Hayashi, Shinji Watanabe, and Tomoki Toda, “The Sequence-to-Sequence Baseline for the Voice Conversion Challenge 2020: Cascading ASR and TTS,” in Interspeech BC/VCC workshop, 2020.


[] Hirokazu Kameoka, Takuhiro Kaneko, Kou Tanaka, and Nobukatsu Hojo, “Stargan-vc: Non-parallel many-tomany voice conversion using star generative adversarial networks,” in SLT, 2018.


[] Kaizhi Qian, Yang Zhang, Shiyu Chang, Xuesong Yang, and Mark Hasegawa-Johnson, “AutoVC: Zero-shot voice style transfer with only autoencoder loss,” in ICML, 2019.


[] Yi Zhao et al., “Voice conversion challenge 2020: Intralingual semi-parallel and cross-lingual voice conversion,” in Interspeech BC/VCC workshop, 2020.


[] Adam Polyak et al., “Speech resynthesis from discrete disentangled self-supervised representations,” in Interspeech, 2021.


[] Benjamin van Niekerk, Leanne Nortje, and Herman Kamper, “Vector-quantized neural networks for acoustic unit discovery in the zerospeech 2020 challenge,” in Interspeech, 2020.


[] Wen-Chin Huang, Yi-Chiao Wu, and Tomoki Hayashi, “Any-to-one sequence-to-sequence voice conversion using self-supervised discrete speech representations,” in
ICASSP, 2021.


[] Wei-Ning Hsu et al., “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,” arXiv preprint arXiv:2106.07447, 2021.


[] Jan van Gemert, Cor J. Veenman, Arnold W. M. Smeulders, and Jan-Mark Geusebroek, “Visual Word Ambiguity,” TPAMI, vol. 32, no. 7, 2010.


[] Tu Anh Nguyen et al., “The zero resource speech benchmark 2021: Metrics and baselines for unsupervised spoken language modeling,” in NeurIPS SAS Workshop,
2020.


[] Benjamin van Niekerk, Leanne Nortje, Matthew Baas, and Herman Kamper, “Analyzing speaker information in self-supervised models to improve zero-resource speech
processing,” in Interspeech, 2021.


[] Shu wen Yang et al., “Superb: Speech processing universal performance benchmark,” in Interspeech, 2021.


vahid  sepehrian       oloom tahghighat     40012340048013
