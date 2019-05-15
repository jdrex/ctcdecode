# ctcdecode

ctcdecode is an implementation of CTC (Connectionist Temporal Classification) beam search decoding for PyTorch.
C++ code borrowed liberally from Paddle Paddles' [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech).
It includes swappable scorer support enabling standard beam search, and KenLM-based decoding.

This fork implements the subword prefix beam search decoding algorithm described in:

Drexler, Jennifer, and James Glass. "Subword Regularization and Beam Search Decoding for End-to-end Automatic Speech Recognition." IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2019.

## Installation
The library is largely self-contained and requires only PyTorch and CFFI. Building the C++ library requires gcc or clang. KenLM language modeling support is also optionally included, and enabled by default.

```bash
# get the code
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```
