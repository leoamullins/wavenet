# wavenet

An unconditional WaveNet trained on LJSpeech at 8 kHz with 8-bit mu-law targets.

## Layout

```text
wavenet/          library code
  data.py         mu-law, clip loading, preprocessing, AudioChunks dataset
  model.py        CausalConv1d, ResidualBlock, WaveNet
preprocess.py     builds data/{train,val}_8000.pt from the raw LJSpeech wavs
train.py          training loop, saves the best checkpoint by val loss
notebooks/        exploration (data inspection, model experiments)
data/             LJSpeech + preprocessed tensors (gitignored)
checkpoints/      saved models (gitignored)
```

## Setup

```sh
uv sync
```

Download [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) and extract it to `data/LJSpeech-1.1/`, then:

```sh
uv run python preprocess.py              # ~5% of clips held out for validation
uv run python train.py --max-steps 5000
```

Run `uv run python train.py --help` to see the model and training options. Checkpoints store the model config
alongside the weights:

```python
ck = torch.load("checkpoints/ckpt.pt")
model = WaveNet(**ck["config"])
model.load_state_dict(ck["model"])
```
