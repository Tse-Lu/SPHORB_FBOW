Train a [FBOW](https://github.com/rmsalinas/fbow) bag-of-words dictionary using [SPHORB](https://github.com/tdsuper/SPHORB) panoramic image feature detector.


# Usage
The executable `train_sporb_fbow` will be generated in the `build/` directory.

```
train_sporb_fbow [options] -s <sphorb_path> -o <output.fbow> <dir1> [dir2 ...]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `-s, --sphorb <path>` | **(Required)** Path to the SPHORB folder. Must contain `Data/`. |
| `-o, --output <path>` | **(Required)** Output vocabulary file (`.fbow`). |
| `<dir1> [dir2 ...]` | One or more image directories. All images are collected **recursively**. |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-k <int>` | 10 | Branching factor (number of children per node). |
| `-l <int>` | 6 | Maximum tree depth. Total vocabulary size ≈ k^L. |
| `-t <int>` | 1 | Number of threads for parallel clustering. |
| `-n <int>` | 500 | Maximum number of keypoints to retain per image. |
| `--max-iters <int>` | 0 | Maximum K-means iterations. 0 = use random centers without iteration (fast, nearly as good). |
| `-v, --verbose` | off | Enable verbose output during training. |

# Example
We provide the `output.fbow` file trained on [RflyPano](https://ojs.aaai.org/index.php/AAAI/article/view/38884), [360DVO](https://arxiv.org/abs/2601.02309), and [CVPG](https://link.springer.com/article/10.1007/s11760-021-02003-3) datasets ($k=10, L=6$) with 158,790 panoramic images ($1920\times 640$).