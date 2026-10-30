# ECE 285 Homework 1: Chest X-ray VAE

[Open the notebook in Colab](https://colab.research.google.com/github/Judy-LIU-Judy/ECE285-HW1-VAE/blob/main/HW1_VAE.ipynb)

A convolutional variational autoencoder implemented in PyTorch for unconditional generation
of chest X-ray images. The experiments compare beta regularization, decoder architecture,
continued training, and an additional lower-beta sweep.

## Files

- `HW1_VAE.ipynb`: main implementation, configuration, training, evaluation, recorded tables,
  loss curves, prior samples, and reconstructions. Figures are embedded in the notebook.
- `hw1_beta_sweep.py`: optional additional controlled sweep with beta 0.1, 0.3, and 1.0.
- `requirements.txt`: recorded PyTorch/torchvision versions and other dependencies.
- `experiment_records/original/`: original configurations, full histories, split manifest,
  exploratory test-reference scores, and validation FID records.
- `experiment_records/beta_sweep/`: completed sweep configurations, histories, evaluation
  records, preview filenames, environment information, and completion status.

Dataset images, preprocessing caches, and checkpoint weights are not included.

## View recorded results

Open `HW1_VAE.ipynb` on GitHub to read the implementation, tables, and embedded figures.
The recorded Colab results are clearly distinguished from new runs.

The default notebook settings are:

```python
RUN_TRAINING = False
RUN_EVALUATION = False
RUN_BETA_SWEEP = False
```

Running the notebook with these settings does not train a model or download the dataset.
The dependency cell installs missing extra packages. Torch and torchvision must already be
installed; Colab provides them. Local users can install dependencies with
`python -m pip install -r requirements.txt` and use Jupyter to open the notebook.

## Train and evaluate in Colab

1. Open the notebook in Colab and select a GPU runtime.
2. Set `RUN_TRAINING = True` and `RUN_EVALUATION = True` in Section 1.
3. Run the cells in order and authorize Google Drive mounting when prompted.

The dataset downloads through kagglehub. New runs are saved under
`/content/drive/MyDrive/HW1_VAE/reproduction_runs`, separate from the original experiments.
Each run saves last and best checkpoints and a complete history. Repeating the cells resumes
from the last checkpoint. The beta-4 continuation starts from the beta-4 run's last epoch-20
checkpoint and continues through epoch 40 with its optimizer and RNG state.

To evaluate the original checkpoints instead, keep training disabled and enable only
`RUN_EVALUATION`. The original four experiment folders must contain `best_checkpoint.pt`
under `/content/drive/MyDrive/HW1_VAE`. These weights remain in the original Drive account;
other users should train from scratch with both flags enabled.

Fresh reruns can differ numerically: the original interactive sessions consumed random
numbers during diagnostics, while the organized implementation initializes each run explicitly.
Fresh evaluation writes `reevaluated_validation_metrics.json` and does not replace the recorded
metric files in this repository.

## Optional additional sweep

Upload `hw1_beta_sweep.py` to Colab's runtime Files panel, enable `RUN_BETA_SWEEP`, and run
the notebook. The script trains three standard-decoder models from the same seeded weights
for 40 epochs, caches preprocessed inputs, saves resumable checkpoints, and evaluates all three.
Sweep reruns are stored in `/content/drive/MyDrive/HW1_VAE/reproduction_beta_sweep`.

The sweep resets an independent validation-noise generator each epoch. Original runs used
global posterior sampling RNG; this changes validation checkpoint selection. The generation
FID preprocessing, reference images, and evaluation latent bank are shared across protocols.

## Data and model

- [Kaggle chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
- Seed-42 holdout of 10% of filename groups within each class: 4,699 training and 517 validation
  images. Filename groups do not establish independently verified patient identities.
- Original 16-image validation folder unused; original test folder contains 624 images.
- Grayscale, bilinear resize to 64 x 64, pixel range [0, 1]. No augmentation or label conditioning.
- Three stride-2 convolutional encoder stages; latent dimension 32; standard normal prior.
- Standard decoder: three transposed convolutions. Alternative: nearest-neighbor upsampling
  and 3 x 3 convolutions; its parameter count differs from the standard decoder.
- Loss per image: pixel SSE / (2 * 0.1**2) + beta * KL. The Gaussian constant is omitted.
- Adam, batch 64, learning rate 0.001 for epochs 1-20 and 0.0003 for continuation/sweep epochs 21-40.
- Best checkpoint: lowest validation total loss within each experiment. Total losses with
  different beta values are not directly comparable.

## Recorded validation results

All FID values below use 517 validation images and the first 517 vectors of the same
624-vector seed-2026 latent draw.

| Experiment | Selected epoch | Validation FID |
|---|---:|---:|
| Baseline, beta 1, 20 epochs | 18 | 266.97 |
| Beta 4, 20 epochs | 20 | 240.17 |
| Resize decoder, beta 4, 20 epochs | 19 | 256.52 |
| Beta 4, continued to 40 epochs | 39 | **229.94** |
| Additional sweep, beta 0.1, 40 epochs | 39 | 280.28 |
| Additional sweep, beta 0.3, 40 epochs | 39 | 269.70 |
| Additional sweep, beta 1, 40 epochs | 40 | 254.51 |

The original beta-4 continuation remains best by recorded validation FID. Samples remain
blurred; lowering beta in the additional sweep did not produce a clear visual improvement.
Sweep Inception Scores are 1.4317, 1.4944, and 1.5910 for beta 0.1, 0.3, and 1.0 respectively.
No recorded Inception Score is available for the original beta-4 continuation.

Earlier exploratory test-reference evaluations used 624 real and 624 generated images:
baseline FID 279.2257 / IS 1.3944; beta-4 epoch 20 FID 255.2344 / IS 1.5989.
Those results were observed before later tuning and are not an untouched final test benchmark.
They must not be mixed with the validation-reference results.

## Evaluation pipeline and limits

Grayscale is repeated across RGB channels. Torchvision Inception v3 IMAGENET1K_V1 transforms
resize to 342, center-crop to 299, and apply ImageNet normalization. Its 2,048-dimensional
penultimate features are passed to Ignite FID with an identity extractor. Inception Score uses
the pretrained classifier's softmax probabilities, once over all generated images, with no
split-wise standard deviation.

Scores depend on this feature/preprocessing pipeline and are not interchangeable with other
FID implementations. One seed, a small validation reference, and ImageNet features limit the
conclusions. The results do not establish clinical utility.

## References

- Kingma and Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- [Ignite tutorial: FID and Inception Score](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).
