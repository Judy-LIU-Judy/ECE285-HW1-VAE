"""Automatic controlled beta sweep for the chest X-ray VAE homework.

Run in a Colab GPU runtime. All generated results come from actual execution.
"""
import csv
import gc
import hashlib
import json
import math
import random
import re
import shutil
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SEED = 42
EVAL_SEED = 2026
VALIDATION_SEED = 2027
IMAGE_SIZE = 64
LATENT_DIM = 32
OBS_STD = 0.1
BATCH_SIZE = 64
EPOCHS = 40
BETAS = (0.1, 0.3, 1.0)
SCHEMA = "clarity_sweep_v2"
ROOT = Path("/content/drive/MyDrive/HW1_VAE/reproduction_beta_sweep")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_NAMES = {0.1: "beta_0p1", 0.3: "beta_0p3", 1.0: "beta_1p0"}


def write_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temp.replace(path)


def save_torch(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def group_key(path):
    pattern = (r"(person\d+)_" if path.parent.name == "PNEUMONIA"
               else r"((?:NORMAL2-)?IM-\d+)-")
    match = re.match(pattern, path.stem, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unrecognized dataset filename: {path.name}")
    return f"{path.parent.name}/{match.group(1).upper()}"


def make_manifest(image_root):
    extensions = {".jpeg", ".jpg", ".png"}
    all_files = sorted(p for p in (image_root / "train").rglob("*")
                       if p.is_file() and p.suffix.lower() in extensions)
    train, validation = [], []
    rng = random.Random(SEED)
    for category in ("NORMAL", "PNEUMONIA"):
        groups = defaultdict(list)
        for path in all_files:
            if path.parent.name == category:
                groups[group_key(path)].append(path)
        identifiers = sorted(groups)
        rng.shuffle(identifiers)
        held_out = set(identifiers[:max(1, round(0.10 * len(identifiers)))])
        for identifier in sorted(groups):
            target = validation if identifier in held_out else train
            target.extend(groups[identifier])
    if len(train) != 4699 or len(validation) != 517:
        raise ValueError("Dataset counts differ from the original 4699/517 split.")
    assert not ({group_key(p) for p in train} & {group_key(p) for p in validation})
    manifest = {
        "seed": SEED,
        "method": "Hold out 10% of filename groups within each class",
        "train": [str(p.relative_to(image_root)) for p in train],
        "validation": [str(p.relative_to(image_root)) for p in validation],
    }
    return manifest


def fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def preprocess_files(paths):
    from tqdm.auto import tqdm
    images = torch.empty(len(paths), 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    for index, path in enumerate(tqdm(paths, desc="Preprocessing images once")):
        with Image.open(path) as image:
            image = image.convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        images[index, 0] = torch.from_numpy(pixels.copy())
    return images


def prepare_data(root):
    cache_path = root / "preprocessed_64.pt"
    manifest_path = root / "split_manifest.json"
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        cache = load_torch(cache_path)
        if cache["fingerprint"] != fingerprint(manifest) or cache["image_size"] != IMAGE_SIZE:
            raise ValueError("Preprocessed cache metadata does not match the split.")
        print("Reusing the saved preprocessing cache.", flush=True)
    else:
        import kagglehub
        dataset_path = Path(kagglehub.dataset_download(
            "paultimothymooney/chest-xray-pneumonia"
        ))
        image_root = dataset_path / "chest_xray"
        if not (image_root / "train").is_dir():
            raise FileNotFoundError(f"Expected dataset folder missing: {image_root}")
        manifest = make_manifest(image_root)
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Saved filename split does not match the current dataset.")
        cache = {
            "image_size": IMAGE_SIZE, "fingerprint": fingerprint(manifest),
            "train": preprocess_files([image_root / p for p in manifest["train"]]),
            "validation": preprocess_files([image_root / p for p in manifest["validation"]]),
        }
        save_torch(cache_path, cache)
        write_json(manifest_path, manifest)
    for name, expected in (("train", 4699), ("validation", 517)):
        tensor = cache[name]
        if tuple(tensor.shape) != (expected, 1, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Unexpected cached {name} shape: {tensor.shape}")
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
        assert tensor.min() >= 0 and tensor.max() <= 1
    print("Train / validation images:", len(cache["train"]), len(cache["validation"]), flush=True)
    return cache["train"], cache["validation"], manifest


class CachedImages(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder_layers = nn.Sequential(
            nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, images):
        features = self.features(images)
        return self.fc_mu(features), self.fc_logvar(features)

    def decode(self, latent):
        features = self.fc_decode(latent).reshape(len(latent), 128, 8, 8)
        return self.decoder_layers(features)

    def forward(self, images, generator=None):
        mu, logvar = self.encode(images)
        noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
        latent = mu + torch.exp(0.5 * logvar) * noise
        return self.decode(latent), mu, logvar


def loss_terms(reconstruction, images, mu, logvar, beta):
    reconstruction_loss = (
        (reconstruction - images).square().flatten(1).sum(1) / (2 * OBS_STD**2)
    ).mean()
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)).mean()
    return reconstruction_loss + beta * kl, reconstruction_loss, kl


def make_loaders(train_images, validation_images):
    train_loader = DataLoader(
        CachedImages(train_images), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=DEVICE.type == "cuda",
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_loader = DataLoader(
        CachedImages(validation_images), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=DEVICE.type == "cuda",
        generator=torch.Generator().manual_seed(VALIDATION_SEED),
    )
    return train_loader, validation_loader


def run_epoch(model, loader, optimizer, beta, training):
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    # The same validation noise is used every epoch, independent of training RNG.
    val_rng = None if training else torch.Generator(device=DEVICE).manual_seed(VALIDATION_SEED)
    for images in loader:
        images = images.to(DEVICE, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction, mu, logvar = model(images, generator=val_rng)
            losses = loss_terms(reconstruction, images, mu, logvar, beta)
            if not all(torch.isfinite(term).item() for term in losses):
                raise RuntimeError("Non-finite loss detected; stopping this run.")
            if training:
                losses[0].backward()
                optimizer.step()
        totals += np.array([term.item() for term in losses]) * len(images)
        count += len(images)
    return dict(zip(("loss", "reconstruction", "kl"), (totals / count).tolist()))


def experiment_config(beta, split_fingerprint):
    return {
        "schema": SCHEMA, "architecture": "Original three-stage convolutional VAE",
        "image_size": IMAGE_SIZE, "latent_dim": LATENT_DIM, "obs_std": OBS_STD,
        "beta": beta, "seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "learning_rate_epochs_1_to_20": 1e-3,
        "learning_rate_epochs_21_to_40": 3e-4,
        "validation_seed": VALIDATION_SEED,
        "validation_protocol": "Sampled posterior with fixed validation noise",
        "split_fingerprint": split_fingerprint,
    }


def train_one(beta, train_images, validation_images, root, split_fingerprint):
    run_dir = root / RUN_NAMES[beta]
    run_dir.mkdir(parents=True, exist_ok=True)
    config = experiment_config(beta, split_fingerprint)
    if (run_dir / "config.json").exists():
        if json.loads((run_dir / "config.json").read_text()) != config:
            raise ValueError(f"Experiment settings changed in {run_dir.name}; refusing to mix runs.")
    write_json(run_dir / "config.json", config)
    seed_all(SEED)
    train_loader, validation_loader = make_loaders(train_images, validation_images)
    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    last_path = run_dir / "last_checkpoint.pt"
    if last_path.exists():
        checkpoint = load_torch(last_path)
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint settings do not match this experiment.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if DEVICE.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        train_loader.generator.set_state(checkpoint["loader_rng_state"])
        if not (run_dir / "best_checkpoint.pt").exists():
            raise FileNotFoundError("The last checkpoint exists but its best checkpoint is missing.")
    assert [row["epoch"] for row in history] == list(range(1, len(history) + 1))
    best_loss = min((row["validation"]["loss"] for row in history), default=float("inf"))
    print(f"\n{run_dir.name}: resuming at epoch {len(history) + 1}; target {EPOCHS}.", flush=True)
    for epoch in range(len(history) + 1, EPOCHS + 1):
        learning_rate = 1e-3 if epoch <= 20 else 3e-4
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer, beta, True)
        validation_metrics = run_epoch(model, validation_loader, optimizer, beta, False)
        seconds = time.perf_counter() - start
        row = {"epoch": epoch, "seconds": seconds, "learning_rate": learning_rate,
               "train": train_metrics, "validation": validation_metrics}
        history.append(row)
        if validation_metrics["loss"] <= best_loss:
            best_loss = validation_metrics["loss"]
            save_torch(run_dir / "best_checkpoint.pt", {
                "epoch": epoch, "config": config, "validation": validation_metrics,
                "model_state": {key: value.detach().cpu().clone()
                                for key, value in model.state_dict().items()},
            })
        save_torch(last_path, {
            "epoch": epoch, "config": config, "history": history,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None,
            "loader_rng_state": train_loader.generator.get_state(),
        })
        write_json(run_dir / "history.json", history)
        recent_seconds = np.mean([r["seconds"] for r in history[-5:]])
        remaining_minutes = recent_seconds * (EPOCHS - epoch) / 60
        print(f"{run_dir.name} | Epoch {epoch:02d}/{EPOCHS} | "
              f"Train {train_metrics['loss']:.2f} | Val {validation_metrics['loss']:.2f} | "
              f"Recon {validation_metrics['reconstruction']:.2f} | "
              f"KL {validation_metrics['kl']:.2f} | {seconds:.1f}s | "
              f"Run ETA {remaining_minutes:.1f} min | Saved", flush=True)
    del model, optimizer
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return history


def make_inception():
    from torchvision.models import Inception_V3_Weights, inception_v3
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, progress=False).eval()
    model.requires_grad_(False)
    classifier = model.fc
    model.fc = nn.Identity()
    return model, classifier, weights.transforms()


def plot_samples(images, path, title, columns=8):
    rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns, rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index < len(images):
            axis.imshow(images[index, 0].numpy(), cmap="gray", vmin=0, vmax=1,
                        interpolation="nearest")
        axis.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(pad=0.15, rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reconstructions(originals, reconstructed, path, title):
    fig, axes = plt.subplots(2, len(originals), figsize=(12, 3.4), squeeze=False)
    for row, images in enumerate((originals, reconstructed)):
        for col in range(len(images)):
            axes[row, col].imshow(images[col, 0].numpy(), cmap="gray", vmin=0, vmax=1,
                                  interpolation="nearest")
            axes[row, col].axis("off")
    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Posterior mean")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_all(validation_images, manifest, root):
    import ignite
    from ignite.metrics import FID, InceptionScore
    inception, classifier, preprocess = make_inception()
    inception, classifier = inception.to(DEVICE), classifier.to(DEVICE)
    latents = torch.randn(624, LATENT_DIM, generator=torch.Generator().manual_seed(EVAL_SEED))
    save_torch(root / "evaluation_latents.pt", latents)
    latents = latents[:len(validation_images)]
    preview_indices = torch.randperm(len(validation_images), generator=
                                    torch.Generator().manual_seed(EVAL_SEED))[:8]
    originals = validation_images[preview_indices]
    write_json(root / "preview_files.json", [manifest["validation"][i] for i in preview_indices.tolist()])
    real_features = []
    with torch.inference_mode():
        for batch in validation_images.split(32):
            inputs = preprocess(batch.to(DEVICE).repeat(1, 3, 1, 1))
            real_features.append(inception(inputs).cpu())
    real_features = torch.cat(real_features).double()
    results, comparison_images, reconstruction_rows = {}, [], []
    for beta in BETAS:
        name = RUN_NAMES[beta]
        run_dir = root / name
        checkpoint = load_torch(run_dir / "best_checkpoint.pt")
        model = VAE().to(DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        features, probabilities, samples = [], [], []
        posterior_mu, posterior_logvar = [], []
        with torch.inference_mode():
            for batch in latents.split(32):
                generated = model.decode(batch.to(DEVICE))
                current_features = inception(preprocess(generated.repeat(1, 3, 1, 1)))
                features.append(current_features.cpu())
                probabilities.append(classifier(current_features).softmax(1).cpu())
                samples.append(generated.cpu())
            for batch in validation_images.split(BATCH_SIZE):
                mu, logvar = model.encode(batch.to(DEVICE))
                posterior_mu.append(mu.cpu())
                posterior_logvar.append(logvar.cpu())
            mu, _ = model.encode(originals.to(DEVICE))
            reconstructed = model.decode(mu).cpu()
        generated = torch.cat(samples)
        fake_features = torch.cat(features).double()
        fake_probabilities = torch.cat(probabilities).double()
        fid = FID(num_features=2048, feature_extractor=nn.Identity(), device="cpu")
        fid.update((fake_features, real_features))
        is_metric = InceptionScore(num_features=1000, feature_extractor=nn.Identity(), device="cpu")
        is_metric.update(fake_probabilities)
        posterior_mu, posterior_logvar = torch.cat(posterior_mu), torch.cat(posterior_logvar)
        latent_mean = posterior_mu.mean(0)
        latent_var = (posterior_logvar.exp() + posterior_mu.square()).mean(0) - latent_mean.square()
        reconstruction_mse = 0.0
        with torch.inference_mode():
            for batch in validation_images.split(BATCH_SIZE):
                batch = batch.to(DEVICE)
                mu, _ = model.encode(batch)
                reconstruction_mse += (model.decode(mu) - batch).square().sum().item()
        reconstruction_mse /= validation_images.numel()
        fid_value, is_value = float(fid.compute()), float(is_metric.compute())
        if not math.isfinite(fid_value) or not math.isfinite(is_value):
            raise RuntimeError(f"Non-finite evaluation score for {name}.")
        result = {
            "beta": beta, "checkpoint_epoch": checkpoint["epoch"],
            "validation_fid": fid_value, "inception_score": is_value,
            "posterior_mean_reconstruction_mse": reconstruction_mse,
            "validation_loss_at_selection": checkpoint["validation"],
            "n_real": len(validation_images), "n_generated": len(generated),
            "mean_absolute_aggregate_latent_mean": latent_mean.abs().mean().item(),
            "average_aggregate_latent_std": latent_var.clamp_min(0).sqrt().mean().item(),
            "average_posterior_std": (0.5 * posterior_logvar).exp().mean().item(),
        }
        results[name] = result
        write_json(run_dir / "evaluation.json", result)
        plot_samples(generated[:64], run_dir / "generated_64.png",
                     f"Beta {beta:g} | Best epoch {checkpoint['epoch']} | Prior samples")
        plot_reconstructions(originals, reconstructed, run_dir / "reconstruction.png",
                             f"Beta {beta:g} | Best epoch {checkpoint['epoch']}")
        comparison_images.append(generated[:8])
        reconstruction_rows.append(reconstructed)
        print(f"{name} | Best epoch {checkpoint['epoch']} | Validation FID {fid_value:.4f} | "
              f"IS {is_value:.4f} | Posterior-mean MSE {reconstruction_mse:.6f}", flush=True)
        del model
    protocol = {
        "reference_split": "validation", "n_reference": len(validation_images),
        "evaluation_seed": EVAL_SEED, "latent_draw_size": 624,
        "latent_selection": "First 517 vectors for the full dataset",
        "feature_extractor": "torchvision Inception v3 IMAGENET1K_V1, penultimate 2048 features",
        "preprocessing": repr(preprocess), "ignite_version": ignite.__version__,
        "IS_protocol": "One score over all generated images; no split-wise standard deviation",
        "selection": "Best validation total loss within each beta; lowest validation FID across betas",
        "limitations": "One seed; small validation set; ImageNet features; no untouched final test evaluation",
        "results": results,
    }
    write_json(root / "comparison.json", protocol)
    with (root / "comparison.csv").open("w", newline="") as f:
        fields = ["experiment", "beta", "checkpoint_epoch", "validation_fid", "inception_score",
                  "posterior_mean_reconstruction_mse"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({"experiment": name, **{field: result[field] for field in fields[1:]}})
    best_name = min(results, key=lambda name: results[name]["validation_fid"])
    write_json(root / "selected_by_fid.json", {"experiment": best_name, **results[best_name]})
    for filename, rows, titles in [
        ("generated_comparison.png", comparison_images, [f"Beta {b:g}" for b in BETAS]),
        ("reconstruction_comparison.png", [originals] + reconstruction_rows,
         ["Original"] + [f"Beta {b:g}" for b in BETAS]),
    ]:
        fig, axes = plt.subplots(len(rows), 8, figsize=(12, 1.7 * len(rows)), squeeze=False)
        for row, (images, title) in enumerate(zip(rows, titles)):
            for col in range(8):
                axes[row, col].imshow(images[col, 0].numpy(), cmap="gray", vmin=0, vmax=1,
                                      interpolation="nearest")
                axes[row, col].axis("off")
            axes[row, 0].set_title(title, loc="left")
        fig.tight_layout()
        fig.savefig(root / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
    print("\nSelected by validation FID:", best_name, flush=True)
    print("Inspect generated and reconstructed images before claiming better visual quality.", flush=True)
    del inception, classifier
    return protocol


def plot_histories(root):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for beta, color in zip(BETAS, ("tab:blue", "tab:orange", "tab:green")):
        path = root / RUN_NAMES[beta] / "history.json"
        if not path.exists():
            continue
        history = json.loads(path.read_text())
        for axis, key in zip(axes, ("loss", "reconstruction", "kl")):
            for split, style in (("train", "--"), ("validation", "-")):
                axis.plot([r["epoch"] for r in history], [r[split][key] for r in history],
                          style, color=color, label=f"Beta {beta:g}: {split}")
    for axis, title in zip(axes, ("Beta-specific total loss", "Reconstruction loss", "Unweighted KL")):
        axis.set(title=title, xlabel="Epoch", ylabel="Loss per image")
        axis.grid(alpha=0.2)
        axis.axvline(20.5, color="gray", linestyle=":")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "loss_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_results(root):
    archive = root / "HW1_clarity_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".png", ".txt", ".py"}:
                z.write(path, str(path.relative_to(root)))
    print("Results archive:", archive, flush=True)
    return archive


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("GPU not detected. In Colab choose Runtime > Change runtime type > T4 GPU, then Run all.")
    from google.colab import drive, files
    import torchvision
    drive.mount("/content/drive")
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), ROOT / "hw1_clarity.py")
    environment = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                   "gpu": torch.cuda.get_device_name(0), "schema": SCHEMA}
    write_json(ROOT / "environment.json", environment)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("GPU:", environment["gpu"], flush=True)
    print("Output folder:", ROOT, flush=True)
    print("All three runs start from the same seeded weights. Original experiments are in separate folders.", flush=True)
    error = None
    try:
        # Fetch evaluation weights before starting long training jobs.
        probe_inception, probe_classifier, _ = make_inception()
        del probe_inception, probe_classifier
        train_images, validation_images, manifest = prepare_data(ROOT)
        for beta in BETAS:
            train_one(beta, train_images, validation_images, ROOT, fingerprint(manifest))
        plot_histories(ROOT)
        evaluate_all(validation_images, manifest, ROOT)
        write_json(ROOT / "status.json", {"status": "complete", "experiments": list(RUN_NAMES.values())})
        print("\nALL EXPERIMENTS AND EVALUATIONS COMPLETE.", flush=True)
    except Exception as exc:
        error = exc
        (ROOT / "error.txt").write_text(traceback.format_exc())
        write_json(ROOT / "status.json", {"status": "incomplete", "error": str(exc)})
        print("Run stopped. Checkpoints and diagnostics are saved. Re-run all cells to resume.", flush=True)
    archive = export_results(ROOT)
    if error is None:
        from IPython.display import display
        display(Image.open(ROOT / "generated_comparison.png"))
        display(Image.open(ROOT / "reconstruction_comparison.png"))
    try:
        files.download(str(archive))
    except Exception as download_error:
        print("Automatic download did not start:", str(download_error), flush=True)
        print("Download the ZIP manually from the saved output folder.", flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
