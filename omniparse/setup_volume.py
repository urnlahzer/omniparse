"""One-time script to download model weights into the Modal Volume.

Run once: modal run omniparse/setup_volume.py

Self-contained — does NOT import from omniparse.app to avoid dependency cascades.
"""
import modal

# Standalone app — no dependency on omniparse.app
setup_app = modal.App("omniparse-setup")

model_volume = modal.Volume.from_name("ocr-models", create_if_missing=True)
MODELS_DIR = "/models"

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs")
    .pip_install("huggingface-hub>=0.24")
)


@setup_app.function(
    image=download_image,
    volumes={MODELS_DIR: model_volume},
    timeout=3600,
    memory=8192,
)
def download_trocr_model():
    """Download TrOCR-large-handwritten from HuggingFace."""
    import os
    from huggingface_hub import snapshot_download

    target = f"{MODELS_DIR}/trocr/trocr-large-handwritten"
    os.makedirs(target, exist_ok=True)

    print("Downloading TrOCR-large-handwritten...")
    snapshot_download(
        "microsoft/trocr-large-handwritten",
        local_dir=target,
        local_dir_use_symlinks=False,
    )
    model_volume.commit()
    print("TrOCR model downloaded.")


@setup_app.function(
    image=download_image,
    volumes={MODELS_DIR: model_volume},
    timeout=3600,
    memory=16384,
)
def download_qwen_model():
    """Download Qwen3-VL-8B-Instruct-FP8 from HuggingFace."""
    import os
    from huggingface_hub import snapshot_download

    target = f"{MODELS_DIR}/Qwen3-VL-8B-Instruct-FP8"
    os.makedirs(target, exist_ok=True)

    print("Downloading Qwen3-VL-8B-Instruct-FP8 (~10.7 GB)...")
    snapshot_download(
        "Qwen/Qwen3-VL-8B-Instruct-FP8",
        local_dir=target,
        local_dir_use_symlinks=False,
    )
    model_volume.commit()
    print("Qwen3-VL-8B model downloaded.")


@setup_app.function(
    image=download_image,
    volumes={MODELS_DIR: model_volume},
    timeout=3600,
    memory=8192,
)
def download_dots_model():
    """Download Dots.ocr model from HuggingFace."""
    import os
    from huggingface_hub import snapshot_download

    target = f"{MODELS_DIR}/dots/dots-ocr"
    os.makedirs(target, exist_ok=True)

    print("Downloading Dots.ocr model (rednote-hilab/dots.ocr)...")
    snapshot_download(
        "rednote-hilab/dots.ocr",
        local_dir=target,
        local_dir_use_symlinks=False,
    )
    model_volume.commit()
    print("Dots.ocr model downloaded.")


@setup_app.function(
    image=download_image,
    volumes={MODELS_DIR: model_volume},
    timeout=600,
    memory=4096,
)
def verify_volume():
    """Verify all models are downloaded."""
    import os

    print(f"\n{'='*60}")
    print("Volume contents:")
    print(f"{'='*60}")

    total_size = 0
    for root, dirs, files in os.walk(MODELS_DIR):
        level = root.replace(MODELS_DIR, "").count(os.sep)
        indent = " " * 2 * level
        subdir = os.path.basename(root)
        dir_size = sum(os.path.getsize(os.path.join(root, f)) for f in files)
        total_size += dir_size
        if level <= 2:
            print(f"{indent}{subdir}/ ({dir_size / 1e9:.2f} GB, {len(files)} files)")

    print(f"\nTotal: {total_size / 1e9:.1f} GB")

    expected = ["trocr", "dots", "Qwen3-VL-8B-Instruct-FP8"]
    missing = [d for d in expected if not os.path.isdir(f"{MODELS_DIR}/{d}")]
    if missing:
        print(f"\nMISSING: {missing}")
    else:
        print("\nAll HF model directories present.")
    print("\nNote: PaddleOCR and Docling models auto-download on first engine use.")


@setup_app.local_entrypoint()
def main():
    """Download HF models in parallel, then verify."""
    print("Starting model downloads (3 parallel jobs)...")
    print("PaddleOCR and Docling models will auto-download on first engine use.\n")

    handles = [
        download_trocr_model.spawn(),
        download_qwen_model.spawn(),
        download_dots_model.spawn(),
    ]

    for h in handles:
        h.get()

    print("\nAll downloads complete. Verifying...")
    verify_volume.remote()
    print("\nSetup complete!")
