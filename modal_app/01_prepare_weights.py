import os
from pathlib import Path
import modal

# Persistent Volume for model weights
WEIGHTS_VOL = modal.Volume.from_name("wan21-weights", create_if_missing=True)
MODEL_ID = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

# Minimal image for downloading from the Hub with fast transfer
image = modal.Image.debian_slim().pip_install("huggingface_hub[hf_transfer]")

app = modal.App("wan21-prepare-weights", image=image)

# Hugging Face token is provided via a Modal Secret called 'hf-token'
hf_secret = modal.Secret.from_name("hf-token")

@app.function(image=image, secrets=[hf_secret], volumes={"/models": WEIGHTS_VOL}, timeout=3600)
def download_weights():
    """Download the full model repository into the Modal Volume once."""
    from huggingface_hub import snapshot_download

    # Speed up downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    target = Path("/models/Wan2.1-T2V-1.3B-Diffusers")
    target.mkdir(parents=True, exist_ok=True)

    local = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return {"ok": True, "local_dir": local}
