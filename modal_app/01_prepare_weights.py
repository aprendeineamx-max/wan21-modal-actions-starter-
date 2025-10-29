import os
import re
from pathlib import Path

import modal

# Volumen persistente donde se almacenan los pesos descargados
WEIGHTS_VOL = modal.Volume.from_name("wan21-weights", create_if_missing=True)
MODEL_ID = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")


def _modelo_a_slug(model_id: str) -> str:
    """Genera un nombre de carpeta seguro a partir del identificador del modelo."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id).strip("._-")
    return slug or "modelo_desconocido"


MODELO_SLUG = _modelo_a_slug(MODEL_ID)

# Imagen mínima para acelerar las descargas desde Hugging Face con hf_transfer
image = modal.Image.debian_slim().pip_install("huggingface_hub[hf_transfer]")

app = modal.App("wan21-prepare-weights", image=image)

# El token de Hugging Face se obtiene de un Secret de Modal llamado 'hf-token'
hf_secret = modal.Secret.from_name("hf-token")

@app.function(image=image, secrets=[hf_secret], volumes={"/models": WEIGHTS_VOL}, timeout=3600)
def download_weights():
    """Descarga todo el repositorio del modelo dentro del Volumen de Modal una sola vez."""
    from huggingface_hub import snapshot_download

    # Acelera las descargas paralelas desde el Hub
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    target = Path("/models") / MODELO_SLUG
    target.mkdir(parents=True, exist_ok=True)

    local = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return {"ok": True, "local_dir": local}
