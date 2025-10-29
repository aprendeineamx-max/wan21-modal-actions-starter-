import base64
import os
import re
import uuid
from pathlib import Path

import modal

# Volúmenes persistentes
WEIGHTS_VOL = modal.Volume.from_name("wan21-weights")
OUTPUTS_VOL = modal.Volume.from_name("wan21-outputs", create_if_missing=True)

WEIGHTS_DIR = "/models"
OUTPUTS_DIR = "/outputs"


def _modelo_a_slug(model_id: str) -> str:
    """Normaliza el identificador del modelo para reutilizar los pesos correctos."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id).strip("._-")
    return slug or "modelo_desconocido"


def _entero_env(nombre: str, defecto: int) -> int:
    """Convierte una variable de entorno en entero positivo o devuelve el valor por defecto."""
    valor = os.environ.get(nombre)
    if valor is None:
        return defecto
    try:
        numero = int(valor)
    except ValueError:
        return defecto
    return numero if numero > 0 else defecto


MODEL_ID = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
MODEL_SLUG = _modelo_a_slug(MODEL_ID)
GPU_TYPE = os.environ.get("WAN_GPU_TYPE", "A10G")
DEFAULT_FPS = _entero_env("WAN_VIDEO_FPS", 16)

# Imagen base con soporte CUDA para PyTorch + Diffusers + ffmpeg para exportar MP4
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch==2.4.1",
        "diffusers==0.35.1",
        "transformers>=4.44",
        "accelerate",
        "ftfy",
        "huggingface_hub[hf_transfer]",
        "imageio[ffmpeg]",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

app = modal.App("wan21-t2v-api", image=image)

# Secreto de solo lectura para el token de Hugging Face (modelos privados)
hf_secret = modal.Secret.from_name("hf-token")

@app.cls(
    image=image,
    gpu=GPU_TYPE,  # Por defecto A10G (24GB). Para modelos más pesados usar A100/H100.
    secrets=[hf_secret],
    volumes={WEIGHTS_DIR: WEIGHTS_VOL, OUTPUTS_DIR: OUTPUTS_VOL},
    timeout=1800,
)
class Wan21Service:
    def __init__(self):
        self.pipe = None
        self.model_id = MODEL_ID
        self.cache_dir = Path(WEIGHTS_DIR) / MODEL_SLUG

    @modal.enter()  # Se ejecuta una sola vez por contenedor
    def load_model(self):
        import torch
        from diffusers import WanPipeline, AutoModel  # AutoModel carga subcomponentes como VAE o transformer

        # Receta recomendada: VAE en fp32 + pipeline en bf16
        vae = AutoModel.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=str(self.cache_dir)
        )
        self.pipe = WanPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
            cache_dir=str(self.cache_dir),
        ).to("cuda")

    @modal.method()
    def generate(
        self,
        prompt: str,
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        steps: int = 18,
        guidance: float = 5.0,
        return_base64: bool = True,
        fps: int = DEFAULT_FPS,
    ):
        from diffusers.utils import export_to_video

        result = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        frames = result.frames[0]

        # Guarda el MP4 dentro del Volumen de salidas
        mp4_path = Path(OUTPUTS_DIR) / f"{uuid.uuid4()}.mp4"
        fps_final = fps if fps > 0 else DEFAULT_FPS
        export_to_video(frames, str(mp4_path), fps=fps_final)

        if return_base64:
            b64 = base64.b64encode(mp4_path.read_bytes()).decode("utf-8")
            return {"ok": True, "mime": "video/mp4", "video_base64": b64, "fps": fps_final}
        return {"ok": True, "path": str(mp4_path), "fps": fps_final}

    @modal.asgi_app()  # Expone una aplicación FastAPI
    def web(self):
        from fastapi import FastAPI
        from pydantic import BaseModel

        app = FastAPI()

        @app.get("/health")
        def health():
            return {"ok": True, "modelo": self.model_id, "gpu": GPU_TYPE, "carpeta_modelo": str(self.cache_dir)}

        class GenIn(BaseModel):
            prompt: str
            num_frames: int = 49
            height: int = 480
            width: int = 832
            steps: int = 18
            guidance: float = 5.0
            return_base64: bool = True
            fps: int = DEFAULT_FPS

        @app.post("/generate")
        def generate_endpoint(body: GenIn):
            # Invoca al método remoto dentro del mismo contenedor GPU
            return self.generate.remote(
                body.prompt,
                body.num_frames,
                body.height,
                body.width,
                body.steps,
                body.guidance,
                body.return_base64,
                body.fps,
            )

        return app
