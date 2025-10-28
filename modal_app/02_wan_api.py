import os, uuid, base64
from pathlib import Path
import modal

# Volumes
WEIGHTS_VOL  = modal.Volume.from_name("wan21-weights")
OUTPUTS_VOL  = modal.Volume.from_name("wan21-outputs", create_if_missing=True)

WEIGHTS_DIR  = "/models"
OUTPUTS_DIR  = "/outputs"

# Image with CUDA wheels for PyTorch + Diffusers stack + ffmpeg for MP4 export
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

# Read-only secret for HF token if checkpoints are gated
hf_secret = modal.Secret.from_name("hf-token")

@app.cls(
    image=image,
    gpu="A10G",  # Start with A10G (24GB). For bigger models/resolutions, use A100/H100.
    secrets=[hf_secret],
    volumes={WEIGHTS_DIR: WEIGHTS_VOL, OUTPUTS_DIR: OUTPUTS_VOL},
    timeout=1800,
)
class Wan21Service:
    def __init__(self):
        self.pipe = None
        self.model_id = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.cache_dir = Path(WEIGHTS_DIR) / "Wan2.1-T2V-1.3B-Diffusers"

    @modal.enter()  # Runs once per container
    def load_model(self):
        import torch
        from diffusers import WanPipeline, AutoModel  # AutoModel covers subcomponents like VAE/transformer

        # Recommended recipe: VAE fp32 + pipeline bf16
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

        # Write MP4 into outputs Volume
        mp4_path = Path(OUTPUTS_DIR) / f"{uuid.uuid4()}.mp4"
        export_to_video(frames, str(mp4_path), fps=16)

        if return_base64:
            b64 = base64.b64encode(mp4_path.read_bytes()).decode("utf-8")
            return {"ok": True, "mime": "video/mp4", "video_base64": b64}
        return {"ok": True, "path": str(mp4_path)}

    @modal.asgi_app()  # Expose a FastAPI app
    def web(self):
        from fastapi import FastAPI
        from pydantic import BaseModel

        app = FastAPI()

        @app.get("/health")
        def health():
            return {"ok": True, "model": self.model_id}

        class GenIn(BaseModel):
            prompt: str
            num_frames: int = 49
            height: int = 480
            width: int = 832
            steps: int = 18
            guidance: float = 5.0
            return_base64: bool = True

        @app.post("/generate")
        def generate_endpoint(body: GenIn):
            # Call the Modal method with the container spec
            return self.generate.remote(
                body.prompt, body.num_frames, body.height, body.width, body.steps, body.guidance, body.return_base64
            )

        return app
