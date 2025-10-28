# WAN 2.1 on Modal via GitHub Actions (no local CLI)

This repo deploys a **FastAPI web endpoint** on **Modal GPU** to generate **text-to-video** with **Wan 2.1 (T2V-1.3B)**.
It also **preloads model weights into a Modal Volume** to reduce cold starts. Everything is triggered from **GitHub Actions**.

## What you'll need
- A Modal workspace + **API tokens** (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`).
- A Hugging Face **read token** (`HF_TOKEN`) if the checkpoints are gated.

## 1) Add repository secrets (GitHub → Settings → Secrets and variables → Actions)
Create three secrets:
- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `HF_TOKEN`

> Modal tokens docs and config: see Modal docs for `modal token set` / `MODAL_TOKEN_ID` env variables.
> Hugging Face download speedup uses `huggingface_hub[hf_transfer]` and `HF_HUB_ENABLE_HF_TRANSFER=1`.

## 2) Kick off Actions (no local shell)
Go to **Actions** tab:
1. Run **Modal – Init & Prepare Weights** → creates Modal Secret `hf-token`, Volumes `wan21-weights` and `wan21-outputs`, and **preloads** weights into the weights Volume.
2. Run **Modal – Deploy WAN API** (or push to `main`) → deploys the FastAPI web endpoint on GPU (**A10G**).

After deploy, check your **Modal Dashboard** for the endpoint URL like:
```
https://<something>.modal.run
```

## 3) Quick test (from any HTTP client)
```
POST https://<your>.modal.run/generate
Content-Type: application/json

{
  "prompt": "colibrí volando entre flores, luz suave, DOF",
  "num_frames": 49,
  "height": 480,
  "width": 832
}
```
The response includes a Base64-encoded MP4 (`video_base64`).

## 4) n8n
Use an **HTTP Request** node (POST) to the `/generate` URL and pass the JSON above.
Then add **Move/Convert Binary** to write `video.mp4` from `video_base64`.

## Tech notes
- **Volumes** are designed for model weights (write-once, read-many) and reduce cold starts vs. downloading on every run.
- We install **PyTorch CUDA 12.1 wheels** via `extra_index_url` and **imageio[ffmpeg]** to enable `export_to_video`.
- GPU starts with **A10G (24GB)** suitable for T2V-1.3B (~13GB VRAM). For higher res / larger models, use A100/H100.
- The API class uses `@modal.enter()` to load the model **once per container**.

