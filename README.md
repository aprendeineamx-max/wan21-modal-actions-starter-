# Generación de video con Wan 2.1 en Modal usando GitHub Actions

Este repositorio automatiza la preparación de pesos y el despliegue de un endpoint **FastAPI** sobre **Modal GPU** para generar video a partir de texto con **Wan 2.1 (T2V-1.3B)** u otros checkpoints compatibles. Todo se orquesta desde **GitHub Actions**, sin necesidad de usar la CLI de Modal de forma local.

## Requisitos
- Una cuenta de Modal con los tokens `MODAL_TOKEN_ID` y `MODAL_TOKEN_SECRET`.
- Un token de lectura de Hugging Face (`HF_TOKEN`) cuando el modelo elegido está restringido.

## 1. Configurar secretos del repositorio
En GitHub → *Settings* → *Secrets and variables* → *Actions* crea los siguientes secretos:
- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `HF_TOKEN`

> Modal documenta cómo generar los tokens en su panel de control (`modal token set`).
> La descarga usa `huggingface_hub[hf_transfer]` para acelerar las transferencias (`HF_HUB_ENABLE_HF_TRANSFER=1`).

## 2. Ejecutar los flujos de trabajo
Desde la pestaña **Actions**:
1. **Modal – Inicializar y Preparar Pesos**: crea/actualiza el secreto `hf-token` dentro de Modal, asegura los volúmenes `wan21-weights` y `wan21-outputs`, y descarga los pesos del modelo seleccionado al volumen de pesos.
2. **Modal – Desplegar API de WAN** (o un *push* a `main`): monta los volúmenes en una GPU Modal, carga el pipeline Wan y expone los endpoints `/health` y `/generate`.

Ambos flujos aceptan parámetros manuales para elegir modelo, GPU y FPS por defecto desde la UI de GitHub Actions. Si no se especifican, se usan los valores recomendados.

## 3. Probar rápidamente la API
Una vez desplegada la app, copia la URL que aparece en tu panel de Modal y realiza la petición:

```http
POST https://<tu-app>.modal.run/generate
Content-Type: application/json

{
  "prompt": "colibrí volando entre flores, luz suave, DOF",
  "num_frames": 49,
  "height": 480,
  "width": 832
}
```

La respuesta incluye un MP4 en Base64 (`video_base64`). Si prefieres almacenar el archivo en el volumen `wan21-outputs`, envía `"return_base64": false` y se devolverá la ruta.

## 4. Uso con n8n
1. Añade un nodo **HTTP Request** (POST) apuntando a `/generate` con el JSON de ejemplo.
2. Encadena un nodo **Move/Convert Binary** para guardar `video.mp4` a partir del campo `video_base64`.

## Configuración y parámetros
### Checkpoints disponibles (`WAN_MODEL_ID`)
| Checkpoint sugerido | Descripción | VRAM mínima estimada |
| --- | --- | --- |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Modelo base Wan 2.1 texto-a-video. | ~13 GB (A10G) |
| `Wan-AI/Wan-T2V-1.3B-General` | Variante generalista T2V 1.3B. | ~13 GB |
| `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Versión grande 14B (más nítida). | ≥40 GB (A100/H100) |

> Los pesos se guardan en subcarpetas separadas dentro del volumen `wan21-weights`, por lo que puedes precargar varios modelos sin sobrescribir archivos.

### Parámetros de despliegue (Actions)
- **WAN_MODEL_ID**: elige el checkpoint. Define también la carpeta interna usada como caché.
- **WAN_GPU_TYPE**: GPU de Modal (ej. `A10G`, `A100`, `H100`). Ajusta según VRAM y presupuesto.
- **WAN_VIDEO_FPS**: FPS por defecto para los videos exportados con `export_to_video`.

### Parámetros del endpoint `/generate`
| Campo | Tipo | Valor por defecto | Descripción |
| --- | --- | --- | --- |
| `prompt` | `str` | — | Descripción en texto del video a generar. |
| `num_frames` | `int` | `49` | Número de cuadros generados. |
| `height` | `int` | `480` | Altura del video (múltiplo de 16 recomendado). |
| `width` | `int` | `832` | Anchura del video. |
| `steps` | `int` | `18` | Pasos de inferencia (más pasos = más calidad pero más tiempo). |
| `guidance` | `float` | `5.0` | Intensidad de la guía de texto. |
| `return_base64` | `bool` | `true` | Si es `true`, retorna el MP4 en Base64; si es `false`, retorna la ruta en el volumen. |
| `fps` | `int` | valor de `WAN_VIDEO_FPS` | Cuadros por segundo al exportar el MP4. |

## Notas técnicas
- Los volúmenes de Modal permiten descargar los pesos una sola vez y reutilizarlos en múltiples despliegues.
- La imagen Docker mínima instala PyTorch con CUDA 12.1 y dependencias de Diffusers, incluyendo `imageio[ffmpeg]` para exportar MP4.
- El servicio carga el VAE en `float32` y el resto del pipeline en `bfloat16`, siguiendo las recomendaciones de Diffusers.
- El endpoint `/health` expone el modelo activo, la GPU y la carpeta de caché usada.

Con esta estructura puedes alternar checkpoints, GPU y parámetros sin tocar código, únicamente lanzando los workflows desde GitHub o ajustando las variables de entorno correspondientes en Modal.
