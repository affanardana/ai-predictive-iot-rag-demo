"""Modal deployment for the inference service.

Nothing else depends on this file. The service is a plain ASGI app and runs
under uvicorn locally; this is one way to host it, chosen because the
masterplan's stack names Modal.

Deploying is deliberately not something this repository does on your behalf:

    modal volume create pdm-model
    modal volume put pdm-model colab_report/best.pt /best.pt
    modal volume put pdm-model data/training/normalization.json /artifact/normalization.json
    modal deploy services/inference/modal_app.py

The checkpoint and the normalisation statistics go in a Volume rather than the
image because they change whenever the model is retrained, and rebuilding an
image to pick up a 1.5 MB file is the wrong trade.

Once deployed, point the API at it:

    INFERENCE_SERVICE_URL=https://<workspace>--pdm-inference-fastapi-app.modal.run
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

import modal

APP_NAME = "pdm-inference"
VOLUME_NAME = "pdm-model"

#: Where the Volume is mounted inside the container.
MOUNT = "/model"
CHECKPOINT = f"{MOUNT}/best.pt"
ARTIFACT = f"{MOUNT}/artifact"

#: CPU, not CUDA. The model is two LSTM layers and a linear head, and the CUDA
#: wheel is roughly ten times the size for compute this does not use.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("fastapi", "uvicorn", "numpy", "pyarrow")
    # Every build step has to come before the local directories below. Modal
    # rejects an image that runs a build step afterwards, and it is right to:
    # local files added last are injected at container startup, so a change to
    # one does not invalidate the cached layers and rebuild torch.
    .env(
        {
            "PYTHONPATH": "/root",
            "INFERENCE_CHECKPOINT": CHECKPOINT,
            "INFERENCE_ARTIFACT_DIR": ARTIFACT,
        }
    )
    # The three packages the service imports, copied in as source. They are
    # nested under `src/`, which is not a layout `add_local_python_source` finds
    # on its own, so each is mounted at the name it should be imported by and
    # PYTHONPATH does the rest.
    .add_local_dir("ml/src/ml", remote_path="/root/ml")
    .add_local_dir("services/simulator/src/simulator", remote_path="/root/simulator")
    .add_local_dir("services/inference/src/inference", remote_path="/root/inference")
)

app = modal.App(APP_NAME, image=image)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    volumes={MOUNT: model_volume},
    # Scale to zero between requests. Inference is a 60-step forward pass over a
    # 1.5 MB model, so holding a container warm costs more than the cold start.
    min_containers=0,
    # One request at a time per container: the model is loaded once at startup
    # and held, and a single CPU core is not helped by concurrency here.
    max_containers=4,
    timeout=60,
)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    """The service, wrapped for Modal's ASGI runtime."""
    from inference.app import create_app

    return create_app()


@app.local_entrypoint()
def check() -> None:
    """Report whether the Volume holds what the service needs.

    Run before deploying, and again after retraining. The service refuses to
    start without these files, so finding out here costs seconds rather than a
    failed deployment.
    """
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

    # Listed per directory rather than one recursive walk. `listdir("")` returns
    # only the top level, so checking `"artifact/normalization.json"` against it
    # reports MISSING for a file that is there — which it did, on the first real
    # deployment.
    def contains(directory: str, name: str) -> bool:
        return any(entry.path == f"/{directory}/{name}" for entry in volume.listdir(directory))

    print(f"volume {VOLUME_NAME}")
    missing = []
    for directory, name in (("", "best.pt"), ("artifact", "normalization.json")):
        found = contains(directory, name)
        where = f"{directory or ''}/{name}".lstrip("/")
        print(f"  {'ok  ' if found else 'MISSING'} {where}")
        if not found:
            missing.append(where)
    if missing:
        print("\nUpload the missing files with:")
        for path in missing:
            print(f"  modal volume put {VOLUME_NAME} <local-file> /{path}")
