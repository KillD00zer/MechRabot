"""
Upload extracted_artifacts images to a Modal Volume.

Usage:
    modal run app/upload_images.py
"""

import modal
from pathlib import Path

app = modal.App("mechrabot-image-upload")

volume = modal.Volume.from_name("mechrabot-images", create_if_missing=True)

LOCAL_DIR = Path(__file__).resolve().parent.parent / "kgl V3 outputs" / "kaggle_output_100_131" / "extracted_artifacts"


@app.local_entrypoint()
def main():
    files = sorted(LOCAL_DIR.glob("*.png"))
    print(f"Found {len(files)} images in {LOCAL_DIR}")

    with volume.batch_upload() as batch:
        for f in files:
            batch.put_file(str(f), f"/{f.name}")

    print(f"✅ Uploaded {len(files)} images to volume 'mechrabot-images'")
