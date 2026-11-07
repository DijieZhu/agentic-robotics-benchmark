"""Download the models a supervisor may use offline. Runs during `docker build` only.

    fetch_models.py /opt/models

CLIP and OWLv2 from the Hugging Face Hub; torchvision's COCO Faster R-CNN (ResNet-50 FPN v2)
weights from download.pytorch.org, loadable with
`fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None).load_state_dict(torch.load(...))`.
"""

import os
import sys

from huggingface_hub import snapshot_download

MODELS = ("openai/clip-vit-base-patch16", "google/owlv2-base-patch16-ensemble")
FRCNN_URL = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
FRCNN_FILE = "fasterrcnn_resnet50_fpn_v2_coco.pth"


def main(out):
    for repo in MODELS:
        snapshot_download(repo, local_dir=os.path.join(out, repo.split("/")[1]),
                          allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin", "*.txt",
                                          "*.model"])
    import torch
    state = torch.hub.load_state_dict_from_url(FRCNN_URL, model_dir=os.path.join(out, ".cache"),
                                               check_hash=True, progress=False)
    torch.save(state, os.path.join(out, FRCNN_FILE))


if __name__ == "__main__":
    main(sys.argv[1])
