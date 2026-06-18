# -*- coding: utf-8 -*-
"""Download Llama-3.2-3B weights from ModelScope into ./llm_weights/Llama-3.2-3B."""
import os
from modelscope import snapshot_download

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_weights")
os.makedirs(target, exist_ok=True)

# ModelScope hosts the official Llama-3.2-3B under LLM-Research
model_id = "LLM-Research/Llama-3.2-3B"
print("Downloading %s -> %s" % (model_id, target))
path = snapshot_download(model_id, cache_dir=target)
print("DOWNLOADED_TO:", path)
