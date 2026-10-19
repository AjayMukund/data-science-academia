"""
download_gpt2.py - fetch GPT-2 small's weights (548 MB) from Hugging Face.

The transformer and sampling experiments run GPT-2 in plain numpy, straight
from this one file. It lands in ../models/gpt2/ (ignored by git).

    python common/download_gpt2.py
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, "..", "models", "gpt2", "model.safetensors")
URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"

if os.path.exists(DST):
    print("already there:", os.path.abspath(DST))
else:
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    print("downloading GPT-2 small (548 MB) ...")
    urllib.request.urlretrieve(URL, DST)
    print("saved to", os.path.abspath(DST))
