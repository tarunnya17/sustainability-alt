# #44754 [issue] I really need help regarding to meta device issue
https://github.com/huggingface/transformers/issues/44754  ·  opened by Hany138078 on 2026-03-16

## [body] Hany138078 (NONE) · opening post · 2026-03-16
I installed TRELLIS 2 and getting the error for meta device, I followed the instructions here for installing this release of transformers, but when I launch TRELLIS 2 I still get the same error. Can someone please help me Fix this step by step because I spend 5 days trying without any solution

## [c4103913073] sasmita016 (NONE) · issue_comment · reply to [body] · 2026-03-21
@Hany138078 This looks like the known TRELLIS.2 / BiRefNet issue with Transformers 5.x and meta tensors.

The error:
RuntimeError: Tensor.item() cannot be called on meta tensors

happens during model initialization (briaai/birefnet) when using newer Transformers versions. There are related reports showing this breaks with Transformers 5.x but works with 4.x.

A practical workaround that worked for me / others:

1. Create a clean environment (important — avoids mixed packages)
   conda create -n trellis2_clean python=3.10 -y
   conda activate trellis2_clean

2. Install the PyTorch version TRELLIS.2 expects (CUDA 12.4 example)
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

3. Install dependencies but pin Transformers to <5
   pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh gradio==6.0.1 tensorboard pandas lpips zstandard
   pip install "transformers<5"
   pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
   pip install pillow-simd
   pip install kornia timm

4. Clear cached HF modules (important for this specific error)
   rm -rf ~/.cache/huggingface/modules/transformers_modules/briaai
   rm -rf ~/.cache/huggingface/hub/models--briaai*

5. Verify versions before launching
   python -c "import sys, torch, transformers; print(sys.executable); print(torch.__version__); print(transformers.__version__)"

   Expected:
   - torch: 2.6.0
   - transformers: 4.x

6. Launch again
   python app.py

---

If it still fails, it would help to share:
- OS (Linux / WSL / Windows)
- GPU + VRAM (output of nvidia-smi)
- Python / torch / transformers versions
- full traceback

This seems to be a compatibility issue rather than a user setup mistake, so pinning Transformers is currently the safest workaround.

---
outcome: state=closed
