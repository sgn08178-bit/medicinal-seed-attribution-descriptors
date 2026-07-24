# Final analysis environment evidence

- Operating system: Linux 6.8.0-134-generic x86_64, glibc 2.39
- Python: 3.11.15 (conda-forge build)
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB
- NVIDIA driver: 580.159.03
- PyTorch CUDA runtime: 12.8
- cuDNN reported by PyTorch: 9.19.0
- CUDA available during verification: yes

The values above were queried from the same preserved environment referenced by the Stage 1-3 run documentation. Runtime imports in that environment reported OpenCV 4.10.0 and PyWavelets 1.8.0. These imported versions are authoritative for the final analysis and are pinned in `requirements_frozen.txt` and `environment.yml`.

A legacy `pip freeze` record contained conflicting distribution metadata (`opencv-python` 4.11.0.86 alongside `opencv-python-headless` 4.10.0.84, and PyWavelets 1.9.0). That record is retained in the private submission audit and is not included here because it must not be used to recreate the environment.

The exact nightly PyTorch and torchvision builds require the PyTorch nightly CUDA 12.8 package index. A fresh environment was not built because the preserved analysis environment already contains these exact builds and a clean reinstall could fail if that dated nightly wheel is no longer retained upstream. Import and CLI smoke tests were run in the preserved final environment instead.

Auxiliary dependencies required by the released scripts (`PyYAML`, `tqdm`,
`xgboost`, `lightgbm`, and `catboost`) were recovered from the preserved package
inventory and are also pinned in both environment files.
