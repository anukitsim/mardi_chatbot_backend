# test_torch_faiss.py

try:
    import torch
    print("Torch version:", torch.__version__)
except Exception as e:
    print("Torch import error:", e)

try:
    import faiss
    print("FAISS version:", faiss.__version__)
except Exception as e:
    print("FAISS import error:", e)

