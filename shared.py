import random, sys, platform, importlib.metadata as md, time
import numpy as np

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError: pass

def versions(path):
    names=['numpy','pandas','scikit-learn','scipy','matplotlib','seaborn','torch','torchvision','torchinfo','thop','onnx','onnxruntime','mlflow','memory-profiler','psutil','codecarbon','fastapi','uvicorn','pytest','httpx','locust','requests','pyarrow','joblib','tqdm','PyYAML']
    lines=[f'Python: {sys.version}',f'Platform: {platform.platform()}']
    for name in names:
        try: v=md.version(name)
        except md.PackageNotFoundError: v='NOT INSTALLED'
        lines.append(f'{name}: {v}')
    path.write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))

def median_bench(fn,n=5,warmup=1):
    for _ in range(warmup): fn()
    values=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); values.append((time.perf_counter()-t)*1000)
    return float(np.median(values))
