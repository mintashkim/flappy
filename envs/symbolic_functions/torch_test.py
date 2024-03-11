import numpy as np
import time
import torch
from torch._dynamo import config
config.numpy_default_float = "float32"

def kmeans(X, means):
    return np.argmin(np.linalg.norm(X - means[:, None], axis=2), axis=0)

npts = 10_000_000
X = np.repeat([[5, 5], [10, 10]], [npts, npts], axis=0)
X = X + np.random.randn(*X.shape)  # 2 distinct "blobs"
means = np.array([[5, 5], [10, 10]])

tic = time.perf_counter()
np_pred = kmeans(X, means)
toc = time.perf_counter()
print(toc-tic)

compiled_fn = torch.compile(kmeans, backend="aot_eager")
tic = time.perf_counter()
with torch.device("mps"):
    compiled_pred = compiled_fn(X., means)
toc = time.perf_counter()
print(toc-tic)
assert np.allclose(np_pred, compiled_pred)

