"""Logic for backend selection"""

import os

# 通过环境变量切换后端：nd 为课程自定义后端，np 为纯 numpy 后端
BACKEND = os.environ.get("NEEDLE_BACKEND", "nd")

if BACKEND == "nd":
    print("Using needle backend")
    from . import backend_ndarray as array_api
    from .backend_ndarray import (
        all_devices,
        cuda,
        cpu,
        cpu_numpy,
        default_device,
        BackendDevice as Device,
    )

    NDArray = array_api.NDArray

elif BACKEND == "np":
    print("Using numpy backend")
    import numpy as array_api
    from .backend_numpy import all_devices, cpu, cuda, default_device, Device

    NDArray = array_api.ndarray

else:
    raise RuntimeError("Unknown needle array backend %s" % BACKEND)
