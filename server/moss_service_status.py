"""Local-only status endpoint for the shared MOSS-Image service."""
import os
import time

import psutil
import torch
from aiohttp import web
from server import PromptServer

from comfy import model_management

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
STARTED = time.time()


@PromptServer.instance.routes.get("/moss/status")
async def moss_status(request):
    loaded = []
    loaded_dtypes = []
    try:
        for item in model_management.current_loaded_models:
            patcher = item.model
            model = getattr(patcher, "model", None)
            if model is not None:
                loaded.append(type(model).__name__)
                entry = {"name": type(model).__name__, "dtype": None, "dtypes": []}
                try:
                    counts = {}
                    first = None
                    for parameter in model.parameters():
                        key = str(parameter.dtype)
                        counts[key] = counts.get(key, 0) + 1
                        if first is None:
                            first = key
                    entry["dtype"] = first
                    entry["dtypes"] = [
                        {"dtype": dtype, "count": count}
                        for dtype, count in sorted(counts.items(), key=lambda item: -item[1])[:3]
                    ]
                except Exception:
                    pass
                loaded_dtypes.append(entry)
    except Exception:
        pass
    vram_free = vram_total = None
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            vram_free, vram_total = int(free), int(total)
    except Exception:
        pass
    args = model_management.args
    return web.json_response({
        "service": "ok",
        "pid": os.getpid(),
        "uptime_seconds": int(time.time() - STARTED),
        "loaded_models": loaded,
        "loaded_dtypes": loaded_dtypes,
        "text_encoder_dtype": str(model_management.text_encoder_dtype()),
        "text_encoder_flags": {
            "bf16": bool(args.bf16_text_enc),
            "fp8_e4m3fn": bool(args.fp8_e4m3fn_text_enc),
            "fp8_e5m2": bool(args.fp8_e5m2_text_enc),
            "fp16": bool(args.fp16_text_enc),
            "fp32": bool(args.fp32_text_enc),
        },
        "unet_dtype_flag": bool(args.bf16_unet),
        "process_rss": psutil.Process().memory_info().rss,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_free": vram_free,
        "vram_total": vram_total,
        "vram_state": str(model_management.vram_state),
    })
