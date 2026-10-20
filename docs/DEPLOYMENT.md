# 部署与服务端集成

插件只负责“选择 + 把 Text Encoder 对齐到所选档位”。DiT 精度需要由**提交生成请求的
客户端**写进工作流，因此一个可用的部署包含三部分：

1. 本仓库的精度层 `server/moss_precision.py`；
2. ComfyUI 侧的两处 FP8 修正与 `/moss/status` 扩展；
3. 生成客户端按契约读取 `state/precision.json`。

## 目录布局

```text
<MOSS_ROOT>\
├─ ComfyUI\
│  └─ custom_nodes\
│     ├─ moss-image\                 # 官方 MOSS 节点（需要打两处补丁）
│     └─ moss_service\               # /moss/status 扩展（本仓库提供）
├─ tools\
│  └─ moss_precision.py             # 本仓库 server/moss_precision.py
├─ venv\Scripts\python.exe
├─ start_service.py                 # 需要支持 --text-enc
├─ state\precision.json             # 运行时生成
└─ logs\comfyui.log
```

## 1. 复制精度层

```bat
copy server\moss_precision.py "<MOSS_ROOT>\tools\moss_precision.py"
```

脚本只用标准库；`ROOT` 由脚本位置（`tools/` 的父目录）推导，所以**必须**放在
`<MOSS_ROOT>\tools\`。

## 2. `/moss/status` 扩展（必需）

插件与 CLI 通过 `/moss/status` 的 `text_encoder_dtype` 判断服务实际精度、决定是否
需要重启。把 [`server/moss_service_status.py`](../server/moss_service_status.py) 放到
`<MOSS_ROOT>\ComfyUI\custom_nodes\moss_service\__init__.py`。

如果已经有 `moss_service` 节点，不要整文件覆盖，只合并这两个字段：

```python
"text_encoder_dtype": str(model_management.text_encoder_dtype()),
"loaded_dtypes": loaded_dtypes,   # [{name, dtype, dtypes:[{dtype,count}]}]
```

缺失时：插件会显示“实际 TE 未知”，切换会因无法确认而回滚（`fp8-fp8` 尤其明显）。

## 3. `start_service.py` 支持 `--text-enc`

Text Encoder 精度只能由启动参数决定。在现有脚本上做三处修改：

```python
# 1) 顶部
import argparse
import sys
sys.path.insert(0, str(ROOT / "tools"))
import moss_precision

# 2) main() 开头
parser = argparse.ArgumentParser(description="启动本地 ComfyUI + MOSS-Image 服务")
parser.add_argument("--text-enc", choices=moss_precision.TEXT_ENC_CHOICES, default=None,
                    help="Text Encoder 精度；默认取当前所选精度组合")
args = parser.parse_args()
text_enc = args.text_enc or moss_precision.level_of(
    moss_precision.load_state()["level"])["te_dtype"]

# 3) 启动参数里把固定的 --bf16-text-enc 换成
f"--{text_enc}-text-enc",
```

其余参数（`--lowvram`、`--disable-async-offload`、`--bf16-unet` 等）保持原样。
`moss-service.cmd restart` 不需要改动：不传 `--text-enc` 时会自动读取当前选择。

## 4. `moss-image` 节点补丁

官方节点默认会让检查点的 norm dtype 覆盖 `--fp8_e4m3fn-text-enc`；不补这两处，
第三档与第二档等价。

### 4.1 `text_encoder.py`：显式 fp8 参数优先

```python
# 顶部
import torch
```

```python
class MOSSImageTEModel_(MOSSImageTEModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        # 显式 fp8 启动参数优先；其余情况保持原行为（检查点 dtype 优先）
        if dtype is None or (dtype_llama is not None and dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)):
            dtype = dtype_llama
        if llama_quantization_metadata is not None:
            model_options = {**model_options, "quantization_metadata": llama_quantization_metadata}
        super().__init__(device=device, dtype=dtype, model_options=model_options)
```

### 4.2 `loaders.py`：一维小参数提升为 fp32

llama 的 `RMSNorm.scale()` 直接做 `weight + 1.0`，Qwen3.5 的 `GatedDeltaNet` 也会对
`A_log` / `dt_bias` 做标量运算，这些都没有 fp8 算子。把一维参数提升到 fp32（几 MB），
大矩阵仍以 fp8 存储：

```python
def promote_small_params(clip):
    for _, parameter in clip.cond_stage_model.named_parameters():
        if parameter.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and parameter.dim() <= 1:
            parameter.data = parameter.data.to(torch.float32)
    return clip
```

然后把 `load_clip_state()` 末尾的 `return clip` 改成：

```python
    return promote_small_params(clip)
```

## 5. 客户端契约

生成请求的客户端（本部署是 `moss` CLI）必须做三件事：

```python
import moss_precision

level = moss_precision.level_of(moss_precision.load_state()["level"])

# 1) DiT 精度写进工作流
workflow["1"]["inputs"]["weight_dtype"] = level["dit_dtype"]   # bf16 / fp8_e4m3fn

# 2) 组合 id 纳入幂等键（不同档位不复用同一结果）
key_material["precision"] = level["id"]

# 3) 提交前确认服务 TE 精度与所选组合一致（缺失状态时跳过）
status = moss_precision.service_telemetry() or {}
running = moss_precision.normalize_dtype(status.get("text_encoder_dtype"))
if running is not None and running != level["te_dtype"]:
    raise RuntimeError("服务 Text Encoder 与所选组合不一致，请先切换")
```

`state/precision.json` 形如：

```json
{ "version": 1, "level": "fp8-bf16", "updated_at": "2026-10-20T18:15:13+0800", "updated_by": "ui" }
```

## 6. 验证

```bat
"<MOSS_ROOT>\venv\Scripts\python.exe" "<MOSS_ROOT>\tools\moss_precision.py" --status
"<MOSS_ROOT>\venv\Scripts\python.exe" "<MOSS_ROOT>\tools\moss_precision.py" --set bf16-bf16
"<MOSS_ROOT>\venv\Scripts\python.exe" "<MOSS_ROOT>\tools\moss_precision.py" --set fp8-fp8
```

- `--set fp8-fp8` 会重启 ComfyUI（约 1–2 分钟），完成后 `--status` 应显示
  `实际 TE: torch.float8_e4m3fn 匹配`。
- 服务启动日志里应能看到
  `CLIP/text encoder model load ... dtype: torch.float8_e4m3fn` 和
  `model weight dtype torch.float8_e4m3fn`。

## 7. 回滚

1. 还原 `start_service.py` 与 `moss-image` 两处补丁；
2. 删除 `<MOSS_ROOT>\tools\moss_precision.py` 与 `state\precision.json`；
3. 在 DSH 插件页卸载本插件，并删除插件目录。

回滚不会改动任何模型权重文件。
