# 测试记录（2026-10-20）

本文件记录三档精度组合的资源与画质复核。所有数字都能在
[`docs/data/verification.json`](data/verification.json)、
[`docs/data/quality-apple-metrics.json`](data/quality-apple-metrics.json)、
[`docs/data/quality-text-metrics.json`](data/quality-text-metrics.json) 中找到。
未测量的项目在文末明确标为“未覆盖”。

## 条件

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB |
| 桌面空闲占用 | 约 1.37–1.45 GiB（整卡） |
| 分辨率 / 步数 | 512×512 / 8 步（文字压力测试 20 步） |
| seed | 42（主对比）、43（第二提示词） |
| 提示词 A | `a single red apple on a white ceramic plate, soft studio light` |
| 提示词 B | `a blue ceramic mug on a wooden table, soft daylight` |
| 文字提示词 | 咖啡店黑板「今日咖啡」，20 步，seed 42 |
| 服务启动 | 每档一个全新进程：`start_service.py --text-enc <bf16|fp8_e4m3fn>`，固定 `--lowvram --disable-async-offload --bf16-unet` |
| 采样 | 外部进程每 0.5 s：`nvidia-smi memory.used` + `psutil` 进程 RSS |
| 冷 / 热 | 冷＝新进程第一次生成（含模型加载）；热＝同进程第二次生成 |
| RSS | 工作集；生成结束后等 6 s 再取“生成后 RSS” |

> 数据文件（`docs/data/verification.json`）中的 `rss_*` 字段单位是 MiB；本文与 README
> 的表格换算为 GiB（1 GiB = 1024 MiB）。

## 资源结果

| 档位 | 冷峰值显存（MiB） | 冷峰值 RSS（GiB） | 生成后 RSS（GiB） | 热峰值显存（MiB） |
|---|---|---|---|---|
| `bf16-bf16` | 7406 / 7392（均值 7399） | 21.28 / 23.72 | 19.94 / 20.94 | 7434 / 7454 |
| `fp8-bf16` | 6757 / 6747（6752） | 23.73 / 23.08 | 21.29 / 20.93 | 7236 |
| `fp8-fp8` | 6756 / 6747 / 6779（6761） | 23.62 / 23.15 / 23.85 | 21.49 / 21.48 / 21.38 | 6692 / 7452 / 7185 |

判定：

- 显存：`bf16` 与两档 FP8 的冷峰值差约 **647 MiB（8.7%）**，档内波动 ≤32 MiB；
  `fp8-bf16` 与 `fp8-fp8` 只差 9 MiB（TE 在 CPU，不影响显存）。
- 内存：档间均值差 0.7–1.0 GiB，但 `bf16` 自身两轮漂移 2.4 GiB，排序会翻转；
  **无法判定任何一档更省内存**。
- 热峰值显存受上一轮驻留影响（6.1–6.6 GiB 基线），只作范围参考，不用于排序。

## dtype 证据

服务日志（每次运行前记录日志偏移，运行后截取）：

```text
bf16-bf16   model weight dtype torch.bfloat16, manual cast: None
            CLIP/text encoder model load ... dtype: torch.bfloat16
fp8-bf16    model weight dtype torch.float8_e4m3fn, manual cast: torch.bfloat16
            CLIP/text encoder model load ... dtype: torch.bfloat16
fp8-fp8     model weight dtype torch.float8_e4m3fn, manual cast: torch.bfloat16
            CLIP/text encoder model load ... dtype: torch.float8_e4m3fn
```

服务进程内 `/moss/status` 的 `loaded_dtypes`：

| 档位 | DiT（MOSSImage） | Text Encoder |
|---|---|---|
| `bf16-bf16` | 462 × bf16 | 759 × bf16 + 1 × fp32 |
| `fp8-bf16` | 462 × bf16 | 759 × bf16 + 1 × fp32 |
| `fp8-fp8` | 462 × bf16 | 470 × bf16 + 265 × fp32 + **25 × fp8** |

独立加载探测（服务停止、非 lowvram）：DiT fp8 = 462/462 fp8；TE fp8 =
385 fp8 + 375 fp32（375 个一维小参数被补丁提升为 fp32）。

> 解释：请求的 fp8 存储 dtype 在加载时成立；低显存驻留/搬运阶段 ComfyUI 会把大部分
> 权重转成计算 dtype，因此运行时 `parameters()` 看到的多为 bf16。**哪些层保留 fp8
> 未逐层枚举，标为不确定**；但显存峰值下降与输出差异都是实测的。

## 输出与确定性

| 档位 | seed 42 SHA-256（前 12） | seed 43 SHA-256（前 12） |
|---|---|---|
| `bf16-bf16` | `b74772e4343d` | `fab9d76d8066` |
| `fp8-bf16` | `2c9899213dea` | `f27cd48ccb40` |
| `fp8-fp8` | `c56e462e04a9` | `7968a879ff06` |

- 三档在同一 prompt/seed 下互不相同。
- 干净进程内 A→B→A 对照：bf16 两次 A 完全相同；`fp8-fp8` 两次 A 完全相同。
- 跨进程：bf16（3 次）、`fp8-bf16`（2 次）、`fp8-fp8`（3 个干净进程）均复现同一 SHA。

## 画质

指标计算：8-bit RGB；MAE/RMSE/PSNR 基于全部通道；SSIM 在灰度图上用
Gaussian σ=1.5 的加窗实现（scipy）。

| 对比（同 seed 42） | 苹果：PSNR / SSIM | 文字：PSNR / SSIM |
|---|---|---|
| BF16 vs FP8 DiT | 30.49 dB / 0.982 | 19.95 dB / 0.800 |
| FP8 DiT vs FP8 双 | 31.46 dB / 0.975 | 30.14 dB / 0.968 |
| BF16 vs FP8 双 | 28.91 dB / 0.965 | 19.77 dB / 0.801 |

- 苹果图：并排几乎一致，差异是细节/高光位移，无局部崩坏。
- 文字图：三档「今日咖啡」四字全部正确、笔画完整；文字组的低 SSIM 来自构图/机位
  变化，不是文字质量下降。
- 对照图：[苹果](assets/compare-apple-seed42.png)、
  [文字 2× 放大](assets/compare-text-zoom-2x.png)。

**结论**：像素差异真实；在这两个提示词上未观察到可辨的画质下降。差异更像同一
提示词的重新采样。不能外推到所有提示词。

## 异常

8 次冷启动生成中有 1 次（`fp8-fp8`）首次加载报：

```text
HostBuffer.read_file_slice failed
```

该进程没有复位，之后仍能出图，但同 seed 输出与干净进程不同
（`1cca065d…` / `9d7c8398…` vs 干净进程的 `c56e462e…` / `7968a879…`）。
建议：加载报错后重启服务，不要沿用当前进程。

## 未覆盖

- 其他分辨率、步数、模型版本；
- 多人、多手、密集小字、极细纹理等更苛刻场景；
- Linux / macOS；其他 GPU / 显存档位；
- 逐进程显存（Windows WDDM 下 `nvidia-smi --query-compute-apps` 返回 `[N/A]`）；
- 盲评或多人评审；
- 数小时连续运行的稳定性与内存泄漏。

## 复现说明

复核脚本与部署强耦合（读取本机 `comfyui.log`、PID、服务状态），未随仓库发布；
本文件给出的协议、命令与原始数据足以复现同一组指标。复现时请保持：每档全新进程、
固定 seed、外部 0.5 s 采样、生成后等待 6 s 再读 RSS。
