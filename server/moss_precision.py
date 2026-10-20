#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MOSS-Image 精度组合：选择、持久化、服务重启与应用状态。

三档组合（只包含前三档，不含 DiT FP16）：

| id          | DiT weight_dtype | Text Encoder 启动参数     |
|-------------|------------------|---------------------------|
| bf16-bf16   | bf16             | --bf16-text-enc           |
| fp8-bf16    | fp8_e4m3fn       | --bf16-text-enc           |
| fp8-fp8     | fp8_e4m3fn       | --fp8_e4m3fn-text-enc     |

- DiT 精度由 MOSSImageHFLoader.weight_dtype 控制，跟随每次请求的工作流，
  不需要重启服务，也不需要重新下载/替换权重（bf16 完整权重在加载时转 FP8）。
- Text Encoder 精度只能由 ComfyUI 启动参数决定，因此切换 TE 精度会重启本地
  ComfyUI 服务；同一 TE 精度内切换（DiT bf16 ↔ fp8）不会重启。
- 未选择时使用第一档 bf16-bf16。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_FILE = STATE_DIR / "precision.json"
LOCK_FILE = STATE_DIR / "precision.lock"
PID_FILE = STATE_DIR / "comfyui.pid"
START_SCRIPT = ROOT / "start_service.py"
STOP_SCRIPT = ROOT / "stop_service.py"
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
API = os.environ.get("MOSS_API_URL", "http://127.0.0.1:8188").rstrip("/")
HOST = "127.0.0.1"
PORT = 8188
STATE_VERSION = 1
LOCK_STALE_SECONDS = 900

LEVELS = (
    {
        "id": "bf16-bf16",
        "label": "DiT BF16 + Text Encoder BF16",
        "short": "画质",
        "description": "画质最好，内存占用最高（默认）",
        "dit_dtype": "bf16",
        "te_dtype": "bf16",
        "te_arg": "--bf16-text-enc",
    },
    {
        "id": "fp8-bf16",
        "label": "DiT FP8 + Text Encoder BF16",
        "short": "预览",
        "description": "显存略省，画质略降，用于快速预览",
        "dit_dtype": "fp8_e4m3fn",
        "te_dtype": "bf16",
        "te_arg": "--bf16-text-enc",
    },
    {
        "id": "fp8-fp8",
        "label": "DiT FP8 + Text Encoder FP8",
        "short": "低显存",
        "description": "内存明显降低，画质有明显风险，不推荐",
        "dit_dtype": "fp8_e4m3fn",
        "te_dtype": "fp8_e4m3fn",
        "te_arg": "--fp8_e4m3fn-text-enc",
    },
)
LEVEL_BY_ID = {level["id"]: level for level in LEVELS}
# 允许 --text-enc 直接使用的 ComfyUI 取值（与 cli_args.py 保持一致）。
TEXT_ENC_CHOICES = ("bf16", "fp8_e4m3fn", "fp8_e5m2", "fp16", "fp32")
TEXT_ENC_ARG = {name: f"--{name}-text-enc" for name in TEXT_ENC_CHOICES}


def out(text: str = "") -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(text, flush=True)


def default_level_id() -> str:
    return LEVELS[0]["id"]


def level_of(level_id: str | None) -> dict:
    """按 id 取档位；未知/缺失一律回落到默认档（不猜测用户意图）。"""
    return LEVEL_BY_ID.get(level_id or "", LEVELS[0])


def levels_payload() -> list[dict]:
    return [dict(level) for level in LEVELS]


# ---------------------------------------------------------------- 状态文件
def load_state() -> dict:
    """读取选择；文件缺失或损坏时返回默认档（不写盘）。"""
    fallback = {
        "version": STATE_VERSION,
        "level": default_level_id(),
        "updated_at": None,
        "updated_by": None,
    }
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
    if not isinstance(raw, dict):
        return fallback
    level = raw.get("level")
    fallback.update(
        {
            "level": level if level in LEVEL_BY_ID else default_level_id(),
            "updated_at": raw.get("updated_at") if isinstance(raw.get("updated_at"), str) else None,
            "updated_by": raw.get("updated_by") if isinstance(raw.get("updated_by"), str) else None,
        }
    )
    return fallback


def save_state(level_id: str, by: str | None) -> dict:
    if level_id not in LEVEL_BY_ID:
        raise ValueError(f"未知精度组合：{level_id}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": STATE_VERSION,
        "level": level_id,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "updated_by": by or "unknown",
    }
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, STATE_FILE)
    return payload


@contextlib.contextmanager
def apply_lock(timeout: float = 600):
    """跨进程互斥：避免界面与命令行同时重启同一个服务。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {int(time.time())}".encode("ascii"))
            os.close(fd)
            break
        except FileExistsError:
            try:
                pid_text, started_text = LOCK_FILE.read_text(encoding="ascii").split()
                started = int(started_text)
            except (OSError, ValueError):
                pid_text, started = "0", 0
            if time.time() - started > LOCK_STALE_SECONDS:
                LOCK_FILE.unlink(missing_ok=True)
                continue
            if time.time() > deadline:
                raise RuntimeError(f"另一个精度切换仍在进行（PID {pid_text}），请稍后重试")
            time.sleep(0.5)
    try:
        yield
    finally:
        LOCK_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------- 服务探测
def http_json(path: str, timeout: float = 5) -> dict:
    req = urllib.request.Request(API + path, headers={"Accept": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else {}


def port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((HOST, PORT)) == 0


def service_telemetry(timeout: float = 5) -> dict | None:
    try:
        return http_json("/moss/status", timeout=timeout)
    except Exception:
        return None


def service_up() -> bool:
    try:
        http_json("/system_stats", timeout=3)
        return True
    except Exception:
        return False


def normalize_dtype(value: str | None) -> str | None:
    """`torch.bfloat16` → `bf16`、`torch.float8_e4m3fn` → `fp8_e4m3fn` 等。"""
    if not value:
        return None
    text = str(value).lower()
    if text.startswith("torch."):
        text = text[len("torch."):]
    aliases = {
        "bfloat16": "bf16",
        "float32": "fp32",
        "float16": "fp16",
        "float8_e4m3fn": "fp8_e4m3fn",
        "float8_e4m3fnuz": "fp8_e4m3fnuz",
        "float8_e5m2": "fp8_e5m2",
        "float8_e5m2fnuz": "fp8_e5m2fnuz",
    }
    return aliases.get(text, text)


def service_report(level: dict) -> dict:
    up = service_up()
    telemetry = service_telemetry() if up else None
    running = normalize_dtype((telemetry or {}).get("text_encoder_dtype"))
    return {
        "up": up,
        "port_open": port_open(),
        "pid": (telemetry or {}).get("pid"),
        "uptime_seconds": (telemetry or {}).get("uptime_seconds"),
        "text_encoder_dtype": (telemetry or {}).get("text_encoder_dtype"),
        "text_encoder_match": (running == level["te_dtype"]) if running else None,
        "loaded": (telemetry or {}).get("loaded_dtypes") or [],
        "restart_required": bool(up and running and running != level["te_dtype"]),
    }


def status() -> dict:
    state = load_state()
    level = level_of(state["level"])
    service = service_report(level)
    warnings: list[str] = []
    if service["up"] and service["text_encoder_match"] is False:
        warnings.append(
            f"服务当前 Text Encoder 为 {service['text_encoder_dtype']}，与所选组合的 "
            f"{level['te_dtype']} 不一致；选择该组合或重启服务后生效。"
        )
    if not service["up"]:
        warnings.append("ComfyUI 服务未运行；所选组合会在下次启动服务时生效。")
    return {
        "ok": True,
        "level": level["id"],
        "level_index": list(LEVEL_BY_ID).index(level["id"]),
        "label": level["label"],
        "short": level["short"],
        "description": level["description"],
        "dit_dtype": level["dit_dtype"],
        "te_dtype": level["te_dtype"],
        "te_arg": level["te_arg"],
        "selected_at": state["updated_at"],
        "selected_by": state["updated_by"],
        "levels": levels_payload(),
        "service": service,
        "warnings": warnings,
    }


# ---------------------------------------------------------------- 启停服务
def read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def stop_service(wait_seconds: float = 30) -> dict:
    pid = read_pid()
    result = {"ok": True, "pid": pid, "stopped": False}
    if pid is None:
        if port_open():
            return {"ok": False, "error": f"端口 {PORT} 被占用，但没有可用的 PID 记录；请手动检查"}
        return result
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    PID_FILE.unlink(missing_ok=True)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not port_open():
            result["stopped"] = True
            return result
        time.sleep(0.5)
    return {"ok": False, "pid": pid, "error": f"{wait_seconds:.0f} 秒内端口 {PORT} 仍未释放"}


def start_service(text_enc: str, wait_seconds: float = 420) -> dict:
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    cmd = [str(PYTHON), str(START_SCRIPT), "--text-enc", text_enc]
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=wait_seconds, env=env,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip()[-4000:],
        "stderr": (proc.stderr or "").strip()[-4000:],
    }


# ---------------------------------------------------------------- 应用选择
def apply_level(
    level_id: str,
    by: str | None = None,
    restart: bool = True,
    timeout: float = 420,
) -> dict:
    """写入选择并让运行中的服务与之对齐；失败时回滚选择。

    - 服务未运行：只持久化选择，下次启动时生效。
    - TE 精度未变：不重启，DiT 精度在下次请求时由工作流生效。
    - TE 精度变化：重启 ComfyUI（不下载、不替换任何权重文件）。
    """
    if level_id not in LEVEL_BY_ID:
        return {"ok": False, "error": f"未知精度组合：{level_id}"}
    level = LEVEL_BY_ID[level_id]
    previous = load_state()
    with apply_lock():
        saved = save_state(level_id, by)
        try:
            service = service_report(level)
            if not service["up"]:
                return {
                    "ok": True, "level": level_id, "changed": previous["level"] != level_id,
                    "restarted": False, "applied": False,
                    "note": "服务未运行；已记录选择，下次启动服务时生效。",
                    "service": service, "state": saved,
                }
            if service["text_encoder_match"] is True:
                return {
                    "ok": True, "level": level_id, "changed": previous["level"] != level_id,
                    "restarted": False, "applied": True,
                    "note": "Text Encoder 精度未变，无需重启；DiT 精度随下次请求生效。",
                    "service": service, "state": saved,
                }
            if not restart:
                save_state(previous["level"], previous["updated_by"])
                return {
                    "ok": False, "level": level_id, "restarted": False, "applied": False,
                    "error": "需要重启 ComfyUI 才能切换 Text Encoder 精度（已回滚选择）",
                    "service": service,
                }
            started_at = time.time()
            stopped = stop_service()
            if not stopped.get("ok"):
                raise RuntimeError(stopped.get("error") or "停止 ComfyUI 失败")
            started = start_service(level["te_dtype"])
            elapsed = round(time.time() - started_at, 1)
            if not started.get("ok"):
                raise RuntimeError(
                    "启动 ComfyUI 失败：" + (started.get("stderr") or started.get("stdout") or "未知错误")
                )
            service = service_report(level)
            if service["text_encoder_match"] is not True:
                raise RuntimeError(
                    f"服务已启动，但 Text Encoder 实际为 {service['text_encoder_dtype']}，"
                    f"与所选组合 {level['te_dtype']} 不一致"
                )
            return {
                "ok": True, "level": level_id, "changed": previous["level"] != level_id,
                "restarted": True, "applied": True, "elapsed_seconds": elapsed,
                "note": "已重启 ComfyUI 以应用 Text Encoder 精度。",
                "service": service, "state": saved,
            }
        except Exception as error:
            save_state(previous["level"], previous["updated_by"])
            return {
                "ok": False, "level": level_id, "restarted": False, "applied": False,
                "error": str(error), "service": service_report(level),
            }


# ---------------------------------------------------------------- CLI
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="moss-precision",
        description="选择并应用 MOSS-Image 精度组合（DiT weight_dtype + Text Encoder 启动参数）",
    )
    parser.add_argument("--set", metavar="LEVEL", help="应用一个精度组合：" + ", ".join(LEVEL_BY_ID))
    parser.add_argument("--status", action="store_true", help="显示当前选择与服务实际精度")
    parser.add_argument("--levels", action="store_true", help="列出三档组合")
    parser.add_argument("--state", action="store_true", help="输出完整状态（界面/插件使用）")
    parser.add_argument("--by", default=None, help="调用来源，写入状态文件（ui / cli / tool）")
    parser.add_argument("--no-restart", action="store_true", help="不重启服务；需要重启时直接失败并回滚")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.levels:
        payload = levels_payload()
        out(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else
            "\n".join(f"{i + 1}. {lv['id']}  {lv['label']}  DiT={lv['dit_dtype']} TE={lv['te_dtype']}"
                      for i, lv in enumerate(payload)))
        return 0
    if args.set:
        result = apply_level(args.set, by=args.by or "cli", restart=not args.no_restart)
        if args.json:
            out(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            if result.get("ok"):
                out(f"精度组合：{result['level']}  ({LEVEL_BY_ID[result['level']]['label']})")
                out(result.get("note", ""))
                service = result.get("service") or {}
                if service.get("up"):
                    out(f"服务：运行中  PID={service.get('pid')}  "
                        f"TextEncoder={service.get('text_encoder_dtype')}")
            else:
                out(f"切换失败：{result.get('error')}")
        return 0 if result.get("ok") else 1
    if args.state:
        out(json.dumps(status(), ensure_ascii=False))
        return 0
    if args.json:
        out(json.dumps(status(), ensure_ascii=False, indent=2))
        return 0
    payload = status()
    out("MOSS-Image 精度组合")
    out("=" * 58)
    out(f"当前选择   : {payload['level']}  {payload['label']}")
    out(f"DiT        : {payload['dit_dtype']}   (节点 weight_dtype)")
    out(f"Text Enc.  : {payload['te_dtype']}   (启动参数 {payload['te_arg']})")
    out(f"选择时间   : {payload['selected_at'] or '未选择（使用默认档）'}"
        f"{'  来源=' + payload['selected_by'] if payload['selected_by'] else ''}")
    service = payload["service"]
    out(f"服务       : {'运行中' if service['up'] else '未运行'}"
        + (f"  PID={service['pid']}  运行 {service['uptime_seconds']}s" if service["up"] else ""))
    if service["up"]:
        out(f"实际 TE    : {service['text_encoder_dtype'] or '未知'}  "
            f"{'匹配' if service['text_encoder_match'] else '不匹配' if service['text_encoder_match'] is False else '无法确认'}")
    for warning in payload["warnings"]:
        out(f"注意       : {warning}")
    out("=" * 58)
    out("三档：")
    for index, level in enumerate(LEVELS, 1):
        mark = "→" if level["id"] == payload["level"] else " "
        out(f" {mark} {index}. {level['id']:<10} {level['label']}  — {level['description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
