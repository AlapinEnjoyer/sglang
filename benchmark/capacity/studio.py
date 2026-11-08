"""Benchmark studio: plan, run and compare SGLang serving setups across machines.

  studio.py plan --model DIR [--gpu H100:80:2] [--context 131072] [--write-suite FILE]
      How many users of a given context fit, and the serve flags for them.
  studio.py run SUITE.toml [--only a,b] [--out-dir DIR]
      Launch each [[configs]] entry in turn and run the suite's steps against it:
      cache (prefix reuse), gsm8k (sgl-eval), burst (capacity_bench), multiturn
      (bench_multiturn sessions, optional /v1/decisions load) and disk (a session
      restored from the HiCache storage tier, also across a restart).
  studio.py report RUN_DIR [RUN_DIR ...] [--out FILE]
      One comparison report over runs from any number of machines.

See benchmark/capacity/README.md for the suite format.
"""

import argparse
import datetime
import json
import os
import random
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cache_check
import capacity_bench as cb
import config_sweep as cs
import msgspec
import requests
import studio_plan as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STEP_ORDER = ("cache", "gsm8k", "burst", "multiturn", "disk")


# ---------------------------------------------------------------- suite format


class CacheStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    prompt_len: int = 4000


class Gsm8kStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    num_examples: int = 250
    num_threads: int = 8
    max_tokens: int = 2048
    # Greedy, thinking off: the convention of the Ampere results in this repo.
    extra_args: str = "--temperature 0 --chat-template-kwarg enable_thinking=false"
    # Empty means `uvx --from sgl-eval==<version pinned in python/pyproject.toml> sgl-eval`.
    command: str = ""


class BurstStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    args: str = "--context-lens 32k --users 1,4,8 --repeats 2"


class MultiturnStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    clients: List[int] = msgspec.field(default_factory=lambda: [10, 20])
    rounds: int = 6
    request_length: int = 4000
    output_length: int = 500
    request_rate: float = 0.5
    max_parallel: int = 8
    decision_rate: float = 0.0
    session_ids: bool = False
    priority: Optional[int] = None


class DiskStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    session_len: int = 30000
    restart: bool = True


class Steps(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    cache: Optional[CacheStep] = None
    gsm8k: Optional[Gsm8kStep] = None
    burst: Optional[BurstStep] = None
    multiturn: Optional[MultiturnStep] = None
    disk: Optional[DiskStep] = None


class StudioConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    server_args: str = ""
    env: Dict[str, str] = {}
    launch: str = ""
    # Subset of the suite's steps for this config; None runs all of them.
    steps: Optional[List[str]] = None
    # Per-config overrides of the suite's step settings.
    step_overrides: Steps = Steps()


class StudioSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    configs: List[StudioConfig]
    launch: str = "sglang serve"
    port: int = 30000
    startup_timeout: float = 1800
    server_args: str = ""
    env: Dict[str, str] = {}
    steps: Steps = Steps()


def load_spec(path: Path) -> StudioSpec:
    import tomllib

    return msgspec.convert(tomllib.loads(path.read_text()), type=StudioSpec)


def _steps_for(spec: StudioSpec, cfg: StudioConfig) -> Dict[str, Any]:
    out = {}
    for name in STEP_ORDER:
        step = getattr(cfg.step_overrides, name) or getattr(spec.steps, name)
        if step is not None and (cfg.steps is None or name in cfg.steps):
            out[name] = step
    return out


# ---------------------------------------------------------------- machine


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def disk_read_mb_s(directory: Path, size_mb: int = 1024) -> Optional[float]:
    """Sequential read speed with O_DIRECT, so the page cache does not answer."""
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".studio_disk_probe"
    try:
        _run(
            [
                "dd",
                "if=/dev/zero",
                f"of={probe}",
                "bs=4M",
                f"count={size_mb // 4}",
                "oflag=direct",
                "status=none",
            ]
        )
        out = subprocess.run(
            ["dd", f"if={probe}", "of=/dev/null", "bs=4M", "iflag=direct"],
            capture_output=True,
            text=True,
            timeout=600,
        ).stderr
        m = re.search(r"copied, ([\d.]+) s", out)
        return size_mb / float(m[1]) if m else None
    finally:
        probe.unlink(missing_ok=True)


def gather_machine(storage_dir: Optional[Path]) -> Dict[str, Any]:
    gpus = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap,driver_version,"
            "pcie.link.gen.max,pcie.link.width.max,power.limit",
            "--format=csv,noheader",
        ]
    )
    topo = _run(["nvidia-smi", "topo", "-m"])
    mem = {
        l.split(":")[0]: l.split()[1]
        for l in Path("/proc/meminfo").read_text().splitlines()
    }
    cpu = next(
        (
            l.split(":", 1)[1].strip()
            for l in Path("/proc/cpuinfo").read_text().splitlines()
            if l.startswith("model name")
        ),
        "",
    )
    sglang_rev = _run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "host": socket.gethostname(),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "gpus": [g.strip() for g in gpus.splitlines()],
        "nvlink": "NV" in topo,
        "cpu": cpu,
        "ram_gb": int(mem["MemTotal"]) * 1024 / 1e9,
        "ram_available_gb": int(mem["MemAvailable"]) * 1024 / 1e9,
        "repo": f"{branch}@{sglang_rev}",
        "storage_dir": str(storage_dir) if storage_dir else None,
        "storage_read_mb_s": disk_read_mb_s(storage_dir) if storage_dir else None,
    }


# ---------------------------------------------------------------- server


def _expand(value: str, where: str) -> str:
    out = os.path.expandvars(value)
    if "$" in out:
        raise SystemExit(f"{where}: unset environment variable in {value!r}")
    return out


def server_command(spec: StudioSpec, cfg: StudioConfig) -> List[str]:
    return (
        cs.launch_prefix(cfg.launch or spec.launch)
        + cs.split_args(spec.server_args, where="server_args")
        + cs.split_args(cfg.server_args, where=f"{cfg.name}.server_args")
        + ["--port", str(spec.port)]
    )


_POOL_PATTERNS = {
    "mamba_slots": r"max_mamba_cache_size: (\d+)",
    "host_kv_tokens": r"hierarchical KV host pool: (\d+) tokens",
    "host_state_slots": r"Mamba cache \((\d+) state slots",
    "weights_gb": r"Load weight end.*?mem usage=([\d.]+) GB",
    "free_after_init_gb": r"available_gpu_mem=([\d.]+) GB",
}


def pool_info(base_url: str, log: Path) -> Dict[str, Any]:
    si = requests.get(f"{base_url}/server_info", timeout=60).json()
    text = log.read_text(errors="ignore")
    info = {
        "version": si.get("version"),
        "kv_pool_tokens": si.get("max_total_num_tokens"),
        "max_running": cs.max_running(si),
        "page_size": si.get("page_size"),
        "context_len": si.get("context_length"),
        "model_path": si.get("model_path"),
    }
    for key, pat in _POOL_PATTERNS.items():
        m = re.search(pat, text)
        info[key] = float(m[1]) if m and "." in m[1] else (int(m[1]) if m else None)
    return info


class Server:
    """One launch of a config; relaunch() restarts it with the same command."""

    def __init__(self, spec: StudioSpec, cfg: StudioConfig, cfg_dir: Path):
        self.cmd = server_command(spec, cfg)
        self.env = {
            k: _expand(v, where=f"{cfg.name}.env.{k}")
            for k, v in {**spec.env, **cfg.env}.items()
        }
        self.base_url = f"http://127.0.0.1:{spec.port}"
        self.timeout = spec.startup_timeout
        self.cfg_dir = cfg_dir
        self.proc: Optional[subprocess.Popen] = None
        self.launches = 0

    def start(self) -> Optional[str]:
        """None once ready, else the reason it is not."""
        if cs.server_answers(self.base_url):
            raise SystemExit(f"{self.base_url} already serves something; stop it first")
        self.launches += 1
        self.log = (
            self.cfg_dir / f"server{'' if self.launches == 1 else self.launches}.log"
        )
        t0 = time.time()
        self.proc = cs.start_server(cmd=self.cmd, env=self.env, log=self.log)
        why = cs.wait_ready(
            proc=self.proc, base_url=self.base_url, timeout=self.timeout
        )
        self.startup_s = time.time() - t0
        return None if why is None else f"{why}: {cs.log_error(self.log)}"

    def stop(self) -> None:
        if self.proc is not None:
            cs.stop_server(self.proc)
            self.proc = None
            _wait_for_ram()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def _wait_for_ram(timeout: float = 120) -> None:
    """Pinned HiCache host memory is released a few seconds after the server exits."""
    deadline = time.time() + timeout
    start = sp.host_ram_gb()
    while time.time() < deadline:
        time.sleep(3)
        now = sp.host_ram_gb()
        if now - start < 0.5:
            return
        start = now


# ---------------------------------------------------------------- steps


def step_cache(server: Server, step: CacheStep, out: Path) -> Dict[str, Any]:
    return cache_check.check_reuse(server.base_url, prompt_len=step.prompt_len)


def _sgl_eval_command(step: Gsm8kStep) -> List[str]:
    if step.command:
        return shlex.split(step.command)
    pin = re.search(
        r'"sgl-eval==([^"]+)"', (REPO / "python" / "pyproject.toml").read_text()
    )
    return ["uvx", "--from", f"sgl-eval=={pin[1] if pin else '0.1.3'}", "sgl-eval"]


def step_gsm8k(server: Server, step: Gsm8kStep, out: Path) -> Dict[str, Any]:
    model = requests.get(f"{server.base_url}/v1/models", timeout=60).json()["data"][0][
        "id"
    ]
    eval_dir = out / "gsm8k"
    cmd = _sgl_eval_command(step) + [
        "run",
        "gsm8k",
        "--base-url",
        f"{server.base_url}/v1",
        "--model",
        model,
        "--num-examples",
        str(step.num_examples),
        "--num-threads",
        str(step.num_threads),
        "--max-tokens",
        str(step.max_tokens),
        *shlex.split(step.extra_args),
        "--out-dir",
        str(eval_dir),
    ]
    code = cs.run_bench(cmd=cmd, log=out / "gsm8k.log")
    metrics = sorted(eval_dir.glob("*/metrics.json"))
    if code != 0 or not metrics:
        return {"error": cs.log_error(out / "gsm8k.log")}
    agg = json.loads(metrics[-1].read_text())["aggregate"]
    return {
        "score": agg["score"],
        "truncated_rate": agg.get("truncated_rate"),
        "settings": step.extra_args,
        "num_examples": step.num_examples,
    }


def step_burst(server: Server, step: BurstStep, out: Path) -> Dict[str, Any]:
    burst_dir = out / "burst"
    cmd = [
        sys.executable,
        str(HERE / "capacity_bench.py"),
        "--base-url",
        server.base_url,
        *cs.split_args(step.args, where="burst.args"),
        "--out-dir",
        str(burst_dir),
    ]
    code = cs.run_bench(cmd=cmd, log=out / "burst.log")
    groups: Dict[str, List[cb.BurstResult]] = {}
    raw = burst_dir / "results.jsonl"
    for line in raw.read_text().splitlines() if raw.exists() else []:
        res = msgspec.convert(json.loads(line), type=cb.BurstResult)
        groups.setdefault(f"{res.context_len}x{res.users}", []).append(res)
    skipped = re.findall(r"skip (\S+) N=(\d+): (.*)", (out / "burst.log").read_text())
    return {
        "cells": {k: cb.median_burst(v) for k, v in groups.items()},
        "skipped": [f"{ctx} x {n}: {why}" for ctx, n, why in skipped],
        "error": None if code == 0 else cs.log_error(out / "burst.log"),
    }


def multiturn_ceiling(step: MultiturnStep) -> float:
    """Best possible hit rate: every turn's new user tokens are uncached by definition."""
    prompts = cached = 0
    for r in range(1, step.rounds + 1):
        prompts += r * step.request_length + (r - 1) * step.output_length
        cached += (r - 1) * (step.request_length + step.output_length)
    return cached / prompts


def _decision_load(
    step: MultiturnStep, server: Server, out: Path
) -> Optional[Callable]:
    if not step.decision_rate:
        return None
    stop_file, result = out / "decisions.stop", out / "decisions.json"
    stop_file.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "studio_workloads.py"),
            "decisions",
            "--url",
            f"{server.base_url}/v1/decisions",
            "--rate",
            str(step.decision_rate),
            "--stop",
            str(stop_file),
            "--out",
            str(result),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    def finish() -> Dict[str, Any]:
        stop_file.touch()
        proc.wait(timeout=1800)
        return json.loads(result.read_text()) if result.exists() else {}

    return finish


def step_multiturn(server: Server, step: MultiturnStep, out: Path) -> Dict[str, Any]:
    model = requests.get(f"{server.base_url}/model_info", timeout=60).json()[
        "model_path"
    ]
    port = server.base_url.rsplit(":", 1)[1]
    runs = {}
    for clients in step.clients:
        cb.flush_cache(server.base_url)
        log_file = out / f"multiturn_{clients}c.jsonl"
        log_file.unlink(missing_ok=True)
        tagging = (["--session-ids"] if step.session_ids else []) + (
            ["--priority", str(step.priority)] if step.priority is not None else []
        )
        cmd = [
            sys.executable,
            str(HERE / "studio_workloads.py"),
            "multiturn",
            *tagging,
            "--model-path",
            model,
            "--port",
            port,
            "--num-clients",
            str(clients),
            "--max-parallel",
            str(step.max_parallel),
            "--num-rounds",
            str(step.rounds),
            "--request-length",
            str(step.request_length),
            "--output-length",
            str(step.output_length),
            "--request-rate",
            str(step.request_rate),
            "--disable-auto-run",
            "--log-file",
            str(log_file),
            "--tag",
            f"{clients}c",
        ]
        finish = _decision_load(step, server, out)
        code = cs.run_bench(cmd=cmd, log=out / f"multiturn_{clients}c.log")
        decisions = finish() if finish else None
        summary = (
            json.loads(log_file.read_text().splitlines()[-1])["summary"]
            if (code == 0 and log_file.exists())
            else {"error": cs.log_error(out / f"multiturn_{clients}c.log")}
        )
        runs[str(clients)] = {**summary, "decisions": decisions}
    return {
        "ceiling": multiturn_ceiling(step),
        "settings": msgspec.to_builtins(step),
        "runs": runs,
    }


def _generate(base_url: str, ids: List[int], max_new: int):
    t0 = time.perf_counter()
    r = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new,
                "ignore_eos": True,
            },
        },
        timeout=1800,
    )
    r.raise_for_status()
    return r.json(), time.perf_counter() - t0


def _next_turn(base_url: str, history: List[int]) -> Dict[str, Any]:
    turn = history + [random.randrange(1000, 100000) for _ in range(100)]
    out, dt = _generate(base_url, turn, 1)
    return {
        "prompt": out["meta_info"]["prompt_tokens"],
        "cached": out["meta_info"]["cached_tokens"],
        "first_token_s": dt,
    }


def step_disk(server: Server, step: DiskStep, out: Path) -> Dict[str, Any]:
    rng = random.Random()
    fresh = lambda: [rng.randrange(1000, 100000) for _ in range(step.session_len)]  # noqa: E731
    cold = _next_turn(server.base_url, fresh())
    prompt = fresh()
    first, _ = _generate(server.base_url, prompt, 200)
    history = prompt + first["output_ids"]
    res = {
        "session_len": step.session_len,
        "cold": cold,
        "gpu_hit": _next_turn(server.base_url, history),
    }
    # Flushing empties the GPU and host tiers; it waits until pending storage writes finish.
    cb.flush_cache(server.base_url)
    res["after_flush"] = _next_turn(server.base_url, history)
    if step.restart:
        server.stop()
        why = server.start()
        res["after_restart"] = (
            {"error": why} if why else _next_turn(server.base_url, history)
        )
    return res


STEPS = {
    "cache": step_cache,
    "gsm8k": step_gsm8k,
    "burst": step_burst,
    "multiturn": step_multiturn,
    "disk": step_disk,
}


# ---------------------------------------------------------------- run


def run_config(spec: StudioSpec, cfg: StudioConfig, cfg_dir: Path) -> Dict[str, Any]:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    server = Server(spec, cfg, cfg_dir)
    summary: Dict[str, Any] = {
        "name": cfg.name,
        "server_cmd": shlex.join(server.cmd),
        "env": server.env,
        "steps": {},
    }
    print(f"\n=== {cfg.name}\n  $ {summary['server_cmd']}", flush=True)
    why = server.start()
    try:
        if why:
            summary["status"] = f"launch failed: {why}"
            return summary
        summary["startup_s"] = server.startup_s
        summary["pools"] = pool_info(server.base_url, server.log)
        print(
            f"  ready after {server.startup_s:.0f} s; pools {summary['pools']}",
            flush=True,
        )
        for name, step in _steps_for(spec, cfg).items():
            if not server.alive():
                summary["steps"][name] = {"error": "server not running"}
                continue
            print(f"  -- {name}", flush=True)
            try:
                summary["steps"][name] = STEPS[name](server, step, cfg_dir)
            except (requests.RequestException, RuntimeError, OSError, KeyError) as e:
                summary["steps"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
            _write_json(cfg_dir / "summary.json", summary)
        summary["status"] = (
            "ok" if server.alive() else f"server crashed: {cs.log_error(server.log)}"
        )
        return summary
    finally:
        server.stop()
        _write_json(cfg_dir / "summary.json", summary)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, default=str))


def cmd_run(args: argparse.Namespace) -> None:
    spec = load_spec(args.suite)
    only = set(args.only.split(",")) if args.only else None
    out = (
        args.out_dir
        or HERE / "results" / f"studio-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    )
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.suite, out / "suite.toml")
    envs = [spec.env, *(c.env for c in spec.configs), os.environ]
    storage = next(
        (
            e["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"]
            for e in envs
            if e.get("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR")
        ),
        "",
    )
    storage = os.path.expandvars(storage)
    storage = "" if "$" in storage else storage
    _write_json(
        out / "machine.json", gather_machine(Path(storage) if storage else None)
    )
    for cfg in spec.configs:
        if only and cfg.name not in only:
            continue
        if args.resume and (out / cfg.name / "summary.json").exists():
            if (
                json.loads((out / cfg.name / "summary.json").read_text()).get("status")
                == "ok"
            ):
                continue
        run_config(spec, cfg, out / cfg.name)
        write_report([out], out / "report.md")
    print(f"\nreport: {out / 'report.md'}")


# ---------------------------------------------------------------- report


def _f(v: Any, spec: str = ".1f") -> str:
    return "-" if v is None else format(v, spec)


def _load_run(run_dir: Path) -> Dict[str, Any]:
    machine = json.loads((run_dir / "machine.json").read_text())
    configs = [
        json.loads(p.read_text()) for p in sorted(run_dir.glob("*/summary.json"))
    ]
    return {"dir": run_dir, "machine": machine, "configs": configs}


def _gpu_label(machine: Dict[str, Any]) -> str:
    gpus = machine["gpus"]
    name = gpus[0].split(",")[0] if gpus else "?"
    return f"{len(gpus)}x {name}" + (" NVLink" if machine.get("nvlink") else "")


def _machines_table(runs: List[Dict[str, Any]]) -> List[str]:
    rows = [
        "| Run | GPUs | RAM | Storage read | Repo | Date |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs:
        m = r["machine"]
        rows.append(
            f"| {r['dir'].name} | {_gpu_label(m)} | {m['ram_gb']:.0f} GB | "
            f"{_f(m.get('storage_read_mb_s'), '.0f')} MB/s | {m['repo']} | {m['date']} |"
        )
    return rows


def _configs_table(runs: List[Dict[str, Any]]) -> List[str]:
    rows = [
        "| Config | GPUs | Status | Start | KV pool | Running | GDN slots GPU / host | "
        "Host KV | GSM8K | Cache reuse |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        for c in r["configs"]:
            p, s = c.get("pools", {}), c["steps"]
            gsm = s.get("gsm8k", {}).get("score")
            cache = s.get("cache", {}).get("ok")
            rows.append(
                f"| {c['name']} | {_gpu_label(r['machine'])} | {c.get('status', '?')[:60]} | "
                f"{_f(c.get('startup_s'), '.0f')} s | {p.get('kv_pool_tokens', '-')} | "
                f"{p.get('max_running', '-')} | {p.get('mamba_slots', '-')} / "
                f"{p.get('host_state_slots', '-')} | {p.get('host_kv_tokens', '-')} | "
                f"{'-' if gsm is None else f'{gsm:.1%}'} | {'-' if cache is None else ('ok' if cache else 'NO REUSE')} |"
            )
    return rows


def _burst_table(runs: List[Dict[str, Any]]) -> List[str]:
    rows = [
        "| Config | Context | Users | Decode tok/s per user | Total | TTFT last (s) | "
        "Prefill tok/s | Accept len |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        for c in r["configs"]:
            burst = c["steps"].get("burst") or {}
            for key, cell in sorted(
                burst.get("cells", {}).items(),
                key=lambda kv: tuple(map(int, kv[0].split("x"))),
            ):
                rows.append(
                    f"| {c['name']} | {cell['context_len']} | {cell['users']} | "
                    f"{_f(cell['decode_tps_per_user'])} | {_f(cell['decode_tps_total'])} | "
                    f"{_f(cell['ttft_last'])} | {_f(cell['prefill_tps_first_user'], '.0f')} | "
                    f"{_f(cell['accept_length'], '.2f')} |"
                )
            for s in burst.get("skipped", []):
                rows.append(f"| {c['name']} | {s} | | | | | | |")
    return rows


def _multiturn_table(runs: List[Dict[str, Any]]) -> List[str]:
    rows = [
        "| Config | Sessions | Hit rate (ceiling) | Avg / P99 TTFT (s) | Median TPOT (ms) | "
        "Req/s done (offered) | Decisions ok, median s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        for c in r["configs"]:
            mt = c["steps"].get("multiturn")
            if not mt:
                continue
            for clients, s in mt["runs"].items():
                if "error" in s:
                    rows.append(
                        f"| {c['name']} | {clients} | {s['error'][:80]} | | | | |"
                    )
                    continue
                d = s.get("decisions") or {}
                rows.append(
                    f"| {c['name']} | {clients} | {s['cache_hit_rate']:.3f} ({mt['ceiling']:.3f}) | "
                    f"{s['average_ttft']:.1f} / {s['p99_ttft']:.1f} | {s['median_tpot'] * 1000:.0f} | "
                    f"{s['throughput']:.2f} ({s['request_rate']}) | "
                    f"{(str(d.get('ok')) + ', ' + _f(d.get('median_s'), '.2f')) if d else '-'} |"
                )
    return rows


def _disk_table(runs: List[Dict[str, Any]]) -> List[str]:
    rows = [
        "| Config | Session | Cold prefill | GPU hit | After flush (storage) | After restart |",
        "|---|---|---|---|---|---|",
    ]
    cell = lambda t: (
        "-"
        if not t
        else (
            t["error"][:40]
            if "error" in t
            # noqa: E731
            else f"{t['first_token_s']:.2f} s, {t['cached']} cached"
        )
    )
    for r in runs:
        for c in r["configs"]:
            d = c["steps"].get("disk")
            if d and "error" not in d:
                rows.append(
                    f"| {c['name']} | {d['session_len']} | {cell(d['cold'])} | "
                    f"{cell(d['gpu_hit'])} | {cell(d['after_flush'])} | "
                    f"{cell(d.get('after_restart'))} |"
                )
    return rows


def write_report(run_dirs: List[Path], out: Path) -> None:
    runs = [_load_run(d) for d in run_dirs if (d / "machine.json").exists()]
    lines = [
        "# Benchmark studio report",
        "",
        "## Machines",
        "",
        *_machines_table(runs),
        "",
        "## Configs",
        "",
        *_configs_table(runs),
        "",
        "## Burst: N users arrive together",
        "",
        *_burst_table(runs),
        "",
        "## Multi-turn sessions",
        "",
        "Ceiling: the hit rate if every turn's history were cached; new user tokens never are.",
        "",
        *_multiturn_table(runs),
        "",
        "## Session restore from the storage tier",
        "",
        *_disk_table(runs),
        "",
        "## Server commands",
        "",
    ]
    for r in runs:
        for c in r["configs"]:
            lines += [f"- **{c['name']}** ({r['dir'].name}): `{c['server_cmd']}`"]
    out.write_text("\n".join(lines) + "\n")


def cmd_report(args: argparse.Namespace) -> None:
    out = args.out or args.run_dirs[0] / "report.md"
    write_report(args.run_dirs, out)
    print(out.read_text())


# ---------------------------------------------------------------- plan


def cmd_plan(args: argparse.Namespace) -> None:
    m = sp.model_sizes(args.model)
    gpu = sp.parse_gpu(args.gpu) if args.gpu else sp.local_gpus()
    if gpu is None:
        raise SystemExit("no local GPU found; pass --gpu NAME:MEM_GB[:COUNT[:SM]]")
    ram = args.ram_gb if args.ram_gb is not None else sp.host_ram_gb()
    choice = sp.ServeChoice(
        kv_dtype=args.kv_dtype,
        state_dtype=args.state_dtype,
        draft_quant=args.draft_quant,
        token_map_vocab=None if args.no_token_map else 32768,
        chunk=args.chunk,
        language_model_only=args.language_model_only,
        idle_slots=args.idle_slots,
        spec=not args.no_spec,
    )
    kv_tok = sp.kv_bytes_per_token(m, choice)
    print(
        f"{args.model.name}: {m.target_bytes / 1e9:.1f} GB weights, "
        f"{kv_tok / 1024:.1f} KiB KV per token, "
        f"{sp.state_bytes_per_slot(m, choice) / 1e6:.0f} MB per GDN state slot"
    )
    print(
        f"{gpu.count}x {gpu.name} ({gpu.mem_gib:.1f} GiB, SM {gpu.sm}), "
        f"{ram:.0f} GB RAM available\n"
    )
    print(
        "| TP | mem fraction | KV pool | Users at context | GDN slots | "
        "hicache GB/GPU | Host KV tokens | Host GDN slots |"
    )
    print("|---|---|---|---|---|---|---|---|")
    plans = []
    tp = 1
    while tp <= gpu.count:
        p = sp.plan_for(m, gpu, choice, tp, args.context, ram)
        plans.append(p)
        print(
            f"| {tp} | {p.mem_fraction:.2f} | {p.kv_tokens:,} | {p.users_at_context} | "
            f"{p.mamba_slots} | {p.hicache_gb_per_gpu} | {p.host_kv_tokens:,} | "
            f"{p.host_state_slots} |" + (f"  {'; '.join(p.notes)}" if p.notes else "")
        )
        tp *= 2
    best = max(plans, key=lambda p: (p.users_at_context, -p.tp))
    session_gb = args.context * kv_tok / 1e9
    print(
        f"\nOne {args.context}-token session: {session_gb:.2f} GB of KV. "
        f"For S sessions kept warm beyond the GPU, RAM or disk needs about S x {session_gb:.1f} GB."
    )
    print(
        f"\nSuggested flags (tp={best.tp}, {best.users_at_context} users at {args.context}):\n"
    )
    print("  " + sp.server_flags(best, choice, args.context, str(args.model)))
    if args.write_suite:
        args.write_suite.write_text(_suite_for(best, choice, args))
        print(f"\nwrote {args.write_suite}")


def _suite_for(p: sp.Plan, choice: sp.ServeChoice, args: argparse.Namespace) -> str:
    users = sorted({1, max(1, p.max_running // 2), p.max_running})
    flags = sp.server_flags(p, choice, args.context, "$MODEL").replace(
        " \\\n  ", "\n  "
    )
    return f'''# Generated by `studio.py plan`; check the flags against the model's cookbook page.
launch = "sglang serve"
port = 30000
server_args = """
  --served-model-name local --host 127.0.0.1 --disable-prefill-cuda-graph
  --enable-mixed-chunk --prefill-decode-interval 4
"""

[env]
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK = "1"

[steps.cache]
[steps.gsm8k]
[steps.burst]
args = "--context-lens 32k,fit --users {",".join(map(str, users))} --repeats 2 --sampling-params '{{\\"temperature\\":1.0,\\"top_p\\":0.95,\\"top_k\\":20}}'"
[steps.multiturn]
clients = [{p.max_running}, {2 * p.max_running}]

[[configs]]
name = "planned-tp{p.tp}"
server_args = """
  {flags}
"""
'''


# ---------------------------------------------------------------- cli


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Launch the suite's configs and run its steps.")
    run.add_argument("suite", type=Path)
    run.add_argument("--only", default=None, help="Comma-separated config names.")
    run.add_argument("--out-dir", type=Path, default=None)
    run.add_argument(
        "--resume", action="store_true", help="Skip configs already marked ok."
    )
    rep = sub.add_parser("report", help="Compare one or more run directories.")
    rep.add_argument("run_dirs", type=Path, nargs="+")
    rep.add_argument("--out", type=Path, default=None)
    plan = sub.add_parser("plan", help="Capacity plan and serve flags for a GPU setup.")
    plan.add_argument("--model", type=Path, required=True)
    plan.add_argument(
        "--gpu", default=None, help="NAME:MEM_GB[:COUNT[:SM]]; default: local."
    )
    plan.add_argument("--context", type=int, default=131072, help="Tokens per user.")
    plan.add_argument(
        "--ram-gb", type=float, default=None, help="Default: available RAM."
    )
    plan.add_argument("--kv-dtype", default="fp8_e4m3")
    plan.add_argument(
        "--state-dtype", default="bfloat16", choices=["bfloat16", "float32"]
    )
    plan.add_argument("--draft-quant", default="fp8", choices=["fp8", "bf16"])
    plan.add_argument("--no-token-map", action="store_true")
    plan.add_argument("--no-spec", action="store_true")
    plan.add_argument("--chunk", type=int, default=2048)
    plan.add_argument("--idle-slots", type=int, default=sp.DEFAULT_IDLE_SLOTS)
    plan.add_argument("--language-model-only", action="store_true")
    plan.add_argument("--write-suite", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    {"run": cmd_run, "report": cmd_report, "plan": cmd_plan}[args.cmd](args)


if __name__ == "__main__":
    main()
