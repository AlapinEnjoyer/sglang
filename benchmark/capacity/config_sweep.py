"""Benchmark several server configurations back to back and compare them.

Each [[configs]] entry of a TOML file is one server launch. For every entry the sweep
starts the server, waits until it answers /health_generate, runs capacity_bench.py
against it, stops it, and moves on. A launch that fails (OOM, rejected flag) is recorded
and skipped. summary.md compares every config per (context length, users) cell.

Usage:
  MODEL=/models/Qwen3.8-27B python benchmark/capacity/config_sweep.py \
      benchmark/capacity/configs/qwen3.8-27b-1gpu.toml --slo-decode 30 --slo-ttft 60

See benchmark/capacity/README.md for the file format.
"""

import argparse
import datetime
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import capacity_bench as cb
import msgspec
import requests

BENCH_SCRIPT = Path(__file__).with_name("capacity_bench.py")
ERROR_LINE = re.compile(r"error|exception|out of memory|traceback", re.IGNORECASE)


class ServerConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    server_args: str = ""
    # Appended after the shared bench_args, so its flags win.
    bench_args: str = ""
    env: Dict[str, str] = {}


class SweepSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    configs: List[ServerConfig]
    launch: str = "sglang serve"
    port: int = 30000
    startup_timeout: float = 1800
    server_args: str = ""
    bench_args: str = ""
    env: Dict[str, str] = {}


class Outcome(msgspec.Struct, frozen=True):
    name: str
    status: str
    detail: str
    startup_s: Optional[float]
    server_cmd: List[str]
    bench_cmd: List[str]


# ---------------------------------------------------------------- commands


def split_args(text: str, where: str) -> List[str]:
    tokens = shlex.split(os.path.expandvars(text))
    unset = [t for t in tokens if "$" in t]
    if unset:
        raise SystemExit(f"{where}: unset environment variable in {unset}")
    return tokens


def launch_prefix(launch: str) -> List[str]:
    tokens = split_args(launch, where="launch")
    # Prefer the `sglang` script of the interpreter running the sweep, not whatever PATH has.
    if tokens[0] == "sglang":
        local = Path(sys.executable).with_name("sglang")
        if local.exists():
            tokens[0] = str(local)
    return tokens


def server_command(spec: SweepSpec, cfg: ServerConfig) -> List[str]:
    return (
        launch_prefix(spec.launch)
        + split_args(spec.server_args, where="server_args")
        + split_args(cfg.server_args, where=f"{cfg.name}.server_args")
        + ["--port", str(spec.port)]
    )


def bench_command(
    spec: SweepSpec,
    cfg: ServerConfig,
    base_url: str,
    out_dir: Path,
    extra_args: List[str],
) -> List[str]:
    return (
        [sys.executable, str(BENCH_SCRIPT), "--base-url", base_url]
        + split_args(spec.bench_args, where="bench_args")
        + split_args(cfg.bench_args, where=f"{cfg.name}.bench_args")
        + extra_args
        + ["--out-dir", str(out_dir)]
    )


# ---------------------------------------------------------------- server lifecycle


def server_answers(base_url: str) -> bool:
    try:
        return requests.get(f"{base_url}/health", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def start_server(cmd: List[str], env: Dict[str, str], log: Path) -> subprocess.Popen:
    with open(log, "w") as f:
        # Own session, so stop_server can signal every scheduler / detokenizer child.
        return subprocess.Popen(
            cmd,
            env={**os.environ, **env},
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def wait_ready(proc: subprocess.Popen, base_url: str, timeout: float) -> Optional[str]:
    """None once the server serves a real generation, otherwise why it never did."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return f"server exited with code {proc.returncode}"
        try:
            r = requests.get(f"{base_url}/health_generate", timeout=60)
            if r.status_code == 200:
                return None
        except requests.RequestException:
            pass
        time.sleep(5)
    return f"not ready after {timeout:.0f} s"


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_server(proc: subprocess.Popen) -> None:
    for sig, grace_s in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        proc.poll()
        if not _group_alive(proc.pid):
            break
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.time() + grace_s
        while time.time() < deadline and _group_alive(proc.pid):
            proc.poll()
            time.sleep(1)
    proc.wait(timeout=30)
    # Give the driver a moment to hand the memory back before the next launch.
    time.sleep(5)


def log_error(log: Path) -> str:
    lines = log.read_text(errors="ignore").splitlines() if log.exists() else []
    hits = [l.strip() for l in lines if ERROR_LINE.search(l)]
    return (hits or lines or ["no output"])[-1][:300]


def run_bench(cmd: List[str], log: Path) -> int:
    with (
        open(log, "w") as f,
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        ) as proc,
    ):
        for line in proc.stdout:
            f.write(line)
            print("  | " + line, end="", flush=True)
    return proc.returncode


def run_config(
    spec: SweepSpec,
    cfg: ServerConfig,
    cfg_dir: Path,
    extra_args: List[str],
) -> Outcome:
    base_url = f"http://127.0.0.1:{spec.port}"
    server_cmd = server_command(spec=spec, cfg=cfg)
    bench_cmd = bench_command(
        spec=spec, cfg=cfg, base_url=base_url, out_dir=cfg_dir, extra_args=extra_args
    )

    def outcome(status: str, detail: str = "", startup_s: Optional[float] = None):
        return Outcome(
            name=cfg.name,
            status=status,
            detail=detail,
            startup_s=startup_s,
            server_cmd=server_cmd,
            bench_cmd=bench_cmd,
        )

    if server_answers(base_url):
        raise SystemExit(f"port {spec.port} already serves something; stop it first")
    server_log = cfg_dir / "server.log"
    print(f"\n=== {cfg.name}: launching\n  $ {shlex.join(server_cmd)}", flush=True)
    t0 = time.time()
    proc = start_server(cmd=server_cmd, env={**spec.env, **cfg.env}, log=server_log)
    try:
        why = wait_ready(proc=proc, base_url=base_url, timeout=spec.startup_timeout)
        if why is not None:
            print(f"  launch failed: {why}: {log_error(server_log)}")
            return outcome("launch failed", f"{why}: {log_error(server_log)}")
        startup_s = time.time() - t0
        print(f"  ready after {startup_s:.0f} s; benchmarking", flush=True)
        code = run_bench(cmd=bench_cmd, log=cfg_dir / "bench.log")
        if proc.poll() is not None:
            return outcome("server crashed", log_error(server_log), startup_s)
        if code != 0:
            return outcome("bench failed", log_error(cfg_dir / "bench.log"), startup_s)
        return outcome("ok", "", startup_s)
    finally:
        stop_server(proc)


# ---------------------------------------------------------------- summary


class ConfigResults(msgspec.Struct, frozen=True):
    outcome: Outcome
    server_info: Dict[str, Any]
    # (context_len, users) -> median burst / bench_serving row
    bursts: Dict[Tuple[int, int], Dict[str, Any]]
    steadies: Dict[Tuple[int, int], Dict[str, Any]]


def load_results(cfg_dir: Path) -> Optional[ConfigResults]:
    status = cfg_dir / "status.json"
    if not status.exists():
        return None
    info_path = cfg_dir / "server_info.json"
    info = (
        json.loads(info_path.read_text())["server_info"] if info_path.exists() else {}
    )

    groups: Dict[Tuple[int, int], List[cb.BurstResult]] = {}
    raw = cfg_dir / "results.jsonl"
    for line in raw.read_text().splitlines() if raw.exists() else []:
        res = msgspec.convert(json.loads(line), type=cb.BurstResult)
        groups.setdefault((res.context_len, res.users), []).append(res)

    steadies: Dict[Tuple[int, int], Dict[str, Any]] = {}
    serving = cfg_dir / "bench_serving.jsonl"
    for line in serving.read_text().splitlines() if serving.exists() else []:
        row = json.loads(line)
        m = re.fullmatch(r"L(\d+)_N(\d+)", row.get("tag") or "")
        if m:
            steadies[(int(m[1]), int(m[2]))] = row
    return ConfigResults(
        outcome=msgspec.json.decode(status.read_bytes(), type=Outcome),
        server_info=info,
        bursts={k: cb.median_burst(v) for k, v in groups.items()},
        steadies=steadies,
    )


def max_running(si: Dict[str, Any]) -> Optional[int]:
    per_dp = [
        s["effective_max_running_requests_per_dp"]
        for s in si.get("internal_states") or []
        if s.get("effective_max_running_requests_per_dp")
    ]
    return min(per_dp) if per_dp else si.get("max_running_requests")


def _summary_configs(spec: SweepSpec, results: Dict[str, ConfigResults]) -> List[str]:
    lines = [
        "## Configurations",
        "",
        "| Config | Status | KV pool tokens | Max running | Context | Speculative "
        "| Startup (s) | Server args (on top of the shared ones) |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for cfg in spec.configs:
        args = " ".join(cfg.server_args.split())
        r = results.get(cfg.name)
        if r is None:
            lines.append(f"| {cfg.name} | not run | | | | | | `{args}` |")
            continue
        si, o = r.server_info, r.outcome
        status = o.status if o.status == "ok" else f"**{o.status}**: {o.detail}"
        kv = si.get("max_total_num_tokens")
        lines.append(
            f"| [{cfg.name}]({cfg.name}/report.md) | {status} "
            f"| {f'{kv:,}' if kv else '-'} | {max_running(si) or '-'} "
            f"| {si.get('context_length') or '-'} | {si.get('speculative_algorithm') or '-'} "
            f"| {cb.fmt(o.startup_s, '.0f')} | `{args}` |"
        )
    return lines


def _summary_burst(results: Dict[str, ConfigResults]) -> List[str]:
    cells = sorted({k for r in results.values() for k in r.bursts})
    if not cells:
        return []
    lines = [
        "",
        "## Burst: N users arrive together",
        "",
        "Medians over repeats. **Bold** is the highest total decode among the configs where "
        "all N users ran together. See capacity_bench's report.md for metric definitions.",
        "",
        "| Context | Users | Config | All concurrent | TTFT 1st / last (s) | Prefill tok/s, all "
        "| Decode tok/s per user | Decode tok/s total | Decode tok/s while others prefill "
        "| Accept len |",
        "|---:|---:|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        rows = [(n, r.bursts[cell]) for n, r in results.items() if cell in r.bursts]
        best = max(
            (b["decode_tps_total"] or 0 for _, b in rows if b["all_concurrent"]),
            default=None,
        )
        for name, b in rows:
            total = cb.fmt(b["decode_tps_total"])
            if best and b["all_concurrent"] and b["decode_tps_total"] == best:
                total = f"**{total}**"
            lines.append(
                f"| {cell[0]:,} | {cell[1]} | {name} "
                f"| {'yes' if b['all_concurrent'] else '**no**'} "
                f"| {cb.fmt(b['ttft_first'], '.2f')} / {cb.fmt(b['ttft_last'], '.2f')} "
                f"| {cb.fmt(b['prefill_tps_all'], ',.0f')} | {cb.fmt(b['decode_tps_per_user'])} "
                f"| {total} | {cb.fmt(b['decode_tps_while_others_prefill'])} "
                f"| {cb.fmt(b['accept_length'], '.2f')} |"
            )
    return lines


def _steady_decode(row: Dict[str, Any]) -> Optional[float]:
    tpot = row.get("median_tpot_ms")
    return 1e3 / tpot if tpot else None


def _summary_steady(results: Dict[str, ConfigResults]) -> List[str]:
    cells = sorted({k for r in results.values() for k in r.steadies})
    if not cells:
        return []
    lines = [
        "",
        "## Steady state: `sglang.bench_serving`, concurrency N",
        "",
        "| Context | Users | Config | Median TTFT (s) | P99 TTFT (s) | Median TPOT (ms) "
        "| P99 ITL (ms) | Decode tok/s per user | Output tok/s |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        for name, r in results.items():
            s = r.steadies.get(cell)
            if s is None:
                continue
            lines.append(
                f"| {cell[0]:,} | {cell[1]} | {name} | {cb.fmt(s['median_ttft_ms'] / 1e3, '.2f')} "
                f"| {cb.fmt(s['p99_ttft_ms'] / 1e3, '.2f')} | {cb.fmt(s.get('median_tpot_ms'), '.2f')} "
                f"| {cb.fmt(s.get('p99_itl_ms'), '.0f')} | {cb.fmt(_steady_decode(s))} | {cb.fmt(s.get('output_throughput'))} |"
            )
    return lines


def _users_within_slo(
    r: ConfigResults, context_len: int, slo_decode: float, slo_ttft: Optional[float]
) -> Tuple[Optional[int], Optional[int]]:
    burst = [
        n
        for (L, n), b in r.bursts.items()
        if L == context_len
        and b["all_concurrent"]
        and (b["decode_tps_per_user"] or 0) >= slo_decode
        and (slo_ttft is None or b["ttft_last"] <= slo_ttft)
    ]
    steady = [
        n
        for (L, n), s in r.steadies.items()
        if L == context_len
        and (_steady_decode(s) or 0) >= slo_decode
        and (slo_ttft is None or s["median_ttft_ms"] / 1e3 <= slo_ttft)
    ]
    has_burst = any(L == context_len for L, _ in r.bursts)
    has_steady = any(L == context_len for L, _ in r.steadies)
    return (
        max(burst, default=0) if has_burst else None,
        max(steady, default=0) if has_steady else None,
    )


def _summary_slo(
    results: Dict[str, ConfigResults], slo_decode: float, slo_ttft: Optional[float]
) -> List[str]:
    lines = [
        "",
        "## Users within the SLO",
        "",
        f"Per-user decode >= {slo_decode} tok/s and TTFT <= {slo_ttft or 'inf'} s. Burst "
        "uses the last user's TTFT; steady uses the median TTFT and 1000 / median TPOT. "
        "A user count that was not run cannot count, so this is a lower bound.",
        "",
        "| Config | Context | Users (burst) | Users (steady) |",
        "|---|---:|---:|---:|",
    ]
    for name, r in results.items():
        for L in sorted({L for L, _ in [*r.bursts, *r.steadies]}):
            burst, steady = _users_within_slo(
                r=r, context_len=L, slo_decode=slo_decode, slo_ttft=slo_ttft
            )
            lines.append(
                f"| {name} | {L:,} | {cb.fmt(burst, 'd')} | {cb.fmt(steady, 'd')} |"
            )
    return lines


def write_summary(
    out_dir: Path, spec_path: Path, spec: SweepSpec, args: argparse.Namespace
) -> Path:
    results = {
        cfg.name: r
        for cfg in spec.configs
        if (r := load_results(out_dir / cfg.name)) is not None
    }
    gpus = ", ".join(f"{g['name']} ({g['memory.total']})" for g in cb.local_gpus())
    lines = [
        "# SGLang server configuration sweep",
        "",
        f"- **Date:** {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"- **Sweep file:** `{spec_path}` (copied to `sweep.toml`)",
        f"- **GPUs on this host:** {gpus or 'unknown'}",
        f"- **Shared server args:** `{' '.join(spec.server_args.split())}`",
        f"- **Shared bench args:** `{' '.join(spec.bench_args.split())}`",
        "",
    ]
    lines += _summary_configs(spec=spec, results=results)
    ok = {n: r for n, r in results.items() if r.bursts or r.steadies}
    lines += _summary_burst(ok)
    lines += _summary_steady(ok)
    if ok and (args.slo_decode or args.slo_ttft):
        lines += _summary_slo(
            results=ok, slo_decode=args.slo_decode or 0, slo_ttft=args.slo_ttft
        )
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("spec", type=Path, help="Sweep TOML file.")
    p.add_argument(
        "--only", default=None, help="Comma list of config names to run (default all)."
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: benchmark/capacity/results/sweep-<timestamp>.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs that already finished with status ok in --out-dir.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print the commands and exit."
    )
    p.add_argument(
        "--bench-args",
        default="",
        help="Extra capacity_bench.py flags for every config, appended last, "
        "e.g. '--steady --context-lens 32k'.",
    )
    p.add_argument("--slo-decode", type=float, default=None)
    p.add_argument("--slo-ttft", type=float, default=None)
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild summary.md from --out-dir without running anything.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = msgspec.toml.decode(args.spec.read_bytes(), type=SweepSpec)
    names = [c.name for c in spec.configs]
    if len(set(names)) != len(names):
        raise SystemExit(f"config names must be unique: {names}")
    selected = (
        [c for c in spec.configs if c.name in args.only.split(",")]
        if args.only
        else spec.configs
    )
    for cfg in selected:
        # Fail on an unset $VAR before any server starts, not halfway through the sweep.
        server_command(spec=spec, cfg=cfg)
    extra_args = shlex.split(args.bench_args)
    if args.slo_decode:
        extra_args += ["--slo-decode", str(args.slo_decode)]
    if args.slo_ttft:
        extra_args += ["--slo-ttft", str(args.slo_ttft)]

    if args.dry_run:
        for cfg in selected:
            print(f"# {cfg.name}\n{shlex.join(server_command(spec=spec, cfg=cfg))}")
            cmd = bench_command(
                spec=spec,
                cfg=cfg,
                base_url=f"http://127.0.0.1:{spec.port}",
                out_dir=Path("<out>") / cfg.name,
                extra_args=extra_args,
            )
            print(shlex.join(cmd) + "\n")
        return

    if (args.resume or args.summary_only) and args.out_dir is None:
        raise SystemExit("--resume and --summary-only need --out-dir")
    out_dir = args.out_dir or (
        Path(__file__).parent
        / "results"
        / datetime.datetime.now().strftime("sweep-%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        (out_dir / "sweep.toml").write_bytes(args.spec.read_bytes())
        for cfg in selected:
            cfg_dir = out_dir / cfg.name
            done = load_results(cfg_dir)
            if args.resume and done is not None and done.outcome.status == "ok":
                print(f"=== {cfg.name}: already done, skipping")
                continue
            # A rerun replaces this config's earlier results instead of appending to them.
            for stale in ("results.jsonl", "bench_serving.jsonl", "status.json"):
                (cfg_dir / stale).unlink(missing_ok=True)
            cfg_dir.mkdir(parents=True, exist_ok=True)
            outcome = run_config(
                spec=spec, cfg=cfg, cfg_dir=cfg_dir, extra_args=extra_args
            )
            (cfg_dir / "status.json").write_bytes(msgspec.json.encode(outcome))
            print(f"=== {cfg.name}: {outcome.status} {outcome.detail}", flush=True)
            write_summary(out_dir=out_dir, spec_path=args.spec, spec=spec, args=args)
    summary = write_summary(out_dir=out_dir, spec_path=args.spec, spec=spec, args=args)
    print(f"\nsummary: {summary}\n")
    print(summary.read_text())


if __name__ == "__main__":
    main()
