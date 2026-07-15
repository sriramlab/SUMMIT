#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description="Benchmark a command and its process-tree RSS.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    for path in (args.output_json, args.stdout, args.stderr):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    peak_rss = 0
    peak_pss = 0
    peak_processes = 0
    cpu_user = 0.0
    cpu_system = 0.0
    read_bytes = 0
    write_bytes = 0
    with open(args.stdout, "w") as stdout, open(args.stderr, "w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=os.environ.copy())
        root = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                processes = [root, *root.children(recursive=True)]
                rss = 0
                pss = 0
                sample_user = 0.0
                sample_system = 0.0
                sample_read = 0
                sample_write = 0
                sampled = 0
                for observed in processes:
                    try:
                        memory = observed.memory_full_info()
                        times = observed.cpu_times()
                        io = observed.io_counters()
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        continue
                    rss += int(memory.rss)
                    pss += int(getattr(memory, "pss", memory.rss))
                    sample_user += float(times.user)
                    sample_system += float(times.system)
                    sample_read += int(io.read_bytes)
                    sample_write += int(io.write_bytes)
                    sampled += 1
                peak_rss = max(peak_rss, int(rss))
                peak_pss = max(peak_pss, int(pss))
                peak_processes = max(peak_processes, sampled)
                cpu_user = max(cpu_user, sample_user)
                cpu_system = max(cpu_system, sample_system)
                read_bytes = max(read_bytes, sample_read)
                write_bytes = max(write_bytes, sample_write)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
            time.sleep(0.2)
        returncode = process.wait()

    payload = {
        "command": command,
        "returncode": int(returncode),
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_rss_bytes": int(peak_rss),
        "peak_pss_bytes": int(peak_pss),
        "peak_processes": int(peak_processes),
        "cpu_user_seconds": float(cpu_user),
        "cpu_system_seconds": float(cpu_system),
        "io_read_bytes": int(read_bytes),
        "io_write_bytes": int(write_bytes),
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
