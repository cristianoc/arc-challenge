#!/usr/bin/env python3
"""Measure the sole solver child, excluding Cargo build resource usage (Linux)."""
import pathlib, resource, subprocess, sys
code = subprocess.call(sys.argv[2:])
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
pathlib.Path(sys.argv[1]).write_text(f"peak_rss_kib={rss}\n")
sys.exit(code)
