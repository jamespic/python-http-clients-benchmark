# HTTP Client Benchmark

## What's this?

This project benchmarks different Python HTTP clients—mostly async ones (like `httpx`, `aiohttp`, `pyreqwest`, etc.), plus some sync clients for comparison. It also checks out HTTPX with different backends. The main goal is to see which client is fastest under real workloads, and whether async really beats sync. We're also curious if Python's new no-GIL interpreter (Python 3.14) changes anything.

## What does it do?

- Runs benchmarks using Python 3.14 (no-GIL), but you don't need it preinstalled—just use `uv`!
- Tests a bunch of async and sync HTTP clients.
- Benchmarks against several endpoints and server types (HTTP/1.1, HTTPS/1.1, HTTPS/2).
- Lets you tweak request rates, durations, and more.
- Generates HTML reports and SVG graphs for throughput, latency, and failures.

## Why?

- **Main goal:** Find out which async Python HTTP client is fastest, and how they handle different workloads.
- **Bonus goal:** See if async setups really outperform sync ones, and whether Python 3.14's no-GIL interpreter makes a difference.

## How do I use it?

1. **Run the benchmarks:**

   ```bash
   uv run run_suite.py --duration 600 --initial-rate 0 --final-rate 600
   ```

   (Change the numbers if you want.)

2. **Generate the report:**

   ```bash
   uv run analyse_results.py
   ```

   You'll get a `report.html` and SVG graphs in the `graphs/` folder.

## What do I need?

- [uv](https://github.com/astral-sh/uv) (handles Python and dependencies for you)
- A Linux machine (that's what it's tested on)

## What do I get?

- A clear comparison of async vs sync HTTP clients in Python 3.14.
- Insight into whether the no-GIL interpreter actually helps.
- Pretty graphs and a report you can show off.

---

_Check out the code and the report for all the details!_
