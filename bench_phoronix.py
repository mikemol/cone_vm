import argparse
import csv
import json
import math
import os
import platform as platform_mod
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_csv_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in _parse_csv_list(raw):
        if not item.isdigit():
            continue
        value = int(item)
        if value > 0:
            values.append(value)
    return values


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_cmd(cmd: List[str]) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output if output else None


def _read_cpuinfo() -> Dict[str, str]:
    info: Dict[str, str] = {}
    cpuinfo_path = "/proc/cpuinfo"
    if not os.path.exists(cpuinfo_path):
        return info
    model = None
    cores = None
    with open(cpuinfo_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("model name") and model is None:
                model = line.split(":", 1)[1].strip()
            if line.startswith("cpu cores") and cores is None:
                cores = line.split(":", 1)[1].strip()
    if model:
        info["model"] = model
    if cores:
        info["cores_per_socket"] = cores
    return info


def _read_meminfo_gb() -> Optional[float]:
    meminfo_path = "/proc/meminfo"
    if not os.path.exists(meminfo_path):
        return None
    with open(meminfo_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    kb = int(parts[1])
                    return kb / (1024.0 * 1024.0)
    return None


def _collect_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["hostname"] = platform_mod.node()
    info["os"] = platform_mod.platform()
    info["python_version"] = sys.version.split()[0]
    info["cpu_count"] = os.cpu_count()
    cpuinfo = _read_cpuinfo()
    if cpuinfo:
        info["cpu_model"] = cpuinfo.get("model", "")
        info["cpu_cores_per_socket"] = cpuinfo.get("cores_per_socket", "")
    mem_gb = _read_meminfo_gb()
    if mem_gb is not None:
        info["memory_gb"] = round(mem_gb, 2)

    nvidia_info = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if nvidia_info:
        info["nvidia_smi"] = nvidia_info
    return info


def _introspect_jax(platform: str) -> Dict[str, Any]:
    env = os.environ.copy()
    env["JAX_PLATFORM_NAME"] = platform
    code = (
        "import json\n"
        "import jax\n"
        "import jaxlib\n"
        "devices = []\n"
        "for d in jax.devices():\n"
        "    devices.append({\n"
        "        'platform': d.platform,\n"
        "        'device_kind': getattr(d, 'device_kind', ''),\n"
        "        'id': getattr(d, 'id', ''),\n"
        "    })\n"
        "info = {\n"
        "    'jax_version': jax.__version__,\n"
        "    'jaxlib_version': getattr(jaxlib, '__version__', ''),\n"
        "    'default_backend': jax.default_backend(),\n"
        "    'devices': devices,\n"
        "}\n"
        "print(json.dumps(info))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip() or "jax introspection failed"}
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"error": "jax introspection parse failed"}


def _platform_available(info: Dict[str, Any], requested: str) -> bool:
    if "devices" not in info:
        return False
    for device in info["devices"]:
        if device.get("platform") == requested:
            return True
    return False


def _run_bench_compare(
    platform: str,
    out_csv: str,
    args: argparse.Namespace,
) -> bool:
    env = os.environ.copy()
    env["JAX_PLATFORM_NAME"] = platform
    cmd = [
        sys.executable,
        "bench_compare.py",
        "--out",
        out_csv,
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--cycles",
        str(args.cycles),
        "--block-sizes",
        args.block_sizes,
        "--arena-counts",
        args.arena_counts,
        "--swizzle-backends",
        args.swizzle_backends,
        "--hierarchy-l1-mult",
        str(args.hierarchy_l1_mult),
        "--metrics-block-size",
        str(args.metrics_block_size),
    ]
    if args.workloads:
        cmd.extend(["--workloads", args.workloads])
    if args.modes:
        cmd.extend(["--modes", args.modes])
    if args.no_baseline:
        cmd.append("--no-baseline")
    if args.hierarchy_no_global:
        cmd.append("--hierarchy-no-global")
    if args.hierarchy_morton:
        cmd.append("--hierarchy-morton")
    if args.hierarchy_workload_l2:
        cmd.extend(["--hierarchy-workload-l2", str(args.hierarchy_workload_l2)])
    if args.hierarchy_workload_l1:
        cmd.extend(["--hierarchy-workload-l1", str(args.hierarchy_workload_l1)])

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(cmd, env=env, cwd=repo_dir)
    return result.returncode == 0


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    k = (len(values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values[int(k)])
    d = values[c] - values[f]
    return float(values[f] + d * (k - f))


def _compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "stdev": 0.0,
        }
    vals = sorted(values)
    mean = sum(vals) / len(vals)
    stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": float(mean),
        "median": float(statistics.median(vals)),
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "p95": _percentile(vals, 95.0),
        "p99": _percentile(vals, 99.0),
        "stdev": float(stdev),
    }


def _group_stats(
    rows: List[Dict[str, str]],
    keys: List[str],
    value_key: str,
    metric: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], List[float]] = {}
    for row in rows:
        value = _to_float(row.get(value_key))
        if value is None:
            continue
        key = tuple(row.get(k, "") for k in keys)
        grouped.setdefault(key, []).append(value)
    out: List[Dict[str, Any]] = []
    for key, values in grouped.items():
        stats = _compute_stats(values)
        row = {k: key[idx] for idx, k in enumerate(keys)}
        row["metric"] = metric
        row.update({f"{value_key}_{k}": v for k, v in stats.items()})
        out.append(row)
    return out


def _find_inflections(counts: List[int], values: List[float]) -> List[int]:
    if len(counts) < 3:
        return []
    slopes = []
    for i in range(len(counts) - 1):
        dx = counts[i + 1] - counts[i]
        if dx <= 0:
            continue
        slopes.append((values[i + 1] - values[i]) / float(dx))
    if not slopes:
        return []
    median_slope = statistics.median(slopes)
    if median_slope <= 0:
        return []
    inflections = []
    for i, slope in enumerate(slopes):
        if slope > median_slope * 1.5:
            inflections.append(counts[i + 1])
    return inflections


class SimpleCanvas:
    def __init__(self, width: int, height: int, bg: Tuple[int, int, int]):
        self.width = width
        self.height = height
        self.buf = bytearray(width * height * 3)
        r, g, b = bg
        for i in range(0, len(self.buf), 3):
            self.buf[i] = r
            self.buf[i + 1] = g
            self.buf[i + 2] = b

    def _set_pixel(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        idx = (y * self.width + x) * 3
        self.buf[idx] = color[0]
        self.buf[idx + 1] = color[1]
        self.buf[idx + 2] = color[2]

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self._set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def rect(self, x: int, y: int, w: int, h: int, color: Tuple[int, int, int]) -> None:
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self._set_pixel(xx, yy, color)

    def draw_text(
        self,
        x: int,
        y: int,
        text: str,
        color: Tuple[int, int, int],
        scale: int = 1,
    ) -> None:
        for char in text:
            glyph = FONT_5X7.get(char.upper(), FONT_5X7["?"])
            for row, bits in enumerate(glyph):
                for col in range(5):
                    if bits & (1 << (4 - col)):
                        for sy in range(scale):
                            for sx in range(scale):
                                self._set_pixel(
                                    x + col * scale + sx,
                                    y + row * scale + sy,
                                    color,
                                )
            x += (5 + 1) * scale

    def write_png(self, path: str) -> None:
        import struct
        import zlib

        raw = bytearray()
        row_bytes = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * row_bytes
            raw.extend(self.buf[start : start + row_bytes])
        compressed = zlib.compress(bytes(raw), 9)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            ihdr = struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
            f.write(chunk(b"IHDR", ihdr))
            f.write(chunk(b"IDAT", compressed))
            f.write(chunk(b"IEND", b""))


FONT_5X7 = {
    "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "B": [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    "C": [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
    "D": [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
    "E": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    "F": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    "G": [0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
    "H": [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "I": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111],
    "J": [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
    "K": [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    "L": [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    "M": [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
    "N": [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
    "O": [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "P": [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    "Q": [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    "R": [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    "S": [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
    "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "U": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "V": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    "W": [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
    "X": [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
    "Y": [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
    "Z": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    "3": [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
    "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    "5": [0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110],
    "6": [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
    "-": [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
    "_": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
    ".": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110],
    "/": [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000],
    ":": [0b00000, 0b00110, 0b00110, 0b00000, 0b00110, 0b00110, 0b00000],
    " ": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    "?": [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100],
}


class Plotter:
    def __init__(self) -> None:
        self.backend = "simple"
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F401

            self.backend = "matplotlib"
        except Exception:
            self.backend = "simple"

    def line_chart(
        self,
        path_base: str,
        title: str,
        x_values: List[int],
        series: List[Tuple[str, List[float], str]],
        x_label: str,
        y_label: str,
    ) -> None:
        if self.backend == "matplotlib":
            self._line_chart_matplotlib(
                path_base, title, x_values, series, x_label, y_label
            )
        else:
            self._line_chart_simple(
                path_base, title, x_values, series, x_label, y_label
            )

    def bar_chart(
        self,
        path_base: str,
        title: str,
        categories: List[str],
        series: List[Tuple[str, List[float], str]],
        y_label: str,
    ) -> None:
        if self.backend == "matplotlib":
            self._bar_chart_matplotlib(path_base, title, categories, series, y_label)
        else:
            self._bar_chart_simple(path_base, title, categories, series, y_label)

    def _line_chart_matplotlib(
        self,
        path_base: str,
        title: str,
        x_values: List[int],
        series: List[Tuple[str, List[float], str]],
        x_label: str,
        y_label: str,
    ) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        for label, y_values, color in series:
            plt.plot(x_values, y_values, label=label, color=color)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(path_base + ".svg")
        plt.savefig(path_base + ".png", dpi=150)
        plt.close()

    def _bar_chart_matplotlib(
        self,
        path_base: str,
        title: str,
        categories: List[str],
        series: List[Tuple[str, List[float], str]],
        y_label: str,
    ) -> None:
        import matplotlib.pyplot as plt

        width = 0.8 / max(1, len(series))
        x_positions = list(range(len(categories)))
        plt.figure(figsize=(12, 5))
        for idx, (label, values, color) in enumerate(series):
            offsets = [x + idx * width for x in x_positions]
            plt.bar(offsets, values, width=width, label=label, color=color)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xticks(
            [x + width * (len(series) - 1) / 2 for x in x_positions],
            categories,
            rotation=45,
            ha="right",
            fontsize=8,
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(path_base + ".svg")
        plt.savefig(path_base + ".png", dpi=150)
        plt.close()

    def _line_chart_simple(
        self,
        path_base: str,
        title: str,
        x_values: List[int],
        series: List[Tuple[str, List[float], str]],
        x_label: str,
        y_label: str,
    ) -> None:
        width = 900
        height = 500
        margin = 60
        canvas = SimpleCanvas(width, height, (255, 255, 255))
        min_x = min(x_values) if x_values else 0
        max_x = max(x_values) if x_values else 1
        min_y = min(min(vals) for _, vals, _ in series) if series else 0.0
        max_y = max(max(vals) for _, vals, _ in series) if series else 1.0
        if min_y == max_y:
            max_y += 1.0

        def sx(x: int) -> int:
            if max_x == min_x:
                return margin
            return int(
                margin + (x - min_x) / float(max_x - min_x) * (width - 2 * margin)
            )

        def sy(y: float) -> int:
            return int(
                height
                - margin
                - (y - min_y) / float(max_y - min_y) * (height - 2 * margin)
            )

        axis_color = (30, 30, 30)
        canvas.line(margin, height - margin, width - margin, height - margin, axis_color)
        canvas.line(margin, margin, margin, height - margin, axis_color)

        for i in range(5):
            x = min_x + (max_x - min_x) * i // 4 if max_x != min_x else min_x
            px = sx(x)
            canvas.line(px, height - margin, px, height - margin + 5, axis_color)
            canvas.draw_text(px - 10, height - margin + 10, str(x), axis_color, 1)

        for i in range(5):
            y = min_y + (max_y - min_y) * i / 4.0
            py = sy(y)
            canvas.line(margin - 5, py, margin, py, axis_color)
            canvas.draw_text(5, py - 4, f"{y:.1f}", axis_color, 1)

        palette = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]
        for idx, (label, y_values, _) in enumerate(series):
            color = palette[idx % len(palette)]
            for i in range(len(x_values) - 1):
                canvas.line(
                    sx(x_values[i]),
                    sy(y_values[i]),
                    sx(x_values[i + 1]),
                    sy(y_values[i + 1]),
                    color,
                )
            canvas.draw_text(
                width - margin + 5, margin + idx * 12, label[:12], color, 1
            )

        canvas.draw_text(margin, 10, title.upper()[:40], axis_color, 1)
        canvas.draw_text(width // 2 - 40, height - 20, x_label.upper(), axis_color, 1)
        canvas.draw_text(5, 20, y_label.upper(), axis_color, 1)

        self._write_simple_svg(
            path_base + ".svg",
            width,
            height,
            title,
            x_values,
            series,
            x_label,
            y_label,
        )
        canvas.write_png(path_base + ".png")

    def _bar_chart_simple(
        self,
        path_base: str,
        title: str,
        categories: List[str],
        series: List[Tuple[str, List[float], str]],
        y_label: str,
    ) -> None:
        width = 1000
        height = 500
        margin = 60
        canvas = SimpleCanvas(width, height, (255, 255, 255))
        max_y = 0.0
        for _, values, _ in series:
            if values:
                max_y = max(max_y, max(values))
        if max_y <= 0:
            max_y = 1.0

        def sy(y: float) -> int:
            return int(
                height - margin - y / float(max_y) * (height - 2 * margin)
            )

        axis_color = (30, 30, 30)
        canvas.line(margin, height - margin, width - margin, height - margin, axis_color)
        canvas.line(margin, margin, margin, height - margin, axis_color)

        palette = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]
        group_width = (width - 2 * margin) / max(1, len(categories))
        bar_width = group_width / max(1, len(series) + 1)

        for c_idx, category in enumerate(categories):
            base_x = margin + c_idx * group_width
            for s_idx, (_, values, _) in enumerate(series):
                if c_idx >= len(values):
                    continue
                color = palette[s_idx % len(palette)]
                x0 = int(base_x + s_idx * bar_width)
                y0 = sy(values[c_idx])
                canvas.rect(
                    x0,
                    y0,
                    max(1, int(bar_width * 0.8)),
                    height - margin - y0,
                    color,
                )
            canvas.draw_text(int(base_x), height - margin + 8, category[:10], axis_color, 1)

        for s_idx, (label, _, _) in enumerate(series):
            color = palette[s_idx % len(palette)]
            canvas.draw_text(width - margin + 5, margin + s_idx * 12, label[:12], color, 1)

        canvas.draw_text(margin, 10, title.upper()[:40], axis_color, 1)
        canvas.draw_text(5, 20, y_label.upper(), axis_color, 1)

        self._write_simple_svg(
            path_base + ".svg",
            width,
            height,
            title,
            list(range(len(categories))),
            series,
            "",
            y_label,
            categories=categories,
        )
        canvas.write_png(path_base + ".png")

    def _write_simple_svg(
        self,
        path: str,
        width: int,
        height: int,
        title: str,
        x_values: List[int],
        series: List[Tuple[str, List[float], str]],
        x_label: str,
        y_label: str,
        categories: Optional[List[str]] = None,
    ) -> None:
        margin = 60
        min_x = min(x_values) if x_values else 0
        max_x = max(x_values) if x_values else 1
        min_y = min(min(vals) for _, vals, _ in series) if series else 0.0
        max_y = max(max(vals) for _, vals, _ in series) if series else 1.0
        if min_y == max_y:
            max_y += 1.0

        def sx(x: float) -> float:
            if max_x == min_x:
                return margin
            return margin + (x - min_x) / float(max_x - min_x) * (width - 2 * margin)

        def sy(y: float) -> float:
            return height - margin - (y - min_y) / float(max_y - min_y) * (height - 2 * margin)

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />',
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" />',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" />',
            f'<text x="{margin}" y="20" font-size="12" fill="#333">{title}</text>',
            f'<text x="{width // 2}" y="{height - 10}" font-size="12" fill="#333">{x_label}</text>',
            f'<text x="10" y="30" font-size="12" fill="#333">{y_label}</text>',
        ]
        for i, (label, y_values, _) in enumerate(series):
            color = palette[i % len(palette)]
            if categories:
                group_width = (width - 2 * margin) / max(1, len(categories))
                bar_width = group_width / max(1, len(series) + 1)
                for c_idx, value in enumerate(y_values):
                    x0 = margin + c_idx * group_width + i * bar_width
                    y0 = sy(value)
                    h = height - margin - y0
                    lines.append(
                        f'<rect x="{x0}" y="{y0}" width="{bar_width * 0.8}" height="{h}" fill="{color}" />'
                    )
            else:
                points = []
                for idx, y in enumerate(y_values):
                    points.append(f"{sx(x_values[idx])},{sy(y)}")
                lines.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}" />'
                )
            lines.append(
                f'<text x="{width - margin + 5}" y="{margin + i * 14}" font-size="10" fill="{color}">{label}</text>'
            )
        lines.append("</svg>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def _write_report(
    path: str,
    system_info: Dict[str, Any],
    platform_info: Dict[str, Any],
    args: argparse.Namespace,
    summary_rows: List[Dict[str, Any]],
    inflections: List[Dict[str, Any]],
    plot_files: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append("## System")
    for key, value in system_info.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Platforms")
    for platform, info in platform_info.items():
        lines.append(f"- {platform}: {json.dumps(info, sort_keys=True)}")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- runs: {args.runs}")
    lines.append(f"- warmup: {args.warmup}")
    lines.append(f"- cycles: {args.cycles}")
    lines.append(f"- block_sizes: {args.block_sizes}")
    lines.append(f"- arena_counts: {args.arena_counts or 'default'}")
    lines.append(f"- swizzle_backends: {args.swizzle_backends}")
    lines.append(f"- modes: {args.modes or 'default'}")
    lines.append(f"- workloads: {args.workloads or 'default'}")
    lines.append("")
    lines.append("## Summary (exec_ms mean)")
    lines.append("| platform | backend | workload | mode | exec_ms_mean | exec_ms_p95 | n |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in summary_rows[:30]:
        lines.append(
            "| {platform} | {backend} | {workload} | {mode} | {mean:.2f} | {p95:.2f} | {n} |".format(
                platform=row.get("platform", ""),
                backend=row.get("swizzle_backend_effective", ""),
                workload=row.get("workload", ""),
                mode=row.get("mode", ""),
                mean=row.get("exec_ms_mean", 0.0),
                p95=row.get("exec_ms_p95", 0.0),
                n=row.get("exec_ms_n", 0),
            )
        )
    if len(summary_rows) > 30:
        lines.append("")
        lines.append(f"Truncated summary table to 30 rows (total {len(summary_rows)}).")
    lines.append("")
    lines.append("## Inflection Points")
    if not inflections:
        lines.append("No inflection points detected.")
    else:
        lines.append("| platform | backend | arena_kind | mode | counts |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in inflections:
            lines.append(
                "| {platform} | {backend} | {arena_kind} | {mode} | {counts} |".format(
                    platform=row.get("platform", ""),
                    backend=row.get("swizzle_backend_effective", ""),
                    arena_kind=row.get("arena_kind", ""),
                    mode=row.get("mode", ""),
                    counts=",".join(str(c) for c in row.get("counts", [])),
                )
            )
    lines.append("")
    lines.append("## Plots")
    for plot in plot_files:
        lines.append(f"- {plot}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _select_backend(rows: List[Dict[str, str]], platform: str) -> Optional[str]:
    counts: Dict[str, int] = {}
    for row in rows:
        if row.get("platform") != platform:
            continue
        backend = row.get("swizzle_backend_effective", "")
        counts[backend] = counts.get(backend, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


def _select_modes(rows: List[Dict[str, str]], max_modes: int) -> List[str]:
    modes = sorted({row.get("mode", "") for row in rows if row.get("mode")})
    if len(modes) <= max_modes:
        return modes
    def score(name: str) -> Tuple[int, str]:
        if "hier-global" in name:
            return (0, name)
        if name.startswith("bsp-hier"):
            return (1, name)
        if name.startswith("bsp-block"):
            return (2, name)
        if name == "bsp-rank":
            return (3, name)
        if name.startswith("bsp-morton"):
            return (4, name)
        return (10, name)
    modes_sorted = sorted(modes, key=score)
    return modes_sorted[:max_modes]


def _plot_arena_scales(
    rows: List[Dict[str, str]],
    out_dir: str,
    platform: str,
    backend: str,
    plotter: Plotter,
) -> List[str]:
    plots: List[str] = []
    filtered = [
        row
        for row in rows
        if row.get("platform") == platform
        and row.get("swizzle_backend_effective") == backend
        and row.get("workload_arena_kind")
    ]
    if not filtered:
        return plots
    arena_kinds = sorted({row.get("workload_arena_kind") for row in filtered})
    modes = _select_modes(filtered, 6)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for kind in arena_kinds:
        rows_kind = [row for row in filtered if row.get("workload_arena_kind") == kind]
        counts = sorted(
            {int(row.get("workload_arena_count", "0")) for row in rows_kind if row.get("workload_arena_count")}
        )
        if not counts:
            continue
        series: List[Tuple[str, List[float], str]] = []
        for idx, mode in enumerate(modes):
            values = []
            for count in counts:
                samples = [
                    _to_float(row.get("ms_per_cycle"))
                    for row in rows_kind
                    if row.get("mode") == mode and int(row.get("workload_arena_count", "0")) == count
                ]
                samples = [v for v in samples if v is not None]
                if not samples:
                    values.append(0.0)
                else:
                    values.append(sum(samples) / len(samples))
            series.append((mode, values, palette[idx % len(palette)]))
        plot_base = os.path.join(
            out_dir,
            f"arena_scale_{platform}_{backend}_{kind}",
        )
        plotter.line_chart(
            plot_base,
            f"{platform} {backend} {kind} scale",
            counts,
            series,
            "arena_count",
            "ms_per_cycle",
        )
        plots.extend([plot_base + ".svg", plot_base + ".png"])
    return plots


def _plot_hierarchy_modes(
    rows: List[Dict[str, str]],
    out_dir: str,
    platform: str,
    backend: str,
    plotter: Plotter,
) -> List[str]:
    plots: List[str] = []
    filtered = [
        row
        for row in rows
        if row.get("platform") == platform
        and row.get("swizzle_backend_effective") == backend
        and row.get("workload", "").startswith("arena_hierarchy")
    ]
    if not filtered:
        return plots
    workloads = sorted({row.get("workload", "") for row in filtered if row.get("workload")})
    for workload in workloads:
        rows_workload = [row for row in filtered if row.get("workload") == workload]
        modes = sorted({row.get("mode", "") for row in rows_workload if row.get("mode")})
        values = []
        for mode in modes:
            samples = [
                _to_float(row.get("ms_per_cycle"))
                for row in rows_workload
                if row.get("mode") == mode
            ]
            samples = [v for v in samples if v is not None]
            values.append(sum(samples) / len(samples) if samples else 0.0)
        if not modes:
            continue
        plot_base = os.path.join(out_dir, f"hierarchy_modes_{platform}_{backend}_{workload}")
        plotter.bar_chart(
            plot_base,
            f"{platform} {backend} {workload}",
            modes,
            [("ms_per_cycle", values, "#1f77b4")],
            "ms_per_cycle",
        )
        plots.extend([plot_base + ".svg", plot_base + ".png"])
    return plots


def _compute_inflections(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for platform in sorted({row.get("platform") for row in rows if row.get("platform")}):
        for backend in sorted({row.get("swizzle_backend_effective") for row in rows if row.get("swizzle_backend_effective")}):
            filtered = [
                row
                for row in rows
                if row.get("platform") == platform
                and row.get("swizzle_backend_effective") == backend
                and row.get("workload_arena_kind")
            ]
            for kind in sorted({row.get("workload_arena_kind") for row in filtered}):
                rows_kind = [row for row in filtered if row.get("workload_arena_kind") == kind]
                for mode in sorted({row.get("mode") for row in rows_kind if row.get("mode")}):
                    counts = sorted(
                        {int(row.get("workload_arena_count", "0")) for row in rows_kind if row.get("workload_arena_count")}
                    )
                    values = []
                    for count in counts:
                        samples = [
                            _to_float(row.get("ms_per_cycle"))
                            for row in rows_kind
                            if row.get("mode") == mode and int(row.get("workload_arena_count", "0")) == count
                        ]
                        samples = [v for v in samples if v is not None]
                        values.append(sum(samples) / len(samples) if samples else 0.0)
                    inflections = _find_inflections(counts, values)
                    if inflections:
                        results.append(
                            {
                                "platform": platform,
                                "swizzle_backend_effective": backend,
                                "arena_kind": kind,
                                "mode": mode,
                                "counts": inflections,
                            }
                        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoronix-level benchmark suite.")
    parser.add_argument("--out-dir", default="bench_phoronix", help="Output directory.")
    parser.add_argument("--runs", type=int, default=5, help="Timed runs per workload.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per workload.")
    parser.add_argument("--cycles", type=int, default=3, help="BSP cycles per run.")
    parser.add_argument("--block-sizes", default="256", help="Comma list for block sizes.")
    parser.add_argument("--arena-counts", default="", help="Comma list of arena counts.")
    parser.add_argument("--swizzle-backends", default="jax,pallas,triton", help="Swizzle backend sweep.")
    parser.add_argument("--platforms", default="cpu,gpu", help="Comma list of platforms.")
    parser.add_argument("--workloads", default="", help="Workload filter for bench_compare.")
    parser.add_argument("--modes", default="", help="Mode filter for bench_compare.")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline mode.")
    parser.add_argument("--hierarchy-l1-mult", type=int, default=4, help="Hierarchy L1 multiplier.")
    parser.add_argument("--hierarchy-no-global", action="store_true", help="Disable global migration.")
    parser.add_argument("--hierarchy-morton", action="store_true", help="Include Morton hierarchy modes.")
    parser.add_argument("--metrics-block-size", type=int, default=256, help="Metrics block size.")
    parser.add_argument("--hierarchy-workload-l2", type=int, default=0, help="Hierarchy workload L2.")
    parser.add_argument("--hierarchy-workload-l1", type=int, default=0, help="Hierarchy workload L1.")
    parser.add_argument("--no-run", action="store_true", help="Skip running benchmarks.")
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    plots_dir = os.path.join(args.out_dir, "plots")
    _ensure_dir(plots_dir)

    system_info = _collect_system_info()
    platform_infos: Dict[str, Any] = {}
    raw_rows: List[Dict[str, str]] = []

    platforms = _parse_csv_list(args.platforms)
    for platform in platforms:
        platform_info = _introspect_jax(platform)
        platform_infos[platform] = platform_info
        if not _platform_available(platform_info, platform):
            continue
        out_csv = os.path.join(args.out_dir, f"bench_{platform}.csv")
        if not args.no_run:
            ok = _run_bench_compare(platform, out_csv, args)
            if not ok:
                continue
        if not os.path.exists(out_csv):
            continue
        rows = _read_csv(out_csv)
        for row in rows:
            row["platform"] = platform
            row["platform_effective"] = platform_info.get("default_backend", "")
        raw_rows.extend(rows)

    results_csv = os.path.join(args.out_dir, "results.csv")
    _write_csv(results_csv, raw_rows)

    summary_exec = _group_stats(
        raw_rows,
        [
            "platform",
            "platform_effective",
            "swizzle_backend_requested",
            "swizzle_backend_effective",
            "workload",
            "mode",
        ],
        "exec_ms",
        "exec_ms",
    )
    summary_cycle = _group_stats(
        raw_rows,
        [
            "platform",
            "platform_effective",
            "swizzle_backend_requested",
            "swizzle_backend_effective",
            "workload",
            "mode",
        ],
        "ms_per_cycle",
        "ms_per_cycle",
    )
    summary_rows = summary_exec + summary_cycle
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    _write_csv(summary_csv, summary_rows)

    plotter = Plotter()
    plot_files: List[str] = []
    for platform in platforms:
        backend = _select_backend(raw_rows, platform)
        if not backend:
            continue
        plot_files.extend(_plot_arena_scales(raw_rows, plots_dir, platform, backend, plotter))
        plot_files.extend(_plot_hierarchy_modes(raw_rows, plots_dir, platform, backend, plotter))

    inflections = _compute_inflections(raw_rows)
    summary_exec_sorted = sorted(
        summary_exec,
        key=lambda r: (-float(r.get("exec_ms_mean", 0.0)), r.get("workload", "")),
    )
    report_path = os.path.join(args.out_dir, "report.md")
    _write_report(
        report_path,
        system_info,
        platform_infos,
        args,
        summary_exec_sorted,
        inflections,
        plot_files,
    )

    system_path = os.path.join(args.out_dir, "system.json")
    with open(system_path, "w", encoding="utf-8") as f:
        json.dump(
            {"system": system_info, "platforms": platform_infos},
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote {results_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_path}")
    if plot_files:
        print(f"Wrote {len(plot_files)} plot files under {plots_dir}")


if __name__ == "__main__":
    main()
