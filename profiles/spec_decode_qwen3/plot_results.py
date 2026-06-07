#!/usr/bin/env python
"""Generate dependency-free SVG plots from a spec decode sweep summary."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
]


def as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nice_num(value: float) -> str:
    if math.isnan(value):
        return ""
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def read_rows(summary_csv: Path) -> list[dict[str, object]]:
    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    numeric_fields = {
        "num_speculative_tokens",
        "load_value",
        "output_throughput",
        "total_token_throughput",
        "request_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_ms",
        "mean_e2el_ms",
        "p95_tpot_ms",
        "p95_itl_ms",
        "spec_decode_acceptance_rate",
        "spec_decode_acceptance_length",
    }
    for row in rows:
        for field in numeric_fields:
            row[field] = as_float(row.get(field))
    return rows


def scale(values: list[float], lo_px: float, hi_px: float):
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = abs(lo) * 0.05 or 1.0
        lo -= pad
        hi += pad
    pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad

    def mapper(value: float) -> float:
        return lo_px + (value - lo) * (hi_px - lo_px) / (hi - lo)

    return mapper, lo, hi


def line_chart(
    title: str,
    rows: list[dict[str, object]],
    x_field: str,
    y_field: str,
    series_field: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    *,
    x_log2: bool = False,
) -> None:
    series: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x = row.get(x_field)
        y = row.get(y_field)
        series_key = row.get(series_field)
        if not isinstance(x, float) or not isinstance(y, float):
            continue
        if not isinstance(series_key, float):
            continue
        if x_log2 and x <= 0:
            continue
        series[series_key].append((math.log2(x) if x_log2 else x, y))

    series = {
        key: sorted(points)
        for key, points in sorted(series.items())
        if points
    }
    if not series:
        return

    all_x = [x for points in series.values() for x, _ in points]
    all_y = [y for points in series.values() for _, y in points]
    width = 980
    height = 560
    left = 84
    right = 210
    top = 52
    bottom = 74
    plot_w = width - left - right
    plot_h = height - top - bottom
    map_x, min_x, max_x = scale(all_x, left, left + plot_w)
    map_y, min_y, max_y = scale(all_y, top + plot_h, top)

    grid_y = [min_y + i * (max_y - min_y) / 4 for i in range(5)]
    raw_x_values = sorted(
        {
            float(row[x_field])
            for row in rows
            if isinstance(row.get(x_field), float)
            and (not x_log2 or float(row[x_field]) > 0)
        }
    )
    x_ticks = [(math.log2(x) if x_log2 else x, nice_num(x)) for x in raw_x_values]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="22" '
        f'font-weight="700">{html.escape(title)}</text>',
    ]
    for y in grid_y:
        py = map_y(y)
        parts.append(
            f'<line x1="{left}" x2="{left + plot_w}" y1="{py:.2f}" y2="{py:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{py + 4:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="12" fill="#4b5563">{nice_num(y)}</text>'
        )
    for x_value, label in x_ticks:
        px = map_x(x_value)
        parts.append(
            f'<line x1="{px:.2f}" x2="{px:.2f}" y1="{top}" y2="{top + plot_h}" '
            'stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{top + plot_h + 24}" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="#4b5563">{label}</text>'
        )
    parts.append(
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" '
        'stroke="#111827" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" '
        f'y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" '
        'font-family="Arial" font-size="14" fill="#111827">'
        f'{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text transform="translate(22 {top + plot_h / 2}) rotate(-90)" '
        f'text-anchor="middle" font-family="Arial" font-size="14" '
        f'fill="#111827">{html.escape(y_label)}</text>'
    )

    legend_x = left + plot_w + 26
    legend_y = top + 10
    for idx, (key, points) in enumerate(series.items()):
        color = PALETTE[idx % len(PALETTE)]
        path_points = " ".join(
            f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in points
        )
        parts.append(
            f'<polyline points="{path_points}" fill="none" stroke="{color}" '
            'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{map_x(x):.2f}" cy="{map_y(y):.2f}" r="3.5" '
                f'fill="{color}"/>'
            )
        ly = legend_y + idx * 24
        parts.append(
            f'<line x1="{legend_x}" x2="{legend_x + 20}" y1="{ly}" y2="{ly}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 28}" y="{ly + 4}" font-family="Arial" '
            f'font-size="13" fill="#111827">k={int(key)}</text>'
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n")


def write_dashboard(result_dir: Path, plot_files: list[Path]) -> None:
    cards = []
    for path in plot_files:
        cards.append(
            "<section>"
            f"<h2>{html.escape(path.stem.replace('_', ' ').title())}</h2>"
            f"<img src=\"plots/{html.escape(path.name)}\" "
            f"alt=\"{html.escape(path.stem)}\">"
            "</section>"
        )
    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Spec Decode Sweep Plots</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #111827; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    p {{ margin: 0 0 24px; color: #4b5563; }}
    section {{ margin: 0 0 34px; }}
    h2 {{ font-size: 17px; margin: 0 0 10px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #e5e7eb; }}
  </style>
</head>
<body>
  <h1>Spec Decode Sweep Plots</h1>
  <p>{html.escape(str(result_dir))}</p>
  {''.join(cards)}
</body>
</html>
"""
    (result_dir / "plots.html").write_text(html_doc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir
    summary_csv = result_dir / "summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"Missing summary CSV: {summary_csv}")

    rows = read_rows(summary_csv)
    if not rows:
        raise SystemExit(f"No rows found in {summary_csv}")

    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    load_mode = next(
        (str(row.get("load_mode")) for row in rows if row.get("load_mode")),
        "load",
    )
    x_log2 = load_mode == "concurrency"

    specs = [
        (
            "Output Throughput vs Load",
            "output_throughput",
            "Output throughput (tokens/s)",
            "output_throughput_vs_load.svg",
        ),
        (
            "Mean TPOT vs Load",
            "mean_tpot_ms",
            "Mean TPOT (ms/token)",
            "mean_tpot_vs_load.svg",
        ),
        (
            "Mean ITL vs Load",
            "mean_itl_ms",
            "Mean ITL (ms/update)",
            "mean_itl_vs_load.svg",
        ),
        (
            "Mean E2E Latency vs Load",
            "mean_e2el_ms",
            "Mean E2E latency (ms)",
            "mean_e2el_vs_load.svg",
        ),
        (
            "Acceptance Length vs Load",
            "spec_decode_acceptance_length",
            "Acceptance length",
            "acceptance_length_vs_load.svg",
        ),
        (
            "Acceptance Rate vs Load",
            "spec_decode_acceptance_rate",
            "Acceptance rate (%)",
            "acceptance_rate_vs_load.svg",
        ),
    ]
    plot_files = []
    for title, metric, y_label, filename in specs:
        out_path = plot_dir / filename
        line_chart(
            title,
            rows,
            "load_value",
            metric,
            "num_speculative_tokens",
            load_mode.replace("_", " ").title(),
            y_label,
            out_path,
            x_log2=x_log2,
        )
        if out_path.exists():
            plot_files.append(out_path)

    write_dashboard(result_dir, plot_files)
    print(f"Wrote {len(plot_files)} plots to {plot_dir}")
    print(f"Wrote dashboard to {result_dir / 'plots.html'}")


if __name__ == "__main__":
    main()
