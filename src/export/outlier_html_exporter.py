"""Self-contained HTML report for outlier detection.

Replaces the former openpyxl/Excel workbook export. Produces a single .html file
with: a summary, per-dataset group-statistics tables, the raw values with
detected outliers highlighted, and an embedded swarm/box visualization (base64
PNG). No external assets — the file is portable.
"""
from __future__ import annotations

import base64
import html
import io
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _outlier_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in ("Grubbs_Outlier", "ModZ_Outlier") if c in df.columns]


def _primary_outlier_col(df: pd.DataFrame) -> str | None:
    cols = _outlier_columns(df)
    return cols[0] if cols else None


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _plot_base64(df: pd.DataFrame, group_col: str, value_col: str, outlier_col: str) -> str | None:
    """Box + swarm/strip plot with outliers highlighted, returned as a base64 PNG
    (data payload only). Mirrors the former Excel visualization sheet."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return None

    try:
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("white")
        sns.boxplot(x=group_col, y=value_col, data=df, fliersize=0, width=0.5,
                    color="#DDDDDD", ax=ax)
        plot_fn = sns.swarmplot if len(df) < 100 else sns.stripplot
        kw = {} if len(df) < 100 else {"jitter": 0.25, "alpha": 0.8, "dodge": False}
        plot_fn(x=group_col, y=value_col, data=df, hue=outlier_col,
                palette={False: "#555555", True: "#007AFF"}, ax=ax, **kw)
        sns.despine(ax=ax, top=True, right=True)
        ax.set_xlabel(group_col, color="#333333")
        ax.set_ylabel(value_col, color="#333333")
        ax.set_title(f"{value_col} by group (outliers highlighted)", color="#333333", pad=12)
        handles, labels = ax.get_legend_handles_labels()
        remap = {"False": "Normal", "True": "Outlier"}
        nh = [(h, remap[l]) for h, l in zip(handles, labels) if l in remap]
        if nh:
            ax.legend([h for h, _ in nh], [l for _, l in nh], title="Status",
                      loc="upper right", frameon=False)
        ax.grid(axis="y", color="#f0f0f0", linewidth=0.7)
        ax.set_axisbelow(True)
        fig.tight_layout(pad=1)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def _group_stats_table(df: pd.DataFrame, group_col: str, value_col: str) -> str:
    out_cols = _outlier_columns(df)
    head = ["Group", "Count", "Mean", "StdDev", "Median", "Min", "Max"] + \
           [c.replace("_Outlier", " outliers") for c in out_cols]
    rows = []
    for g, gdf in df.groupby(group_col):
        vals = pd.to_numeric(gdf[value_col], errors="coerce").dropna()
        cells = [
            _esc(g), str(len(gdf)),
            f"{vals.mean():.4g}" if len(vals) else "—",
            f"{vals.std(ddof=1):.4g}" if len(vals) > 1 else "—",
            f"{vals.median():.4g}" if len(vals) else "—",
            f"{vals.min():.4g}" if len(vals) else "—",
            f"{vals.max():.4g}" if len(vals) else "—",
        ]
        cells += [str(int(gdf[c].sum())) for c in out_cols]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in head)
    return f"<table class='t'><thead><tr>{head_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _raw_table(df: pd.DataFrame, group_col: str, value_col: str) -> str:
    out_cols = _outlier_columns(df)
    head = [group_col, value_col] + [c.replace("_Outlier", "") for c in out_cols]
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in head)
    rows = []
    for _, r in df.iterrows():
        is_out = any(bool(r.get(c)) for c in out_cols)
        cells = [_esc(r[group_col]), _esc(r[value_col])] + \
                ["✓" if bool(r.get(c)) else "" for c in out_cols]
        cls = " class='out'" if is_out else ""
        rows.append(f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f"<table class='t'><thead><tr>{head_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _dataset_section(detector, dataset_name: str) -> str:
    df = detector.df
    group_col, value_col = detector.group_col, detector.value_col
    summary = detector.get_summary()
    methods = []
    if "Grubbs_Outlier" in df.columns:
        methods.append(("Grubbs", int(df["Grubbs_Outlier"].sum())))
    if "ModZ_Outlier" in df.columns:
        methods.append(("Modified Z-Score", int(df["ModZ_Outlier"].sum())))
    method_html = ", ".join(f"{name}: {n}" for name, n in methods) or "none"

    parts = [f"<h2>{_esc(dataset_name)}</h2>"]
    parts.append(f"<p class='meta'>Total values: {summary['total_rows']} · "
                 f"Outliers — {method_html}</p>")

    pcol = _primary_outlier_col(df)
    if pcol is not None:
        img = _plot_base64(df, group_col, value_col, pcol)
        if img:
            parts.append(f"<img class='plot' alt='Outlier plot for {_esc(dataset_name)}' "
                         f"src='data:image/png;base64,{img}'/>")
    parts.append("<h3>Group statistics</h3>")
    parts.append(_group_stats_table(df, group_col, value_col))
    parts.append("<h3>Values (outliers highlighted)</h3>")
    parts.append(_raw_table(df, group_col, value_col))
    return "<section>" + "".join(parts) + "</section>"


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1f2a2e;margin:24px;background:#fff;}
h1{color:#0f766e;margin:0 0 4px;} h2{color:#0f766e;border-bottom:2px solid #e2e8f0;padding-bottom:4px;margin-top:32px;}
h3{color:#334155;margin:18px 0 6px;} .meta{color:#64748b;margin:2px 0 12px;}
table.t{border-collapse:collapse;margin:6px 0 16px;font-size:13px;} table.t th,table.t td{border:1px solid #d8e2e8;padding:4px 8px;text-align:right;}
table.t th{background:#eef8f6;color:#0f766e;} table.t td:first-child,table.t th:first-child{text-align:left;}
tr.out{background:#fff1f2;} tr.out td{font-weight:600;color:#be123c;}
img.plot{max-width:760px;width:100%;height:auto;border:1px solid #e2e8f0;border-radius:6px;margin:8px 0;}
.failed{color:#be123c;}
"""


def _wrap(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{_esc(title)}</title><style>{_CSS}</style></head>"
            f"<body>{body}</body></html>")


def export_single(detector, output_path: str) -> str:
    """Write a single-dataset outlier HTML report. Returns the written path."""
    title = f"Outlier Report — {detector.value_col}"
    body = f"<h1>Outlier Detection Report</h1>" + _dataset_section(detector, detector.value_col)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(_wrap(title, body))
    return output_path


def export_multi(all_results: Dict[str, dict], failed_datasets: Dict[str, str],
                 dataset_columns: List[str], output_path: str) -> str:
    """Write a combined multi-dataset outlier HTML report. ``all_results`` maps
    dataset name -> {'detector': OutlierDetector, 'summary': {...}}."""
    body = ["<h1>Multi-Dataset Outlier Report</h1>"]
    body.append(f"<p class='meta'>{len(all_results)} of {len(dataset_columns)} datasets analyzed"
                + (f" · {len(failed_datasets)} failed" if failed_datasets else "") + "</p>")
    for name in dataset_columns:
        entry = all_results.get(name)
        if entry and entry.get("detector") is not None:
            body.append(_dataset_section(entry["detector"], name))
        elif name in failed_datasets:
            body.append(f"<section><h2>{_esc(name)}</h2>"
                        f"<p class='failed'>Analysis failed: {_esc(failed_datasets[name])}</p></section>")
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(_wrap("Multi-Dataset Outlier Report", "".join(body)))
    return output_path
