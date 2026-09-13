"""Dataset comparison module for comparing raw and transformed datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
from typing import Any
import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class DatasetComparison:
    raw_rows: int
    raw_cols: int
    transformed_rows: int
    transformed_cols: int
    row_delta: int
    col_delta: int
    raw_null_cells: int
    transformed_null_cells: int
    null_reduction: int
    null_reduction_pct: float
    retained_columns: list[str]
    dropped_columns: list[str]
    engineered_columns: list[str]
    column_null_deltas: dict[str, dict[str, Any]]
    numeric_shifts: dict[str, dict[str, Any]]
    target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_datasets(
    raw_df: pl.DataFrame,
    transformed_df: pl.DataFrame,
    target: str | None = None,
) -> DatasetComparison:
    """Computes comprehensive comparison metrics between raw and transformed datasets."""
    r_rows, r_cols = raw_df.height, raw_df.width
    t_rows, t_cols = transformed_df.height, transformed_df.width

    row_delta = t_rows - r_rows
    col_delta = t_cols - r_cols

    # Null counts
    raw_null_counts = {c: int(raw_df[c].null_count()) for c in raw_df.columns}
    trans_null_counts = {c: int(transformed_df[c].null_count()) for c in transformed_df.columns}

    r_null_total = sum(raw_null_counts.values())
    t_null_total = sum(trans_null_counts.values())
    null_reduction = r_null_total - t_null_total
    null_red_pct = (null_reduction / r_null_total * 100.0) if r_null_total > 0 else 0.0

    raw_col_set = set(raw_df.columns)
    trans_col_set = set(transformed_df.columns)

    retained = sorted(raw_col_set & trans_col_set)
    dropped = sorted(raw_col_set - trans_col_set)
    engineered = sorted(trans_col_set - raw_col_set)

    # Per-column null deltas for raw columns
    col_null_deltas = {}
    for c in raw_df.columns:
        r_null = raw_null_counts.get(c, 0)
        t_null = trans_null_counts.get(c, 0) if c in trans_col_set else None
        col_null_deltas[c] = {
            "raw_nulls": r_null,
            "transformed_nulls": t_null,
            "resolved": (r_null > 0 and t_null == 0) if t_null is not None else False,
            "status": "retained" if c in trans_col_set else "dropped",
        }

    # Numeric distribution shifts for columns present in both
    numeric_shifts = {}
    for c in retained:
        r_s = raw_df[c]
        t_s = transformed_df[c]
        if r_s.dtype.is_numeric() and t_s.dtype.is_numeric():
            r_non_null = r_s.drop_nulls()
            t_non_null = t_s.drop_nulls()
            if r_non_null.len() > 0 and t_non_null.len() > 0:
                numeric_shifts[c] = {
                    "raw_mean": float(r_non_null.mean() or 0.0),
                    "trans_mean": float(t_non_null.mean() or 0.0),
                    "raw_std": float(r_non_null.std() or 0.0),
                    "trans_std": float(t_non_null.std() or 0.0),
                    "raw_min": float(r_non_null.min() or 0.0),
                    "trans_min": float(t_non_null.min() or 0.0),
                    "raw_max": float(r_non_null.max() or 0.0),
                    "trans_max": float(t_non_null.max() or 0.0),
                }

    return DatasetComparison(
        raw_rows=r_rows,
        raw_cols=r_cols,
        transformed_rows=t_rows,
        transformed_cols=t_cols,
        row_delta=row_delta,
        col_delta=col_delta,
        raw_null_cells=r_null_total,
        transformed_null_cells=t_null_total,
        null_reduction=null_reduction,
        null_reduction_pct=null_red_pct,
        retained_columns=retained,
        dropped_columns=dropped,
        engineered_columns=engineered,
        column_null_deltas=col_null_deltas,
        numeric_shifts=numeric_shifts,
        target=target,
    )


def print_comparison_table(comp: DatasetComparison, console: Console | None = None) -> None:
    """Print an aesthetic, color-coded comparison report to the terminal using Rich."""
    c = console or Console()

    # Overview panel
    row_delta_str = (
        f"+{comp.row_delta}"
        if comp.row_delta > 0
        else (str(comp.row_delta) if comp.row_delta < 0 else "0 (unchanged)")
    )
    col_delta_str = (
        f"+{comp.col_delta}"
        if comp.col_delta > 0
        else (str(comp.col_delta) if comp.col_delta < 0 else "0 (unchanged)")
    )
    null_status = (
        f"[green]-{comp.null_reduction:,} (-{comp.null_reduction_pct:.1f}%)[/green]"
        if comp.null_reduction > 0
        else "[dim]None[/dim]"
    )

    summary_text = (
        f"[bold]Rows:[/bold] {comp.raw_rows:,} -> {comp.transformed_rows:,} ({row_delta_str})\n"
        f"[bold]Columns:[/bold] {comp.raw_cols} -> {comp.transformed_cols} ({col_delta_str})\n"
        f"[bold]Missing Cells:[/bold] {comp.raw_null_cells:,} -> {comp.transformed_null_cells:,} ({null_status})\n"
        f"[bold]Feature Lifecycle:[/bold] [green]{len(comp.retained_columns)} retained[/green], "
        f"[yellow]{len(comp.dropped_columns)} dropped[/yellow], [cyan]{len(comp.engineered_columns)} engineered[/cyan]"
    )
    c.print(Panel(summary_text, title="DATADOC Dataset Comparison Summary", border_style="cyan"))

    # Column Lifecycle Table
    tbl_cols = Table(title="Column Lineage & Lifecycle", show_header=True, header_style="bold cyan")
    tbl_cols.add_column("Category")
    tbl_cols.add_column("Count")
    tbl_cols.add_column("Columns")

    tbl_cols.add_row(
        "[green]Retained[/green]",
        str(len(comp.retained_columns)),
        ", ".join(comp.retained_columns) or "—",
    )
    tbl_cols.add_row(
        "[yellow]Dropped (IDs/Const)[/yellow]",
        str(len(comp.dropped_columns)),
        ", ".join(comp.dropped_columns) or "—",
    )
    tbl_cols.add_row(
        "[cyan]Engineered (OHE/Freq/Flags)[/cyan]",
        str(len(comp.engineered_columns)),
        ", ".join(comp.engineered_columns[:15])
        + ("..." if len(comp.engineered_columns) > 15 else "")
        or "—",
    )
    c.print(tbl_cols)

    # Missing values resolution table
    null_candidates = [c for c, d in comp.column_null_deltas.items() if d["raw_nulls"] > 0]
    if null_candidates:
        tbl_nulls = Table(
            title="Missing Value Resolution", show_header=True, header_style="bold magenta"
        )
        tbl_nulls.add_column("Column")
        tbl_nulls.add_column("Raw Nulls")
        tbl_nulls.add_column("Transformed Nulls")
        tbl_nulls.add_column("Status")

        for col in null_candidates:
            d = comp.column_null_deltas[col]
            t_str = (
                str(d["transformed_nulls"])
                if d["transformed_nulls"] is not None
                else "[dim]dropped[/dim]"
            )
            status = (
                "[green]Resolved (100%)[/green]"
                if d["resolved"]
                else (
                    "[dim]Dropped[/dim]"
                    if d["status"] == "dropped"
                    else f"[yellow]{d['transformed_nulls']} remaining[/yellow]"
                )
            )
            tbl_nulls.add_row(col, f"{d['raw_nulls']:,}", t_str, status)
        c.print(tbl_nulls)

    # Numeric shifts table
    if comp.numeric_shifts:
        tbl_shifts = Table(
            title="Numeric Feature Distribution Shifts", show_header=True, header_style="bold green"
        )
        tbl_shifts.add_column("Feature")
        tbl_shifts.add_column("Mean (Raw -> Trans)")
        tbl_shifts.add_column("Std (Raw -> Trans)")
        tbl_shifts.add_column("Range (Min..Max -> Trans)")

        for col, s in comp.numeric_shifts.items():
            tbl_shifts.add_row(
                col,
                f"{s['raw_mean']:.2f} -> {s['trans_mean']:.2f}",
                f"{s['raw_std']:.2f} -> {s['trans_std']:.2f}",
                f"[{s['raw_min']:.1f}, {s['raw_max']:.1f}] -> [{s['trans_min']:.1f}, {s['trans_max']:.1f}]",
            )
        c.print(tbl_shifts)


def compare_to_html(comp: DatasetComparison, title: str | None = None) -> str:
    """Generate a standalone HTML comparison page."""
    report_title = title or "DATADOC Dataset Comparison (Before vs After)"

    def esc(text: Any) -> str:
        return html.escape(str(text) if text is not None else "")

    # Retained rows HTML
    retained_chips = (
        "".join(f'<span class="chip chip-green">{esc(c)}</span>' for c in comp.retained_columns)
        or "—"
    )
    dropped_chips = (
        "".join(f'<span class="chip chip-amber">{esc(c)}</span>' for c in comp.dropped_columns)
        or "—"
    )
    engineered_chips = (
        "".join(f'<span class="chip chip-cyan">{esc(c)}</span>' for c in comp.engineered_columns)
        or "—"
    )

    # Null rows HTML
    null_rows = []
    for col, d in comp.column_null_deltas.items():
        if d["raw_nulls"] > 0:
            t_val = str(d["transformed_nulls"]) if d["transformed_nulls"] is not None else "Dropped"
            badge = (
                '<span class="badge badge-green">Resolved</span>'
                if d["resolved"]
                else (
                    '<span class="badge badge-dim">Dropped</span>'
                    if d["status"] == "dropped"
                    else '<span class="badge badge-amber">Partial</span>'
                )
            )
            null_rows.append(f"""
            <tr>
                <td class="font-bold">{esc(col)}</td>
                <td>{d["raw_nulls"]:,}</td>
                <td>{t_val}</td>
                <td>{badge}</td>
            </tr>
            """)

    # Numeric rows HTML
    num_rows = []
    for col, s in comp.numeric_shifts.items():
        num_rows.append(f"""
        <tr>
            <td class="font-bold">{esc(col)}</td>
            <td><code>{s["raw_mean"]:.2f} &rarr; {s["trans_mean"]:.2f}</code></td>
            <td><code>{s["raw_std"]:.2f} &rarr; {s["trans_std"]:.2f}</code></td>
            <td><code>[{s["raw_min"]:.1f}, {s["raw_max"]:.1f}] &rarr; [{s["trans_min"]:.1f}, {s["trans_max"]:.1f}]</code></td>
        </tr>
        """)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{esc(report_title)}</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #111827;
            --border: #1f2937;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
            --primary: #6366f1;
            --cyan: #06b6d4;
            --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: var(--bg); color: var(--text); font-family: var(--font); padding: 24px; font-size: 14px; line-height: 1.5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
        .header h1 {{ font-size: 22px; color: #fff; margin-bottom: 6px; }}
        .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }}
        .card-lbl {{ font-size: 11px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 6px; letter-spacing: 0.05em; }}
        .card-val {{ font-size: 24px; font-weight: 700; color: #fff; }}
        .card-sub {{ font-size: 12px; color: var(--text-dim); margin-top: 4px; }}
        .section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
        .sec-title {{ font-size: 16px; font-weight: 700; color: #fff; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }}
        .chips-wrap {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
        .chip {{ font-family: var(--font-mono); font-size: 12px; padding: 4px 10px; border-radius: 6px; }}
        .chip-green {{ background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }}
        .chip-amber {{ background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }}
        .chip-cyan {{ background: rgba(6,182,212,0.15); color: #22d3ee; border: 1px solid rgba(6,182,212,0.3); }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ background: #1e293b; color: #cbd5e1; font-size: 12px; text-transform: uppercase; padding: 10px 14px; border-bottom: 1px solid var(--border); }}
        td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
        code {{ font-family: var(--font-mono); font-size: 12px; color: #38bdf8; }}
        .badge {{ font-size: 11px; font-weight: 600; text-transform: uppercase; padding: 3px 8px; border-radius: 4px; }}
        .badge-green {{ background: rgba(16,185,129,0.2); color: #34d399; }}
        .badge-amber {{ background: rgba(245,158,11,0.2); color: #fbbf24; }}
        .badge-dim {{ background: #1e293b; color: #94a3b8; }}
        .font-bold {{ font-weight: 700; }}
        .footer {{ text-align: center; color: var(--text-dim); font-size: 12px; padding: 18px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{esc(report_title)}</h1>
            <p style="color: var(--text-dim); font-size: 13px;">Visual transformation delta & lineage validation</p>
        </div>

        <div class="cards-grid">
            <div class="card">
                <div class="card-lbl">Dimensions</div>
                <div class="card-val">{comp.raw_rows} &times; {comp.raw_cols} &rarr; {
        comp.transformed_rows
    } &times; {comp.transformed_cols}</div>
                <div class="card-sub">{
        f"+{comp.col_delta}" if comp.col_delta >= 0 else str(comp.col_delta)
    } features delta</div>
            </div>
            <div class="card">
                <div class="card-lbl">Missing Cells</div>
                <div class="card-val" style="color: var(--success);">{
        comp.raw_null_cells:,} &rarr; {comp.transformed_null_cells:,}</div>
                <div class="card-sub">{
        f"-{comp.null_reduction_pct:.1f}% reduction" if comp.null_reduction > 0 else "Zero nulls"
    }</div>
            </div>
            <div class="card">
                <div class="card-lbl">Columns Retained</div>
                <div class="card-val" style="color: var(--success);">{
        len(comp.retained_columns)
    }</div>
                <div class="card-sub">Features kept from raw dataset</div>
            </div>
            <div class="card">
                <div class="card-lbl">Columns Engineered</div>
                <div class="card-val" style="color: var(--cyan);">{
        len(comp.engineered_columns)
    }</div>
                <div class="card-sub">One-hot, frequency, indicators, etc.</div>
            </div>
        </div>

        <div class="section">
            <div class="sec-title">Column Lifecycle & Lineage</div>
            <div style="margin-bottom: 16px;">
                <h4 style="font-size: 13px; color: var(--text-dim); margin-bottom: 4px;">RETAINED COLUMNS ({
        len(comp.retained_columns)
    })</h4>
                <div class="chips-wrap">{retained_chips}</div>
            </div>
            <div style="margin-bottom: 16px;">
                <h4 style="font-size: 13px; color: var(--text-dim); margin-bottom: 4px;">DROPPED COLUMNS ({
        len(comp.dropped_columns)
    })</h4>
                <div class="chips-wrap">{dropped_chips}</div>
            </div>
            <div>
                <h4 style="font-size: 13px; color: var(--text-dim); margin-bottom: 4px;">ENGINEERED COLUMNS ({
        len(comp.engineered_columns)
    })</h4>
                <div class="chips-wrap">{engineered_chips}</div>
            </div>
        </div>

        {
        f'''
        <div class="section">
            <div class="sec-title">Missing Values Resolution</div>
            <table>
                <thead>
                    <tr><th>Column</th><th>Raw Nulls</th><th>Transformed Nulls</th><th>Resolution Status</th></tr>
                </thead>
                <tbody>{''.join(null_rows)}</tbody>
            </table>
        </div>
        '''
        if null_rows
        else ""
    }

        {
        f'''
        <div class="section">
            <div class="sec-title">Numeric Distribution Shifts (Imputation & Scaling)</div>
            <table>
                <thead>
                    <tr><th>Feature</th><th>Mean (Before &rarr; After)</th><th>Std (Before &rarr; After)</th><th>Min..Max (Before &rarr; After)</th></tr>
                </thead>
                <tbody>{''.join(num_rows)}</tbody>
            </table>
        </div>
        '''
        if num_rows
        else ""
    }

        <div class="footer">
            Generated by <b>DATADOC</b> v0.6.0 — The Open Source Operating System for Dataset Engineering.
        </div>
    </div>
</body>
</html>
"""
