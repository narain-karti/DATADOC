"""Standalone, high-fidelity HTML report generator for DATADOC."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import re
from typing import Any
import polars as pl

from datadoc.core.pipeline import (
    DataDocPipeline,
    PipelineConfig,
    _datadoc_version,
    profile_dataset,
)


def _compute_health_score(df: pl.DataFrame, profile_dict: dict[str, Any]) -> tuple[int, str]:
    """Computes a 0-100 data health score and letter grade."""
    score = 100
    rows = df.height
    cols = df.width
    if rows == 0 or cols == 0:
        return 0, "F"

    # Missing value penalty (up to 30 pts)
    total_cells = rows * cols
    null_counts = profile_dict.get("null_counts", {})
    total_nulls = sum(null_counts.values())
    null_pct = total_nulls / total_cells if total_cells else 0
    score -= min(int(null_pct * 100), 30)

    # Duplicate row penalty (up to 15 pts)
    dup_rows = profile_dict.get("duplicate_rows", 0)
    dup_pct = dup_rows / rows if rows else 0
    score -= min(int(dup_pct * 100), 15)

    # Constant columns penalty (10 pts per constant col, up to 20 pts)
    findings = profile_dict.get("findings", [])
    constant_findings = [f for f in findings if f.get("code") == "constant"]
    score -= min(len(constant_findings) * 10, 20)

    # Identifier retained penalty (10 pts)
    id_findings = [f for f in findings if f.get("code") == "identifier"]
    if id_findings:
        score -= 10

    score = max(0, min(100, score))
    if score >= 95:
        grade = "A+"
    elif score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"
    return score, grade


def _format_markdown_to_html(md: str) -> str:
    """Converts a subset of markdown (headers, bold, code, lists) into safe HTML."""
    if not md:
        return ""
    lines = md.splitlines()
    html_lines: list[str] = []
    in_code_block = False
    in_list = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code_block:
                html_lines.append("</code></pre>")
                in_code_block = False
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<pre class='ai-code-block'><code>")
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(html.escape(line))
            continue

        if stripped.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            title_text = html.escape(stripped[4:])
            html_lines.append(f"<h3 class='ai-section-title'>{title_text}</h3>")
            continue
        elif stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            title_text = html.escape(stripped[3:])
            html_lines.append(f"<h2 class='ai-section-title'>{title_text}</h2>")
            continue

        if stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul class='ai-list'>")
                in_list = True
            item_text = html.escape(stripped[2:])
            # inline code
            item_text = re.sub(r"&quot;([^&]+)&quot;", r"&ldquo;\1&rdquo;", item_text)
            item_text = re.sub(r"`([^`]+)`", r"<code class='ai-code'>\1</code>", item_text)
            item_text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", item_text)
            item_text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", item_text)
            html_lines.append(f"<li>{item_text}</li>")
            continue
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False

        if stripped:
            para = html.escape(stripped)
            para = re.sub(r"`([^`]+)`", r"<code class='ai-code'>\1</code>", para)
            para = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", para)
            para = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", para)
            html_lines.append(f"<p class='ai-p'>{para}</p>")

    if in_list:
        html_lines.append("</ul>")
    if in_code_block:
        html_lines.append("</code></pre>")

    return "\n".join(html_lines)


def generate_html_report(
    df: pl.DataFrame,
    target: str | None = None,
    title: str | None = None,
    pipeline: Any | None = None,
    dataset_name: str | None = None,
    ai_explanation: str | None = None,
) -> str:
    """Generate a 100% self-contained, responsive, beautiful HTML data health report."""
    cfg = PipelineConfig(target=target)
    prof = profile_dataset(df, cfg)
    prof_dict = prof.to_dict()

    if pipeline is not None and hasattr(pipeline, "plan_") and pipeline.plan_:
        plan_dict = pipeline.plan_.to_dict()
    else:
        plan_dict = DataDocPipeline(cfg).plan(df).to_dict()

    score, grade = _compute_health_score(df, prof_dict)
    rows = df.height
    cols = df.width
    total_cells = rows * cols
    null_counts = prof_dict.get("null_counts", {})
    total_nulls = sum(null_counts.values())
    null_rate = (total_nulls / total_cells * 100) if total_cells else 0.0
    dup_rows = prof_dict.get("duplicate_rows", 0)
    dup_rate = (dup_rows / rows * 100) if rows else 0.0

    report_title = title or "DATADOC Quality & Preparation Report"
    ds_name = dataset_name or "Dataset"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Column role counts
    roles_list = prof_dict.get("roles", [])
    role_counts: dict[str, int] = {}
    for r in roles_list:
        role = r.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    # Findings
    findings = prof_dict.get("findings", [])

    # HTML Helpers
    def esc(text: Any) -> str:
        return html.escape(str(text) if text is not None else "")

    # Role badge color
    def role_badge(role: str) -> str:
        colors = {
            "target": "badge-target",
            "feature_numeric": "badge-numeric",
            "feature_categorical": "badge-categorical",
            "feature_datetime": "badge-datetime",
            "identifier": "badge-identifier",
            "constant": "badge-constant",
            "text": "badge-text",
        }
        cls = colors.get(role, "badge-default")
        return f'<span class="badge {cls}">{esc(role)}</span>'

    # Build Column Rows Table
    col_rows_html = []
    cardinality_map = prof_dict.get("cardinality", {})
    schema_map = prof_dict.get("schema", {})

    for r in roles_list:
        col_name = r["name"]
        dtype_str = schema_map.get(col_name, str(df[col_name].dtype))
        col_null = null_counts.get(col_name, 0)
        col_null_pct = (col_null / rows * 100) if rows else 0.0
        n_unique = cardinality_map.get(col_name, df[col_name].n_unique())

        # sample 3 non-null values
        samples = [str(v) for v in df[col_name].drop_nulls().head(3).to_list()]
        sample_str = ", ".join(samples) if samples else "—"

        null_bar_class = (
            "bar-green" if col_null_pct == 0 else ("bar-amber" if col_null_pct < 20 else "bar-red")
        )

        col_rows_html.append(f"""
        <tr>
            <td class="font-bold">{esc(col_name)}</td>
            <td>{role_badge(r["role"])}</td>
            <td><code class="code-dtype">{esc(dtype_str)}</code></td>
            <td>
                <div class="null-cell">
                    <span>{col_null:,} ({col_null_pct:.1f}%)</span>
                    <div class="mini-bar-bg"><div class="mini-bar {null_bar_class}" style="width: {min(col_null_pct, 100)}%;"></div></div>
                </div>
            </td>
            <td>{n_unique:,}</td>
            <td class="text-dim text-sm truncate" title="{esc(sample_str)}">{esc(sample_str)}</td>
        </tr>
        """)

    # Build Numeric Feature Deep Dive Cards
    numeric_cards_html = []
    for r in roles_list:
        if r["role"] == "feature_numeric":
            col_name = r["name"]
            series = df[col_name].drop_nulls()
            if series.len() == 0:
                continue
            c_min = float(series.min() or 0.0)
            c_max = float(series.max() or 0.0)
            c_mean = float(series.mean() or 0.0)
            c_median = float(series.median() or 0.0)
            c_std = float(series.std() or 0.0)
            q1 = float(series.quantile(0.25) or 0.0)
            q3 = float(series.quantile(0.75) or 0.0)
            iqr = q3 - q1
            outliers = int(((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum())

            numeric_cards_html.append(f"""
            <div class="stat-card">
                <div class="stat-header">
                    <span class="stat-title">{esc(col_name)}</span>
                    <span class="badge badge-numeric">Numeric</span>
                </div>
                <div class="grid-stats">
                    <div class="stat-box"><span class="sb-lbl">Min</span><span class="sb-val">{c_min:.2f}</span></div>
                    <div class="stat-box"><span class="sb-lbl">Q1</span><span class="sb-val">{q1:.2f}</span></div>
                    <div class="stat-box highlight-box"><span class="sb-lbl">Median</span><span class="sb-val">{c_median:.2f}</span></div>
                    <div class="stat-box"><span class="sb-lbl">Q3</span><span class="sb-val">{q3:.2f}</span></div>
                    <div class="stat-box"><span class="sb-lbl">Max</span><span class="sb-val">{c_max:.2f}</span></div>
                    <div class="stat-box"><span class="sb-lbl">Mean</span><span class="sb-val">{c_mean:.2f}</span></div>
                    <div class="stat-box"><span class="sb-lbl">Std</span><span class="sb-val">{c_std:.2f}</span></div>
                    <div class="stat-box {"alert-box" if outliers > 0 else ""}"><span class="sb-lbl">Outliers</span><span class="sb-val">{outliers:,}</span></div>
                </div>
            </div>
            """)

    # Build Categorical Feature Deep Dive Cards
    categorical_cards_html = []
    for r in roles_list:
        if r["role"] == "feature_categorical":
            col_name = r["name"]
            values = df[col_name].cast(pl.String).fill_null("__MISSING__")
            vc = values.value_counts().sort("count", descending=True).head(5)
            total = max(values.len(), 1)
            cat_bars = []
            for row in vc.iter_rows():
                cat_val = str(row[0])
                count = int(row[1])
                pct = (count / total) * 100
                cat_bars.append(f"""
                <div class="cat-bar-row">
                    <span class="cat-label truncate" title="{esc(cat_val)}">{esc(cat_val)}</span>
                    <div class="cat-bar-track">
                        <div class="cat-bar-fill" style="width: {pct:.1f}%;"></div>
                    </div>
                    <span class="cat-pct">{count:,} ({pct:.1f}%)</span>
                </div>
                """)
            categorical_cards_html.append(f"""
            <div class="stat-card">
                <div class="stat-header">
                    <span class="stat-title">{esc(col_name)}</span>
                    <span class="badge badge-categorical">{cardinality_map.get(col_name, df[col_name].n_unique())} categories</span>
                </div>
                <div class="cat-list">
                    {"".join(cat_bars)}
                </div>
            </div>
            """)

    # Build Findings HTML
    findings_html = []
    if findings:
        for f in findings:
            sev = f.get("severity", "warning").lower()
            badge_class = (
                "badge-red"
                if sev == "error"
                else ("badge-amber" if sev == "warning" else "badge-blue")
            )
            findings_html.append(f"""
            <div class="finding-item finding-{sev}">
                <div class="finding-header">
                    <span class="badge {badge_class}">{esc(sev.upper())}</span>
                    <code class="code-finding">{esc(f.get("code", "finding"))}</code>
                    <span class="text-dim">column: <b>{esc(f.get("column", "*"))}</b></span>
                </div>
                <div class="finding-msg">{esc(f.get("message", ""))}</div>
            </div>
            """)
    else:
        findings_html.append(
            '<div class="clean-state">✅ No quality warnings or data hygiene issues detected.</div>'
        )

    # Build Planned Operations HTML
    plan_ops = plan_dict.get("operations", [])
    plan_rows_html = []
    for i, op in enumerate(plan_ops, 1):
        plan_rows_html.append(f"""
        <tr>
            <td class="text-dim font-mono">{i:02d}</td>
            <td><code class="code-op">{esc(op.get("operation", ""))}</code></td>
            <td class="font-bold">{esc(op.get("column", ""))}</td>
            <td>{esc(op.get("reason", ""))}</td>
        </tr>
        """)

    # Grade Color
    grade_colors = {
        "A+": "#10b981",
        "A": "#10b981",
        "B": "#3b82f6",
        "C": "#f59e0b",
        "D": "#ef4444",
        "F": "#dc2626",
    }
    grade_color = grade_colors.get(grade, "#10b981")

    # Complete Self-Contained HTML Document
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{esc(report_title)} - {esc(ds_name)}</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #111827;
            --surface-hover: #1f293d;
            --border: #1f2937;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --primary: #6366f1;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: var(--font);
            line-height: 1.5;
            padding: 24px;
            font-size: 14px;
        }}
        .container {{ max-width: 1280px; margin: 0 auto; }}
        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        }}
        .header-title h1 {{ font-size: 22px; font-weight: 700; color: #fff; margin-bottom: 4px; }}
        .header-meta {{ display: flex; gap: 12px; align-items: center; font-size: 13px; color: var(--text-dim); flex-wrap: wrap; }}
        .score-pill {{
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 18px;
        }}
        .score-grade {{ font-size: 32px; font-weight: 900; color: {grade_color}; line-height: 1; }}
        .score-val {{ font-size: 14px; font-weight: 600; color: var(--text); }}
        .score-lbl {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim); }}
        
        /* Badges */
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .badge-target {{ background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); }}
        .badge-numeric {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-categorical {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-datetime {{ background: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); }}
        .badge-identifier {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .badge-constant {{ background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }}
        .badge-default {{ background: #1e293b; color: #94a3b8; border: 1px solid #334155; }}
        .badge-red {{ background: rgba(239, 68, 68, 0.2); color: #fca5a5; }}
        .badge-amber {{ background: rgba(245, 158, 11, 0.2); color: #fde68a; }}
        .badge-blue {{ background: rgba(59, 130, 246, 0.2); color: #bfdbfe; }}

        /* KPI Cards */
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px 20px;
        }}
        .kpi-lbl {{ font-size: 12px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
        .kpi-val {{ font-size: 24px; font-weight: 700; color: #fff; }}
        .kpi-sub {{ font-size: 12px; color: var(--text-dim); margin-top: 4px; }}

        /* Section Cards */
        .section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }}
        .section-title {{ font-size: 16px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }}

        /* Tables */
        .table-wrap {{ width: 100%; overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{
            background: #1e293b;
            color: #cbd5e1;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            color: var(--text);
            vertical-align: middle;
        }}
        tr:hover td {{ background: var(--surface-hover); }}
        .code-dtype {{ font-family: var(--font-mono); color: #38bdf8; font-size: 12px; background: rgba(56,189,248,0.1); padding: 2px 6px; border-radius: 4px; }}
        .code-op {{ font-family: var(--font-mono); color: #818cf8; font-size: 12px; font-weight: 600; }}
        .code-finding {{ font-family: var(--font-mono); color: #fde68a; font-size: 12px; background: rgba(245,158,11,0.1); padding: 2px 6px; border-radius: 4px; }}

        /* Mini Progress Bars */
        .null-cell {{ display: flex; flex-direction: column; gap: 4px; }}
        .mini-bar-bg {{ width: 100px; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }}
        .mini-bar {{ height: 100%; border-radius: 3px; }}
        .bar-green {{ background: var(--success); }}
        .bar-amber {{ background: var(--warning); }}
        .bar-red {{ background: var(--danger); }}

        /* Grid for Feature Cards */
        .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
        .stat-card {{
            background: #182234;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
        }}
        .stat-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .stat-title {{ font-size: 14px; font-weight: 700; color: #fff; }}
        .grid-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
        .stat-box {{
            background: #111827;
            padding: 8px 10px;
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            border: 1px solid #1f2937;
        }}
        .highlight-box {{ border-color: rgba(99,102,241,0.4); background: rgba(99,102,241,0.05); }}
        .alert-box {{ border-color: rgba(239,68,68,0.4); }}
        .sb-lbl {{ font-size: 10px; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.04em; }}
        .sb-val {{ font-size: 13px; font-weight: 700; color: #fff; margin-top: 2px; font-family: var(--font-mono); }}

        /* Categorical Bars */
        .cat-list {{ display: flex; flex-direction: column; gap: 8px; }}
        .cat-bar-row {{ display: flex; align-items: center; gap: 10px; font-size: 12px; }}
        .cat-label {{ width: 120px; color: #cbd5e1; font-family: var(--font-mono); }}
        .cat-bar-track {{ flex: 1; height: 8px; background: #111827; border-radius: 4px; overflow: hidden; }}
        .cat-bar-fill {{ height: 100%; background: #6366f1; border-radius: 4px; }}
        .cat-pct {{ width: 90px; text-align: right; color: var(--text-dim); font-size: 11px; }}

        /* Findings */
        .finding-item {{
            background: #182234;
            border-left: 4px solid var(--warning);
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }}
        .finding-error {{ border-left-color: var(--danger); }}
        .finding-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; font-size: 12px; }}
        .finding-msg {{ font-size: 13px; color: #e2e8f0; }}
        .clean-state {{ padding: 18px; text-align: center; color: var(--success); font-weight: 600; font-size: 14px; }}

        /* Utility */
        .font-bold {{ font-weight: 700; }}
        .font-mono {{ font-family: var(--font-mono); }}
        .text-dim {{ color: var(--text-dim); }}
        .text-sm {{ font-size: 12px; }}
        .truncate {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        
        /* AI Advisory Card */
        .ai-card {{
            background: linear-gradient(180deg, rgba(99, 102, 241, 0.08) 0%, rgba(15, 23, 42, 0.6) 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 10px;
            padding: 20px 24px;
            color: #cbd5e1;
            line-height: 1.6;
        }}
        .ai-section-title {{
            font-size: 15px;
            font-weight: 700;
            color: #818cf8;
            margin-top: 18px;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid rgba(99, 102, 241, 0.2);
        }}
        .ai-section-title:first-child {{ margin-top: 0; }}
        .ai-list {{ margin-left: 20px; margin-bottom: 12px; }}
        .ai-list li {{ margin-bottom: 6px; color: #cbd5e1; }}
        .ai-p {{ margin-bottom: 10px; color: #cbd5e1; }}
        .ai-code {{
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: var(--font-mono);
            font-size: 12px;
        }}
        .ai-code-block {{
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 12px;
            overflow-x: auto;
            margin: 10px 0;
            font-size: 12px;
            color: #38bdf8;
            font-family: var(--font-mono);
        }}

        /* Footer */
        .footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-dim);
            padding: 24px 0 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div>
                <div class="header-title">
                    <h1>{esc(report_title)}</h1>
                </div>
                <div class="header-meta">
                    <span>Dataset: <b>{esc(ds_name)}</b></span>
                    <span>•</span>
                    <span>Fingerprint: <code class="font-mono">{
        prof.schema_fingerprint
    }</code></span>
                    <span>•</span>
                    <span>Target: {
        role_badge("target") if target else '<span class="text-dim">None</span>'
    }</span>
                    <span>•</span>
                    <span>Generated: {now_str}</span>
                </div>
            </div>
            <div class="score-pill">
                <div class="score-grade">{grade}</div>
                <div>
                    <div class="score-val">{score} / 100</div>
                    <div class="score-lbl">Health Score</div>
                </div>
            </div>
        </header>

        <!-- KPI Grid -->
        <section class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-lbl">Total Dimensions</div>
                <div class="kpi-val">{rows:,} × {cols:,}</div>
                <div class="kpi-sub">{rows * cols:,} total data cells</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-lbl">Missing Values</div>
                <div class="kpi-val" style="color: {
        "var(--danger)"
        if null_rate > 10
        else ("var(--warning)" if null_rate > 0 else "var(--success)")
    };">
                    {total_nulls:,} <span style="font-size: 14px; font-weight: 500;">({
        null_rate:.2f}%)</span>
                </div>
                <div class="kpi-sub">{
        sum(1 for c, n in null_counts.items() if n > 0)
    } columns with nulls</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-lbl">Duplicate Rows</div>
                <div class="kpi-val" style="color: {
        "var(--warning)" if dup_rows > 0 else "var(--success)"
    };">
                    {dup_rows:,} <span style="font-size: 14px; font-weight: 500;">({
        dup_rate:.1f}%)</span>
                </div>
                <div class="kpi-sub">Exact duplicate records</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-lbl">Feature Roles</div>
                <div class="kpi-val">{role_counts.get("feature_numeric", 0)} num / {
        role_counts.get("feature_categorical", 0)
    } cat</div>
                <div class="kpi-sub">{role_counts.get("identifier", 0)} id / {
        role_counts.get("constant", 0)
    } const / {role_counts.get("feature_datetime", 0)} dt</div>
            </div>
        </section>

        <!-- Findings & Quality Alerts -->
        <section class="section">
            <div class="section-header">
                <div class="section-title">
                    <span>⚠️</span>
                    <span>Quality Warnings & Findings ({len(findings)})</span>
                </div>
            </div>
            <div class="findings-list">
                {"".join(findings_html)}
            </div>
        </section>

        <!-- Column Schema & Role Inventory -->
        <section class="section">
            <div class="section-header">
                <div class="section-title">
                    <span>📋</span>
                    <span>Schema & Role Classification</span>
                </div>
                <span class="text-dim text-sm">{cols} total columns</span>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Role</th>
                            <th>Data Type</th>
                            <th>Missingness</th>
                            <th>Unique</th>
                            <th>Sample Values</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(col_rows_html)}
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Numeric Features Deep Dive -->
        {
        f'''
        <section class="section">
            <div class="section-header">
                <div class="section-title">
                    <span>📊</span>
                    <span>Numeric Distributions & Outliers</span>
                </div>
            </div>
            <div class="cards-grid">
                {''.join(numeric_cards_html)}
            </div>
        </section>
        '''
        if numeric_cards_html
        else ""
    }

        <!-- Categorical Features Deep Dive -->
        {
        f'''
        <section class="section">
            <div class="section-header">
                <div class="section-title">
                    <span>🏷️</span>
                    <span>Categorical Distributions (Top 5 Frequencies)</span>
                </div>
            </div>
            <div class="cards-grid">
                {''.join(categorical_cards_html)}
            </div>
        </section>
        '''
        if categorical_cards_html
        else ""
    }

        <!-- AI Executive Data Science Advisory -->
        {
        f'''
        <section class="section">
            <div class="section-header">
                <div class="section-title">
                    <span>🤖</span>
                    <span>AI Executive Data Science Advisory & Feature Engineering Hypotheses</span>
                </div>
                <span class="badge badge-target">AI Insights</span>
            </div>
            <div class="ai-card">
                {_format_markdown_to_html(ai_explanation)}
            </div>
        </section>
        '''
        if ai_explanation
        else ""
    }

        <!-- Recommended Preparation Plan -->
        <section class="section">
            <div class="section-header">
                <div class="section-title">
                    <span>⚙️</span>
                    <span>Recommended Preparation Pipeline Plan ({len(plan_ops)} operations)</span>
                </div>
                <span class="badge badge-target">Leakage-Safe</span>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Operation</th>
                            <th>Column</th>
                            <th>Rationale & Transformation Logic</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(plan_rows_html)}
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Footer -->
        <footer class="footer">
            Generated by <b>DATADOC</b> v{
        _datadoc_version()
    } — The Open Source Operating System for Dataset Engineering.
        </footer>
    </div>
</body>
</html>
"""
    return html_content
