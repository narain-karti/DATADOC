import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Template
from datadoc.core.engine import DATADOC

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DATADOC Visual Comparison</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-primary: #3b82f6;
            --accent-secondary: #10b981;
            --accent-danger: #ef4444;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 85% 30%, rgba(16, 185, 129, 0.15) 0%, transparent 50%);
            background-attachment: fixed;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            text-align: center;
            margin-bottom: 4rem;
            padding-top: 3rem;
        }

        h1 {
            font-size: 3.5rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(to right, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
        }

        .subtitle {
            font-size: 1.2rem;
            color: var(--text-muted);
            margin-top: 1rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 4rem;
        }

        .stat-card {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0.5rem 0;
            color: var(--text-main);
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .diff-positive { color: var(--accent-secondary); }
        .diff-negative { color: var(--accent-danger); }

        .chart-section {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 3rem;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        }

        .chart-section h2 {
            margin-top: 0;
            font-size: 1.8rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 1rem;
        }

        .plugin-list {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 4rem;
        }

        .plugin-badge {
            background: rgba(59, 130, 246, 0.2);
            color: #93c5fd;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.9rem;
            font-weight: 600;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        /* Plotly overrides for dark theme */
        .js-plotly-plot .plotly .bg { fill: transparent !important; }
        .js-plotly-plot .plotly .paper { fill: transparent !important; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>DATADOC Visual Comparison</h1>
            <div class="subtitle">Dataset Analysis: <strong>{{ filename }}</strong></div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Rows</div>
                <div class="stat-value">{{ original_rows }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Columns</div>
                <div class="stat-value">{{ clean_cols }} <span style="font-size:1.2rem; color:var(--text-muted)">({{ cols_diff }})</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Missing Values (Before)</div>
                <div class="stat-value diff-negative">{{ original_missing }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Missing Values (After)</div>
                <div class="stat-value diff-positive">{{ clean_missing }}</div>
            </div>
        </div>

        <div style="text-align:center; margin-bottom: 1rem;">
            <div class="stat-label">Applied Plugins</div>
        </div>
        <div class="plugin-list">
            {% for plugin in applied_plugins %}
                <div class="plugin-badge">{{ plugin }}</div>
            {% endfor %}
            {% if not applied_plugins %}
                <div class="plugin-badge" style="background: rgba(255,255,255,0.1); color: #ccc;">No plugins applied (Clean)</div>
            {% endif %}
        </div>

        {% if missing_plot_html %}
        <div class="chart-section">
            <h2>Missing Values Resolution</h2>
            <div>{{ missing_plot_html | safe }}</div>
        </div>
        {% endif %}

        {% if num_plots_html %}
        <div class="chart-section">
            <h2>Numerical Feature Distributions (Before vs After)</h2>
            <p style="color:var(--text-muted); margin-bottom:2rem;">
                Shows original data (blue) compared to engineered data (green). Observe imputations, outlier clipping, and scaling.
            </p>
            {% for plot in num_plots_html %}
                <div style="margin-bottom: 3rem;">{{ plot | safe }}</div>
            {% endfor %}
        </div>
        {% endif %}

    </div>
</body>
</html>
"""

class DashboardGenerator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.doc = DATADOC(file_path)
        self.orig_df = self.doc.df.copy()
        
        # Run engineering to get the clean df
        self.clean_df = self.doc.engineer()
        
        # Plot layout defaults for dark theme
        self.plot_layout = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc', family='Inter'),
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        )

    def generate_missing_chart(self) -> str:
        orig_missing = self.orig_df.isnull().sum()
        clean_missing = self.clean_df.isnull().sum()
        
        # Only include columns that originally had missing values
        cols_with_missing = orig_missing[orig_missing > 0].index.tolist()
        
        if not cols_with_missing:
            return ""

        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=cols_with_missing,
            y=orig_missing[cols_with_missing],
            name='Original Missing',
            marker_color='#ef4444' # Red
        ))
        
        # For clean data, some columns might have been dropped (e.g. by dates or encoding),
        # so safely get the count or 0.
        clean_y = [clean_missing.get(col, 0) for col in cols_with_missing]
        
        fig.add_trace(go.Bar(
            x=cols_with_missing,
            y=clean_y,
            name='After Engineering',
            marker_color='#10b981' # Green
        ))

        fig.update_layout(
            **self.plot_layout,
            barmode='group',
            title='Missing Values by Column',
            xaxis_title="Column",
            yaxis_title="Missing Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def generate_numeric_comparisons(self) -> list[str]:
        plots = []
        num_cols = self.orig_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in num_cols:
            if col not in self.clean_df.columns:
                continue # Column was dropped/transformed entirely
                
            orig_data = self.orig_df[col].dropna()
            clean_data = self.clean_df[col].dropna()
            
            # Skip if std is 0 (constant) or empty
            if len(orig_data) == 0 or len(clean_data) == 0:
                continue
                
            # Create overlaid histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=orig_data,
                name='Original',
                marker_color='rgba(59, 130, 246, 0.6)', # Blue, semi-transparent
                xbins=dict(size=(orig_data.max() - orig_data.min())/50 if orig_data.max() != orig_data.min() else 1)
            ))
            fig.add_trace(go.Histogram(
                x=clean_data,
                name='Engineered',
                marker_color='rgba(16, 185, 129, 0.6)', # Green, semi-transparent
                xbins=dict(size=(clean_data.max() - clean_data.min())/50 if clean_data.max() != clean_data.min() else 1)
            ))

            fig.update_layout(
                **self.plot_layout,
                barmode='overlay',
                title=f"Distribution Comparison: {col}",
                xaxis_title="Value",
                yaxis_title="Count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            fig.update_traces(opacity=0.75)
            
            plots.append(fig.to_html(full_html=False, include_plotlyjs=False))
            
        return plots

    def render(self) -> str:
        missing_html = self.generate_missing_chart()
        num_plots = self.generate_numeric_comparisons()
        
        cols_diff = self.clean_df.shape[1] - self.orig_df.shape[1]
        cols_diff_str = f"+{cols_diff}" if cols_diff > 0 else str(cols_diff)
        
        template = Template(HTML_TEMPLATE)
        html_content = template.render(
            filename=self.filename,
            original_rows=self.orig_df.shape[0],
            clean_cols=self.clean_df.shape[1],
            cols_diff=cols_diff_str,
            original_missing=self.orig_df.isnull().sum().sum(),
            clean_missing=self.clean_df.isnull().sum().sum(),
            applied_plugins=self.doc._applied_plugins,
            missing_plot_html=missing_html,
            num_plots_html=num_plots
        )
        
        return html_content
