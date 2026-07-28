import re

with open('docs/how-it-works.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace mermaid div
old_mermaid_div = r'<!-- Mermaid Architecture Diagram -->.*?</div>'
new_mermaid_div = '''<!-- Mermaid Architecture Diagram -->
            <div class="brutal-card dark" style="margin-top: 2rem; padding: 2rem; overflow-x: auto;">
                <pre class="mermaid" style="text-align: center; background: transparent;">
flowchart TD
    %% Brutalist Styling
    classDef default fill:#ccff00,stroke:#000,stroke-width:3px,color:#000,font-weight:bold,font-size:14px,rx:0,ry:0;
    classDef darkNode fill:#000,stroke:#ccff00,stroke-width:3px,color:#ccff00,font-weight:bold,font-size:14px,rx:0,ry:0;
    classDef accent fill:#5a32fa,stroke:#000,stroke-width:3px,color:#fff,font-weight:bold,font-size:14px,rx:0,ry:0;
    classDef subtle fill:#111,stroke:#444,stroke-width:2px,color:#ddd,rx:0,ry:0;

    A["📄 RAW DATASET CSV"]:::accent --> B["⚙️ DATADOC CORE ENGINE"]:::darkNode
    
    subgraph RoleDetection ["1. COLUMN ROLE DETECTION SYSTEM"]
        direction TB
        R1["_detect_column_roles()"]:::default
        R2["Filter out ID / Name / UUID"]:::subtle
        R1 --> R2
    end
    
    B --> RoleDetection
    
    subgraph PluginPipeline ["2. PRIORITY-ORDERED PLUGIN EXECUTION"]
        direction TB
        P1["10: MissingValuePlugin<br/>(Imputes nulls: Median/Mode)"]:::default
        P2["20: OutlierPlugin<br/>(Caps values at 1.5x IQR)"]:::default
        P3["30: DatetimePlugin<br/>(Extracts year/month/day/hour)"]:::default
        P4["40: CategoricalEncoderPlugin<br/>(One-hot encodes strings)"]:::default
        P5["45: ScalingPlugin<br/>(Z-Score Normalization)"]:::default
        
        P1 --> P2 --> P3 --> P4 --> P5
    end
    
    RoleDetection --> PluginPipeline
    
    subgraph Cleanup ["3. POST-PROCESSING"]
        C1["Drop zero-variance constant features"]:::subtle
    end
    
    PluginPipeline --> Cleanup
    
    Cleanup --> Output{"OUTPUT HANDLERS"}:::accent
    Output -->|"doc.engineer()"| D["📊 Polars / Pandas DataFrame"]:::darkNode
    Output -->|"datadoc engineer"| E["💾 Engineered CSV File"]:::darkNode
    Output -->|"datadoc pipeline"| F["🐍 Reproducible Python Script"]:::darkNode
                </pre>
            </div>'''

content = re.sub(old_mermaid_div, new_mermaid_div, content, flags=re.DOTALL)

# Replace mermaid script
old_script = r'<script type="module">\s*import mermaid.*?mermaid\.initialize.*?\s*</script>'
new_script = '''<script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({ 
          startOnLoad: true, 
          theme: 'base',
          themeVariables: {
              primaryColor: '#ccff00',
              primaryTextColor: '#000',
              primaryBorderColor: '#000',
              lineColor: '#5af78e',
              secondaryColor: '#1a1a1a',
              tertiaryColor: '#1a1a1a',
              fontFamily: 'monospace'
          },
          flowchart: {
              curve: 'step'
          }
      });
    </script>'''

content = re.sub(old_script, new_script, content, flags=re.DOTALL)

with open('docs/how-it-works.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('how-it-works.html updated!')
