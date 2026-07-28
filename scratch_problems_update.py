import re

with open('docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

problems_html = '''
        <!-- Real-World Problems Solved -->
        <section class="brutal-card card-problems" style="background: var(--purple); color: var(--white); margin-bottom: 2rem;">
            <div class="card-header" style="color: var(--white); border-bottom-color: var(--white);">
                <div class="icons">
                    <span style="background: var(--white);"></span>
                    <span style="background: var(--white);"></span>
                    <span style="background: var(--white);"></span>
                </div>
                REAL-WORLD PROBLEMS WE SOLVE
                <span style="font-family: monospace;">&#x21F1;</span>
            </div>
            
            <div class="problems-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-top: 1.5rem;">
                
                <div class="problem-box" style="background: var(--black); padding: 1.5rem; border: var(--border-thick);">
                    <h4 style="color: var(--neon); margin-bottom: 0.5rem; font-family: monospace; font-size: 1.1rem;">[FAIL] THE "ID SCALING" DISASTER</h4>
                    <p style="font-size: 0.9rem; color: #ccc; margin-bottom: 1rem;">
                        <strong>Problem:</strong> Standard auto-scalers blindly apply Z-Scores to <code>user_id</code> and <code>transaction_id</code> columns, completely ruining predictive integrity.
                    </p>
                    <p style="font-size: 0.9rem; color: #fff;">
                        <strong>DATADOC Fix:</strong> Native <em>Column Role Detection</em> auto-flags and isolates sequence integers, UUIDs, and names before any plugins even run.
                    </p>
                </div>

                <div class="problem-box" style="background: var(--black); padding: 1.5rem; border: var(--border-thick);">
                    <h4 style="color: var(--neon); margin-bottom: 0.5rem; font-family: monospace; font-size: 1.1rem;">[FAIL] BLACK-BOX AUTO-ML</h4>
                    <p style="font-size: 0.9rem; color: #ccc; margin-bottom: 1rem;">
                        <strong>Problem:</strong> You use an AutoML tool, it works, but you have absolutely no idea what it did to your data under the hood. Impossible to audit.
                    </p>
                    <p style="font-size: 0.9rem; color: #fff;">
                        <strong>DATADOC Fix:</strong> 100% deterministic math. Run <code>datadoc pipeline</code> and it instantly generates a fully readable, reproducible Python script of the exact transforms.
                    </p>
                </div>

                <div class="problem-box" style="background: var(--black); padding: 1.5rem; border: var(--border-thick);">
                    <h4 style="color: var(--neon); margin-bottom: 0.5rem; font-family: monospace; font-size: 1.1rem;">[FAIL] PANDAS OUT-OF-MEMORY</h4>
                    <p style="font-size: 0.9rem; color: #ccc; margin-bottom: 1rem;">
                        <strong>Problem:</strong> Your dataset is large. You try to One-Hot Encode high cardinality data and Pandas instantly crashes with an OOM error.
                    </p>
                    <p style="font-size: 0.9rem; color: #fff;">
                        <strong>DATADOC Fix:</strong> Built purely on the <em>Polars Rust Backend</em>. Executes parallel data mutations on all CPU cores with minimal RAM overhead.
                    </p>
                </div>

                <div class="problem-box" style="background: var(--black); padding: 1.5rem; border: var(--border-thick);">
                    <h4 style="color: var(--neon); margin-bottom: 0.5rem; font-family: monospace; font-size: 1.1rem;">[FAIL] THE 80% TIME SINK</h4>
                    <p style="font-size: 0.9rem; color: #ccc; margin-bottom: 1rem;">
                        <strong>Problem:</strong> Data scientists spend 80% of their time writing the exact same median-imputation and IQR-capping boilerplate code for every project.
                    </p>
                    <p style="font-size: 0.9rem; color: #fff;">
                        <strong>DATADOC Fix:</strong> <code>doc.engineer()</code> runs the industry-standard baseline engineering pipeline instantly. You spend your time on actual model architecture.
                    </p>
                </div>

            </div>
        </section>
'''

# Find the spot before <!-- Top CLI Commands -->
old_str = '        <!-- Top CLI Commands -->'
new_str = problems_html + '\n' + old_str

if old_str in content and 'card-problems' not in content:
    content = content.replace(old_str, new_str)
    with open('docs/index.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print('index.html updated successfully!')
else:
    print('Could not find insertion point or already inserted.')
