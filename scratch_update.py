import os
import re

files = [
    'docs/license.html',
    'docs/contribute.html'
]

prism_css = """    <!-- Prism CSS -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet" />
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/toolbar/prism-toolbar.min.css" rel="stylesheet" />"""

prism_js = """    <!-- PrismJS Syntax Highlighting & Copy Button -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/toolbar/prism-toolbar.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/copy-to-clipboard/prism-copy-to-clipboard.min.js"></script>"""

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Add Prism CSS to head
    if 'prism-tomorrow' not in content:
        content = content.replace('</head>', prism_css + '\n</head>')
        
    # Add Prism JS before </body>
    if 'prism.min.js' not in content:
        content = content.replace('</body>', prism_js + '\n</body>')
        
    # Replace <pre><code> with <pre><code class="language-*">
    def replacer(match):
        code_content = match.group(1)
        first_line = code_content.strip().split('\n')[0].strip()
        if first_line.startswith('$') or first_line.startswith('pip ') or first_line.startswith('uv '):
            return '<pre><code class="language-bash">' + code_content + '</code></pre>'
        elif first_line.startswith('Permission'):
            return '<pre><code class="language-none">' + code_content + '</code></pre>'
        else:
            return '<pre><code class="language-python">' + code_content + '</code></pre>'
            
    content = re.sub(r'<pre><code>(.*?)</code></pre>', replacer, content, flags=re.DOTALL)
    
    content = content.replace('color: #aaa;', 'color: #555; font-weight: 500;')
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
        
print('Docs updated successfully!')
