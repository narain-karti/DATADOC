(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const docsIndex = [
    ['Overview', 'index.html', 'What DATADOC does and the fastest path to a working artifact.'],
    ['Installation', 'setup.html', 'Install the core package and optional extras.'],
    ['How it works', 'how-it-works.html', 'Understand profiling, planning, fitting, and leakage safety.'],
    ['Pipeline guide', 'pipeline.html', 'Follow a complete train, transform, and evaluation workflow.'],
    ['CLI reference', 'cli.html', 'Command options, outputs, and recovery guidance.'],
    ['Python SDK', 'sdk.html', 'Use DataDocPipeline in notebooks and services.'],
    ['API reference', 'api-docs.html', 'Domain objects, API endpoints, and errors.'],
    ['Web UI', 'ui.html', 'Operate the local dashboard and troubleshoot sessions.'],
    ['Production', 'production.html', 'Artifacts, CI, releases, and open-source maintenance.'],
    ['Contributing', 'contribute.html', 'Add safe transformations and tests.'],
  ];

  function currentPage() {
    const name = window.location.pathname.split('/').pop() || 'index.html';
    return name === '' ? 'index.html' : name;
  }

  function setActiveNavigation() {
    const page = currentPage();
    $$('[data-page-link]').forEach((link) => {
      link.classList.toggle('active', link.getAttribute('href') === page);
    });
    const title = document.title.split('|')[0].trim();
    $$('[data-search-title]').forEach((element) => {
      element.dataset.searchTitle = `${element.textContent} ${title}`;
    });
  }

  function setupMobileMenu() {
    const button = $('[data-menu-toggle]');
    const overlay = $('.mobile-overlay');
    if (!button) return;
    const close = () => {
      document.body.classList.remove('menu-open');
      button.setAttribute('aria-expanded', 'false');
    };
    button.addEventListener('click', () => {
      const open = document.body.classList.toggle('menu-open');
      button.setAttribute('aria-expanded', String(open));
    });
    overlay?.addEventListener('click', close);
    $$('.sidebar a').forEach((link) => link.addEventListener('click', close));
  }

  function escapeHtml(value) {
    return value.replace(/[&<>"']/g, (character) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }[character]));
  }

  function highlightSource(source, language) {
    const keywords = new Set([
      'as', 'async', 'await', 'break', 'case', 'class', 'const', 'continue', 'def', 'default',
      'del', 'elif', 'else', 'export', 'extends', 'finally', 'for', 'from', 'function', 'if',
      'import', 'in', 'interface', 'let', 'new', 'pass', 'return', 'static', 'switch', 'throw',
      'try', 'typeof', 'var', 'while', 'with', 'yield', 'param', 'process', 'begin', 'end',
    ]);
    const constants = new Set(['True', 'False', 'None', 'null', 'undefined', 'true', 'false', 'NaN']);
    const types = new Set(['DataDocPipeline', 'PipelineConfig', 'DataFrame', 'JSON', 'Polars', 'Promise', 'String', 'Number', 'Boolean']);
    const shellLanguages = /terminal|powershell|shell|bash|zsh|git|uv/i.test(language);
    const tokenPattern = /#.*|\/\/[^\n]*|\/\*[\s\S]*?\*\/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\b\d+(?:\.\d+)?\b|--?[A-Za-z][\w-]*|\b[A-Za-z_$][\w$]*\b|[{}[\]();,.:=+\-*/<>!?|&]/gm;
    let output = '';
    let cursor = 0;
    let match;
    while ((match = tokenPattern.exec(source)) !== null) {
      output += escapeHtml(source.slice(cursor, match.index));
      const token = match[0];
      const end = match.index + token.length;
      const before = source[match.index - 1] || '';
      const after = source.slice(end);
      const lineStart = source.lastIndexOf('\n', match.index - 1) + 1;
      const atLineStart = source.slice(lineStart, match.index).trim() === '';
      let tokenClass = '';
      if (token.startsWith('#') || token.startsWith('//') || token.startsWith('/*')) tokenClass = 'tok-comment';
      else if (/^["'`]/.test(token)) tokenClass = /^\s*:/.test(after) ? 'tok-property' : 'tok-string';
      else if (/^\d/.test(token)) tokenClass = 'tok-number';
      else if (/^--?\w/.test(token)) tokenClass = 'tok-flag';
      else if (/^[A-Za-z_$]/.test(token)) {
        if (constants.has(token)) tokenClass = 'tok-constant';
        else if (types.has(token)) tokenClass = 'tok-type';
        else if (keywords.has(token)) tokenClass = 'tok-keyword';
        else if (/^\s*\(/.test(after)) tokenClass = 'tok-function';
        else if (before === '.') tokenClass = 'tok-property';
        else if (shellLanguages && atLineStart) tokenClass = 'tok-command';
      } else if (/^[{}[\]();,.:]$/.test(token)) tokenClass = 'tok-punctuation';
      else tokenClass = 'tok-operator';
      const safeToken = escapeHtml(token);
      output += tokenClass ? `<span class="${tokenClass}">${safeToken}</span>` : safeToken;
      cursor = end;
    }
    return output + escapeHtml(source.slice(cursor));
  }

  function setupSyntaxHighlighting() {
    $$('pre[data-copy] code').forEach((code) => {
      if (code.dataset.highlighted) return;
      const block = code.closest('pre');
      const language = $('.code-caption .lang', block)?.textContent || 'Code';
      code.innerHTML = highlightSource(code.textContent, language);
      code.dataset.highlighted = 'true';
    });
  }

  function setupCopyButtons() {
    $$('pre[data-copy]').forEach((block) => {
      if ($('.copy-button', block)) return;
      const button = document.createElement('button');
      button.className = 'copy-button';
      button.type = 'button';
      button.textContent = 'Copy';
      button.setAttribute('aria-label', 'Copy code to clipboard');
      button.addEventListener('click', async () => {
        const code = $('code', block)?.innerText || block.innerText;
        try {
          await navigator.clipboard.writeText(code.trim());
          button.textContent = 'Copied';
          button.classList.add('copied');
          window.setTimeout(() => {
            button.textContent = 'Copy';
            button.classList.remove('copied');
          }, 1400);
        } catch (error) {
          button.textContent = 'Select';
        }
      });
      block.appendChild(button);
    });
    $('[data-copy-page]')?.addEventListener('click', async (event) => {
      const button = event.currentTarget;
      const text = $('.main-content')?.innerText || document.body.innerText;
      try {
        await navigator.clipboard.writeText(text.trim());
        button.textContent = 'Page copied';
        window.setTimeout(() => { button.textContent = 'Copy page'; }, 1400);
      } catch (error) {
        button.textContent = 'Select page text';
      }
    });
  }

  function setupTabs() {
    $$('.tabs').forEach((tabs) => {
      const buttons = $$('.tab-button', tabs);
      const panels = $$('.tab-panel', tabs);
      buttons.forEach((button) => {
        button.addEventListener('click', () => {
          const name = button.dataset.tab;
          buttons.forEach((item) => item.classList.toggle('active', item === button));
          panels.forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === name));
        });
      });
    });
  }

  function setupSearch() {
    const input = $('[data-search-input]');
    const results = $('.search-results');
    if (!input || !results) return;
    const render = (query) => {
      const normalized = query.trim().toLowerCase();
      if (!normalized) {
        results.classList.remove('open');
        results.innerHTML = '';
        return;
      }
      const matches = docsIndex.filter(([title, href, description]) =>
        `${title} ${href} ${description}`.toLowerCase().includes(normalized),
      );
      results.innerHTML = matches.length
        ? matches.map(([title, href, description]) => `<a class="search-result" href="${href}"><strong>${title}</strong><span>${description}</span></a>`).join('')
        : '<div class="no-results">No matching docs. Try “pipeline”, “errors”, or “Python”.</div>';
      results.classList.add('open');
    };
    input.addEventListener('input', () => render(input.value));
    input.addEventListener('focus', () => { if (input.value) render(input.value); });
    document.addEventListener('keydown', (event) => {
      if (event.key === '/' && document.activeElement !== input) {
        event.preventDefault();
        input.focus();
      }
      if (event.key === 'Escape') {
        results.classList.remove('open');
        input.blur();
      }
    });
    document.addEventListener('click', (event) => {
      if (!results.contains(event.target) && !input.contains(event.target)) results.classList.remove('open');
    });
  }

  function setupSectionObserver() {
    const links = $$('[data-section-link]');
    const headings = links.map((link) => $(link.getAttribute('href'))).filter(Boolean);
    if (!headings.length || !('IntersectionObserver' in window)) return;
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          links.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`));
        }
      });
    }, { rootMargin: '-18% 0px -70% 0px', threshold: 0 });
    headings.forEach((heading) => observer.observe(heading));
  }

  document.addEventListener('DOMContentLoaded', () => {
    setActiveNavigation();
    setupMobileMenu();
    setupSyntaxHighlighting();
    setupCopyButtons();
    setupTabs();
    setupSearch();
    setupSectionObserver();
  });
})();
