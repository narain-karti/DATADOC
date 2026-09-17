---
name: brag
description: >
  Creates short, high-impact motion-design product videos and developer launch
  teasers using the latent-spaces/brag and hyperframes workflow. Handles the full
  end-to-end pipeline: codebase analysis, narrative angle & hook discovery,
  composition planning (brag-plan.md, composition-brief.md), HTML/CSS/GSAP scene
  authoring (Swiss typography, Modern Enterprise AI editorial system, modular
  card hierarchies, light/dark mode transitions), contextual 3D visual generation,
  studio-grade neural voiceover generation via edge-tts, beat-synced BGM ducking,
  hyperframes validation and rendering, high-res poster extraction, and frame-0
  poster baking with faststart. Trigger: /brag, "brag", "create video of this project",
  "render a launch teaser", or "hyperframes video".
argument-hint: "[optional: angle, duration, or visual style]"
license: MIT
---

# Brag: Developer Motion Design & Video Teaser Engine

Create flagship, art-directed developer launch and teaser videos directly from codebases using the `latent-spaces/brag` workflow and `hyperframes` runtime.

---

## Slash Command Usage

Trigger anywhere in chat with:
```text
/brag
/brag [duration: 30s] [focus: data leakage]
/brag create a flagship enterprise video for this project
```

---

## The 6-Step Brag Pipeline

```
1. Codebase Extraction → 2. Narrative & Storyboard Contract → 3. Asset & VO Synthesis
       ↓                                                               ↓
6. Final Deliverable Package ← 5. Check, Render & Bake Poster ← 4. Hyperframes Composition
```

---

## Step 1: Codebase Extraction

Before writing visual code, extract the hard technical receipts from the active project:
1. **Name & Tagline**: Read `pyproject.toml`, `package.json`, `README.md`, or docs for the exact core claim.
2. **The Real Pain Point**: Why was this built? (e.g. "Notebooks leak", "CSS specificity hell", "Docker builds taking 20 minutes").
3. **Core Mechanism**: What enforces the solution? (e.g. "Frozen JSON artifact", "Vectorized Polars engine", "Zero-copy Arrow memory").
4. **Verifiable Proof**: Real benchmark numbers (e.g. `10x+ Speed`, `+1.01% Accuracy Delta`, `94/100 Health Score`).
5. **Exact Terminal Commands**: `pip install ...`, `npm i ...`, CLI subcommands.

---

## Step 2: Narrative & Creative Direction Contract

Always write two design documents in the output directory (e.g. `brag-output/`):
- `brag-plan.md`: Storyboard breakdown, audio intent, typography, and card tokens.
- `composition-brief.md`: Technical hyperframes config (dimensions, duration, font family, color codes).

### The Modern Enterprise AI / Editorial System
- **Aesthetic**: Web Design × Enterprise AI × Editorial Magazine × Data Visualization.
- **Palette**:
  - Obsidian Black (`#09090b`): Structural dark scenes, architecture reveals, closing moments.
  - Warm Off-White (`#fcfcfd` / `#fbfbfc`): Problem breakdowns, lifecycle steps.
  - Neon Lime Accent (`#bfff62`): High-energy visual signal for dominant stats, active borders, and highlight capsules.
- **Typography**:
  - Headlines: Geometric Sans (`Plus Jakarta Sans` or `Inter`, 900 ultra-bold, tight line spacing `1.02–1.05`, 60–92px).
  - Highlight Capsules: Rounded neon-lime pills wrapping critical keywords (e.g. `<span class="capsule-lime">leakage-safe</span>`).
  - Code & Metadata: `JetBrains Mono` for commands, paths, and micro-navigation.
- **Modular Card Hierarchy**:
  - *Neon Hero Card*: Dominant metric surface (`92%`, `10x+`) with neon background and dark text.
  - *Dark Card*: Obsidian surface with subtle border for high-value diagnostic findings.
  - *Light Information Card*: Warm white surface with ambient shadow for supporting evidence.
  - *Workflow Cards*: Sequential pipeline cards with live active state tracking.
  - *Micro-Pills*: Tiny uppercase category badges (`CRITICAL FLAW`, `EMPIRICAL BENCHMARK`, `PRODUCTION READY`).

---

## Step 3: Asset Generation & Voiceover Synthesis

### Contextual 3D Visuals
Never use generic stock photography or borrow mismatched UI styles. If a physical or conceptual render is needed, generate original imagery:
- Physical models representing the domain (e.g. acrylic spreadsheet separator, rugged diagnostic tablet, titanium locked cartridge).
- Clean studio lighting, realistic soft ambient drop-shadows, matching color palette.

### Studio-Grade Neural Voiceover
Never rely on missing local dependencies or unconfigured TTS runtimes. Generate clean, high-fidelity neural speech clips via `edge-tts`:
```bash
uvx --from edge-tts edge-tts --voice en-US-AndrewMultilingualNeural --rate=+15% --text "Your voiceover sentence here." --write-media assets/vo_s1.mp3
```
- Generate one discrete `.mp3` per scene.
- Mix background music (`bgm.mp3`) with volume ducked to `0.18` so voiceover speech at `1.0` remains crisp and intelligible.

---

## Step 4: Hyperframes Composition (`index.html`)

Build the composition in `composition/index.html` using GSAP and the hyperframes runtime.

### Critical Hyperframes Rules:
1. **Scene Visibility**:
   - Give each scene `class="scene clip" data-start="X" data-duration="Y"`.
   - **CRITICAL**: Do NOT animate `autoAlpha` or `visibility` directly on `.clip` elements with GSAP; the runtime mounts and unmounts them automatically based on `data-start` and `data-duration`.
   - Animate inner elements instead (e.g. `#s1-inner`, `#s2-inner`).
2. **Layering & Overlaps**:
   - If elements intentionally overlap, sit diagonally, or float over cards, add `data-layout-allow-overlap="true"` to prevent collision warnings during automated checks.
3. **Pacing & Breathing Room**:
   - Standard launch video: **25 to 49 seconds**.
   - Allow at least 1.5s to 2.5s of reading hold time after cards settle before transitioning scenes.
4. **Audio Track Elements**:
   ```html
   <audio id="bgm" class="clip" data-start="0" data-duration="49" data-track-index="0" data-volume="0.18" src="assets/music/bgm.mp3"></audio>
   <audio id="vo-s1" class="clip" data-start="0.3" data-duration="8.6" data-track-index="1" data-volume="1.0" src="assets/vo_s1.mp3"></audio>
   ```

---

## Step 5: Validation, Rendering & Poster Baking

### Automated Check:
Run inside the composition directory:
```bash
npx hyperframes check
```
Must pass with **0 errors** across Runtime, Layout, Motion, Lint, and Contrast.

### Render:
```bash
npx hyperframes render --output ../brag.mp4
```

### High-Res Poster Extraction & Frame-0 Baking:
Extract an iconic frame from a hero scene (e.g. Scene 3 at ~18s–20s):
```bash
ffmpeg -y -ss 19.0 -i brag-output/brag.mp4 -frames:v 1 -q:v 2 brag-output/brag.jpg
```

Bake `brag.jpg` into frame 0 of `brag.mp4` and enable `+faststart` (ensures video feeds and social players display the sharp editorial card instantly):
```powershell
ffmpeg -y -i brag-output/brag.mp4 -i brag-output/brag.jpg -filter_complex "[0:v][1:v]overlay=0:0:enable='eq(n,0)'[v]" -map "[v]" -map 0:a? -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -c:a copy -movflags +faststart brag-output/brag.poster.mp4
Move-Item -Force brag-output/brag.poster.mp4 brag-output/brag.mp4
```

---

## Step 6: Deliverables Checklist

Every completed brag execution must deliver:
1. `brag-output/brag.mp4`: Rendered MP4 (1080p, 30fps, audio mixed, frame 0 poster baked).
2. `brag-output/brag.jpg`: Standalone high-res poster image.
3. `brag-output/brag-plan.md`: Storyboard, typography, and card design contract.
4. `brag-output/composition-brief.md`: Hyperframes technical composition brief.
5. `brag-output/share-copy.txt`: Punchy launch copy for social and developer platforms.
