---
name: linkedin
description: >
  Drafts authentic, human-sounding LinkedIn posts and video captions for developers
  and technical founders that maximize organic reach and bypass LinkedIn's 2026 AI
  slop detection and algorithmic downranking. Incorporates the sergebulaev/linkedin-skills
  system: odd-precision number-first hooks (+34% median likes) fitting within the
  mobile 140-character fold, ruthless scrubbing of AI vocabulary (delve, tapestry,
  leverage, robust, seamless, streamline, elevate, game-changer, thrilled to announce),
  elimination of reveal bridges ("The result?", "Here's why:"), em-dash budgeting (max
  1 per 100 words), zero external links in the post body (+2.1x reach via first-comment
  isolation), and specific experience-anchored closing questions (+20-40% comment
  threading). Trigger: /linkedin, /linkedin-post, "write a linkedin post", "humanize a
  linkedin draft", "create a linkedin caption for this video", or "audit my post for ai tells".
argument-hint: "[optional: topic, draft to audit, or launch angle]"
license: MIT
---

# LinkedIn Post Writer & Humanizer (2026 Edition)

Draft and audit high-converting, human-sounding LinkedIn posts and video captions designed for engineers, open-source maintainers, and technical founders. Built on stylometric research and verified 2026 feed heuristics from `sergebulaev/linkedin-skills`.

---

## Slash Command Usage

Trigger anywhere in chat with:
```text
/linkedin
/linkedin [paste draft to audit or humanize]
/linkedin write a launch post for my new pypi package
/linkedin caption for the brag video
```

---

## Core Algorithm Rules (2026 Heuristics)

1. **The Mobile Cutoff (140 Characters)**:
   - On mobile screens, LinkedIn truncates posts after approximately 140 characters with `…see more`.
   - The hook **must** deliver its thesis or odd-precision number before line 2 wraps.
2. **Number-First Opener (+34% Median Likes)**:
   - Begin with an odd-precision number paired with a concrete referent (`23 people installed... in the last 48 hours`, `$4,730 in overages`).
   - **Never open with a question** (-34% median likes penalty). Move all questions to the close.
3. **External Link Suppression (The First Comment Rule)**:
   - Outbound links in the post body trigger a **40% to 60% reach penalty**.
   - Always keep the post body link-free. Direct readers to the **First Comment** (`+2.1x impression lift`).
4. **Length Sweet Spot**:
   - Short/Medium: **900–1,300 characters** (~150–220 words).
   - Long-form: **1,500–1,900 characters** with 1–2 sentence paragraphs and blank lines between them.
5. **Close With a Topic-Specific Question (+20–40% Response Rate)**:
   - Never use generic closers like "Thoughts?" or "What do you think?".
   - Ask an experience-anchored technical question that prompts peer engineers to share their setup.
6. **The P.S. Multiplier (+7.5% Reach)**:
   - Add a one-line P.S. before the hashtags to provide context (e.g. video explanation or link location).

---

## The AI-Tell Blacklist (Automatic Demotion Flags)

LinkedIn’s AI slop classifiers and human readers react negatively to common LLM patterns. Always scrub these:

### 1. Banned Vocabulary & Corporate Slop
- **Frontier LLM Tells**: `delve`, `tapestry`, `realm`, `journey`, `paradigm`, `beacon`, `testament`.
- **Corporate Buzzwords**: `leverage`, `streamline`, `robust`, `seamless`, `seamlessly`, `elevate`, `empower`, `unlock`, `harness`, `foster`, `landscape`, `ecosystem`, `nuanced`, `multifaceted`, `holistic`.
- **Cliche Fillers**: `game-changer`, `deep dive`, `needle-moving`, `at the end of the day`, `in today's fast-paced world`.
- **Announcement PR Cringe**: `thrilled to announce`, `excited to share`, `humbled and honored`, `proud to present`.

### 2. Banned Reveal Bridges & Fake Drama
- Never use: `"The result?"`, `"The kicker?"`, `"Here's what I learned:"`, `"Here's the thing:"`, `"Let that sink in."`, `"Read that again."`.

### 3. Structural Tells
- **No Staccato Fragment Stacks**: Avoid fake dramatic 1-word or 2-word sentence sequences (*"Clean data. Faster models. Zero leaks. Built different."*).
- **Em-Dash Density**: Max 1 em-dash (`—`) per 100 words (1–2 total in a full post). Use commas, colons, or soft pauses (`..`).
- **No Fake Vulnerability Framing**: State uncomfortable facts flatly without setup sentences (say *"I expected 10 users, we got 2"* instead of *"Let me be completely honest and vulnerable for a second:..."*).
- **Emoji Spam**: Max 0 to 1 natural emoji. Never use `🚀`, `💡`, `🔥`, `✨`, or `👇`.

---

## Post Structure Template

```text
[Odd-precision number or flat concrete fact in first 140 chars]

[1-2 sentences setting the scene or reaction — feeling surreal / unexpected]

[The core technical problem or scar — why you built this / what broke in notebooks]

[The common failure mode — e.g. cross-validation scores 85%, production tanks]

[The concrete technical mechanism — Polars, Arrow, immutable JSON artifact, zero bleed]

[What the tool does locally — terminal audit, health score, local dashboard]

[Specific experience-anchored discussion question for peer engineers]

P.S. [Context about video or asset above]. [Point to First Comment for links].

#NicheTag1 #NicheTag2
```

---

## First Comment Template

Always draft the companion First Comment alongside the post:

```text
Links to inspect the project:

• PyPI: pip install <package-name>
• GitHub: <repository-url>
• Docs: <documentation-url>

If you run into any edge cases on your datasets or have ideas for plugins, issues and PRs are open!
```

---

## Pre-Publish Verification Checklist

Before presenting the draft to the user, run this audit:
- [ ] Hook is within the first 140 characters and opens with a statement or number.
- [ ] First line is NOT a question.
- [ ] 0 words from the AI blacklist (*delve, leverage, robust, seamless, streamline, etc.*).
- [ ] 0 reveal bridges (*"The result?", "Here's why:"*).
- [ ] No staccato 1-word fragment stacks.
- [ ] Em dashes: 1 or fewer per 100 words.
- [ ] 0 external links in the post body (moved to First Comment).
- [ ] Paragraphs are 1–2 sentences with double line breaks for mobile formatting.
- [ ] Closes with an open, experience-anchored technical question.
- [ ] Includes a 1-line P.S. (+7.5% reach lift).
- [ ] 0–2 niche hashtags at the very end.
