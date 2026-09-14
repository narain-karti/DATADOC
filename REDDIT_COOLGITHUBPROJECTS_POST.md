<!--
===================================================================
REDDIT R/COOLGITHUBPROJECTS SUBMISSION FILE
===================================================================

COMMUNITY: r/coolgithubprojects (~65k members, no karma requirement,
explicitly dedicated to sharing GitHub repos — new accounts welcome)
TYPE: Link Post
LINK TO SUBMIT: https://github.com/narain-karti/DATADOC
FLAIR: auto-assigned by language (Python)

TITLE TO COPY:
DATADOC – Fast, zero-leakage tabular ML dataset preparation built on Polars

(No body needed — this community expects the GitHub link as the post.)
===================================================================
-->

Posting guidelines of the community (verified Sep 2026):
- GitHub links only — submit the repository URL directly.
- Keep the title in the "Name – short description" convention.
- Flair (Python) is auto-assigned.

One-liner to drop in the comments right after submitting (first-comment
protocol — keeps the thread alive and answers "what is it?"):

    Built this because ad-hoc pandas preprocessing keeps leaking test-set
    statistics into training (medians/vocab computed before the split).
    DATADOC fits transforms strictly on the train split, freezes them into
    an auditable pipeline.json, and runs on Polars (Apache Arrow).
    Install: pip install datadoc-cli · Docs: https://narain-karti.github.io/DATADOC/
    Happy to answer questions or take PRs!
