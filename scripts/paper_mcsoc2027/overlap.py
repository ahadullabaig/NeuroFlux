"""
Text overlap between the MCSoC 2027 and ICEE 2026 manuscripts.

IEEE screens submissions for text similarity, and the MCSoC paper must reuse no
sentence of the ICEE paper. This script extracts the prose of both `main.tex`
files, dropping the preamble, author blocks, keyword lists, TikZ code, display
math and citation keys, and reports:

  * every run of at least `MIN_RUN` consecutive words that appears in both;
  * the share of MCSoC words that fall inside such runs;
  * MCSoC sentences whose word trigrams overlap an ICEE sentence's by at least
    `SENTENCE_JACCARD`, a looser check for paraphrase that kept most words;
  * figure files the two papers share, by content hash.

Usage:
    uv run python scripts/paper_mcsoc2027/overlap.py
Exit status is non-zero if a shared run or a close sentence is found.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MCSOC = ROOT / "paper" / "mcsoc2027"
ICEE = ROOT / "paper" / "icee2026"
MIN_RUN = 6
SENTENCE_JACCARD = 0.3


def prose(path: Path) -> str:
    tex = path.read_text(encoding="utf-8")
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = tex[tex.index("\\begin{abstract}") : tex.index("\\bibliographystyle")]
    envs = (
        "IEEEkeywords",
        "tikzpicture",
        "equation",
        "align",
        "figure",
        "figure*",
        "table",
    )
    for env in envs:
        pattern = rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}"
        if env in ("figure", "figure*", "table"):
            # keep captions, which are prose a similarity check reads too
            tex = re.sub(
                pattern,
                lambda m: " ".join(
                    re.findall(r"\\caption\{(.*?)\}\s*\\label", m.group(0), re.S)
                ),
                tex,
                flags=re.S,
            )
        else:
            tex = re.sub(pattern, " ", tex, flags=re.S)
    tex = re.sub(r"\$[^$]*\$", " ", tex)
    tex = re.sub(
        r"\\(cite|ref|eqref|label|input|includegraphics)(\[[^]]*\])?\{[^}]*\}", " ", tex
    )
    tex = re.sub(r"\\[a-zA-Z]+\*?", " ", tex)
    tex = tex.replace("{", " ").replace("}", " ").replace("~", " ")
    return re.sub(r"\s+", " ", tex)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def shared_runs(ours: list[str], theirs: list[str]) -> list[tuple[int, int]]:
    grams = {tuple(theirs[i : i + MIN_RUN]) for i in range(len(theirs) - MIN_RUN + 1)}
    covered = [False] * len(ours)
    for i in range(len(ours) - MIN_RUN + 1):
        if tuple(ours[i : i + MIN_RUN]) in grams:
            for j in range(i, i + MIN_RUN):
                covered[j] = True
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate([*covered, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    return runs


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;:?])\s+", text) if len(words(s)) >= 6]


def trigrams(sentence: str) -> set[tuple[str, ...]]:
    w = words(sentence)
    return {tuple(w[i : i + 3]) for i in range(len(w) - 2)}


def main() -> None:
    ours_text, theirs_text = prose(MCSOC / "main.tex"), prose(ICEE / "main.tex")
    ours, theirs = words(ours_text), words(theirs_text)
    runs = shared_runs(ours, theirs)
    shared_words = sum(end - start for start, end in runs)
    print(
        f"MCSoC prose: {len(ours)} words; ICEE prose: {len(theirs)} words.\n"
        f"Runs of >= {MIN_RUN} words shared with ICEE: {len(runs)} "
        f"({shared_words} words, {100 * shared_words / len(ours):.1f}% of MCSoC)"
    )
    for start, end in runs:
        print(f"  [{end - start} words] {' '.join(ours[start:end])}")

    close: list[tuple[float, str, str]] = []
    theirs_sentences = [(s, trigrams(s)) for s in sentences(theirs_text)]
    for sentence in sentences(ours_text):
        mine = trigrams(sentence)
        if not mine:
            continue
        best, match = max(
            ((len(mine & t) / len(mine | t), s) for s, t in theirs_sentences if t),
            default=(0.0, ""),
        )
        if best >= SENTENCE_JACCARD:
            close.append((best, sentence, match))
    print(f"\nSentences with trigram overlap >= {SENTENCE_JACCARD}: {len(close)}")
    for score, sentence, match in sorted(close, reverse=True):
        print(
            f"  {score:.2f}\n    MCSoC: {sentence.strip()}\n    ICEE:  {match.strip()}"
        )

    def hashes(folder: Path) -> dict[str, str]:
        return {
            hashlib.sha256(p.read_bytes()).hexdigest(): p.name
            for p in (folder / "figures").glob("*")
            if p.is_file()
        }

    ours_figs, theirs_figs = hashes(MCSOC), hashes(ICEE)
    common = [ours_figs[h] for h in ours_figs.keys() & theirs_figs.keys()]
    print(f"\nFigure files shared with ICEE: {len(common)} {common}")
    sys.exit(1 if runs or close or common else 0)


if __name__ == "__main__":
    main()
