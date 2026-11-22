"""Undo the 20 Nov 2026 clean-up: move every item listed in cleanup_moves_20Nov2026.tsv back to its old path. usage: undo_cleanup.py [--apply] (without --apply it only reports)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); apply = "--apply" in sys.argv
rows = [l.split("\t") for l in open(os.path.join(HERE, "cleanup_moves_20Nov2026.tsv"), encoding="utf-8").read().splitlines()[1:]]
todo = [(a, b) for a, b in rows if os.path.exists(os.path.join(ROOT, b)) and not os.path.exists(os.path.join(ROOT, a))]
print(f"{len(rows)} moves logged, {len(todo)} can be undone")
for a, b in todo:
    if apply: os.rename(os.path.join(ROOT, b), os.path.join(ROOT, a))
print("done" if apply else "dry run: nothing moved (use --apply)")
