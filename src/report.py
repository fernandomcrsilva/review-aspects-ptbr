"""Tabela final com intervalo de confiança (bootstrap) e a diferença pareada entre os dois alunos.

O ouro tem 329 reviews e poucos casos não-ausentes em preço e embalagem, então diferenças pequenas
de F1 podem ser ruído. O bootstrap pareado diz se a distância entre dois modelos se sustenta.
uso: python src/report.py > results/tabela.md
"""
import json
from pathlib import Path

import numpy as np

from aspects import DATA, aspect_f1, read_jsonl

ORDER = ["aluno-haiku", "aluno-qwen3-8b", "tfidf-haiku", "professor-qwen3-8b", "tfidf-qwen3-8b", "aluno-zero-shot"]
gold = read_jsonl(DATA / "aspectos/gold/test.jsonl")
preds = {n: [r["pred"] for r in read_jsonl(f"runs/preds/{n}.jsonl")] for n in ORDER if Path(f"runs/preds/{n}.jsonl").exists()}
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(gold), len(gold)) for _ in range(1000)]


def macro(name, idx):
    return aspect_f1([gold[i] for i in idx], [preds[name][i] for i in idx])["macro"]


print("| Modelo | F1 macro (IC 95%) | Recall \"pede ação\" | Reviews/s (Ryzen 5700X + RX 7600) |")
print("|---|---|---|---|")
for n in preds:
    r = json.loads(Path(f"results/{n}.json").read_text())
    lo, hi = np.percentile([macro(n, b) for b in boots], [2.5, 97.5])
    print(f"| {n} | {r['macro']:.3f} ({lo:.3f}–{hi:.3f}) | {r['acao_recall']:.1%} | {r['reviews_por_segundo']:.1f} |")

if {"aluno-haiku", "aluno-qwen3-8b"} <= preds.keys():
    d = [macro("aluno-haiku", b) - macro("aluno-qwen3-8b", b) for b in boots]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\nDiferença pareada aluno-haiku − aluno-qwen3-8b: {np.mean(d):+.3f} (IC 95% {lo:+.3f} a {hi:+.3f})")
