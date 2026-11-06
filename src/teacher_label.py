"""Professor aberto: rotula o conjunto prata com um LLM local via Ollama (saída restrita ao esquema JSON).

uso: python src/teacher_label.py --model qwen3:8b --out data/silver-qwen3-8b
Lê os mesmos textos do prata do Claude Haiku, para que a única diferença entre os dois alunos seja o professor.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from aspects import DATA, JSON_SCHEMA, parse, prompt, read_jsonl, write_jsonl

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def label(text: str, model: str) -> str:
    r = requests.post(f"{OLLAMA}/api/chat", timeout=300, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt(text)}],
        "format": JSON_SCHEMA,
        "think": False,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 2048},
    })
    r.raise_for_status()
    return r.json()["message"]["content"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    for split in ["validation", "train"]:
        rows = read_jsonl(DATA / f"aspectos/silver/{split}.jsonl")
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as ex:
            outs = list(ex.map(lambda r: parse(label(r["text"], args.model)), rows))
        kept = [{"review_id": r["review_id"], "text": r["text"], "review_score": r["review_score"], **o}
                for r, o in zip(rows, outs) if o]
        write_jsonl(f"{args.out}/{split}.jsonl", kept)
        print(f"{split}: {len(kept)}/{len(rows)} rotulados em {time.time() - t0:.0f}s", flush=True)
