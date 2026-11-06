"""Baixa o conjunto público (prata do Claude Haiku e ouro de consenso) para data/aspectos. Licença CC BY-NC-SA 4.0."""
import urllib.request

from aspects import DATA

BASE = "https://huggingface.co/datasets/RenatoSpessotto/olist-aspectos-reviews-ptbr/resolve/main/"
for f in ["silver/train.jsonl", "silver/validation.jsonl", "gold/test.jsonl"]:
    dest = DATA / "aspectos" / f
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(BASE + f, dest)
    print(dest)
