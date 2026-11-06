"""Avalia um rotulador no conjunto ouro e grava results/<nome>.json.

uso:
  python src/evaluate.py --name aluno-haiku --adapter runs/aluno-haiku      # aluno ajustado (prompt curto)
  python src/evaluate.py --name aluno-zero-shot                              # Qwen3-1.7B sem ajuste (prompt completo)
  python src/evaluate.py --name professor-qwen3-8b --ollama qwen3:8b         # professor via Ollama
  python src/evaluate.py --name tfidf-haiku --tfidf data/aspectos/silver     # baseline clássico
"""
import argparse
import json
import time
from pathlib import Path

from aspects import ASPECTS, DATA, aspect_f1, parse, prompt, read_jsonl, short_prompt, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--gold", default=str(DATA / "aspectos/gold/test.jsonl"))
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--adapter")
ap.add_argument("--ollama")
ap.add_argument("--tfidf")
args = ap.parse_args()

gold = read_jsonl(args.gold)
texts = [g["text"] for g in gold]
t0 = time.time()

if args.tfidf:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    train = read_jsonl(f"{args.tfidf}/train.jsonl")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    X = vec.fit_transform([r["text"] for r in train])
    t0 = time.time()
    cols = {a: LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, [r[a] for r in train])
            .predict(vec.transform(texts)) for a in ASPECTS}
    raw = [None] * len(texts)
    preds = [{a: cols[a][i] for a in ASPECTS} for i in range(len(texts))]
elif args.ollama:
    from teacher_label import label
    raw = [label(t, args.ollama) for t in texts]
    preds = [parse(r) for r in raw]
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model, padding_side="left")
    dtype = torch.bfloat16 if torch.version.hip or torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to("cuda").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    make = short_prompt if args.adapter else prompt
    chats = [tok.apply_chat_template([{"role": "user", "content": make(t)}], add_generation_prompt=True,
                                     enable_thinking=False, tokenize=False) for t in texts]
    raw = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(chats), 16):
            enc = tok(chats[i:i + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            out = model.generate(**enc, max_new_tokens=64, do_sample=False)
            raw += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    preds = [parse(r) for r in raw]

secs = time.time() - t0
metrics = aspect_f1(gold, preds)
result = {
    "name": args.name,
    "n": len(gold),
    "json_valido": sum(p is not None for p in preds) / len(preds),
    "reviews_por_segundo": len(gold) / secs,
    **metrics,
}
Path("results").mkdir(exist_ok=True)
Path(f"results/{args.name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
write_jsonl(f"runs/preds/{args.name}.jsonl",
            [{"review_id": g["review_id"], "raw": r, "pred": p} for g, r, p in zip(gold, raw, preds)])
print(json.dumps(result, indent=2, ensure_ascii=False))
