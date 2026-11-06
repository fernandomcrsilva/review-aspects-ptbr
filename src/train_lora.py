"""LoRA do aluno (Qwen3-1.7B) num conjunto prata. Roda numa RX 7600 de 8 GB (ROCm) ou numa T4 do Kaggle.

uso: python src/train_lora.py --silver data/aspectos/silver --out runs/aluno-haiku
"""
import argparse
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from aspects import read_jsonl, short_prompt, target

ap = argparse.ArgumentParser()
ap.add_argument("--silver", required=True, help="pasta com train.jsonl e validation.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--limit", type=int, default=0, help="usa só N exemplos (teste rápido)")
args = ap.parse_args()

torch.manual_seed(0)
random.seed(0)
dev = "cuda"
# bf16 na RX 7600 (ROCm) e em GPUs Ampere+; fp32 na T4 do Kaggle, que não tem bf16 nativo
dtype = torch.bfloat16 if torch.version.hip or torch.cuda.get_device_capability()[0] >= 8 else torch.float32

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
base = model.get_base_model()


def encode(row):
    p = tok.apply_chat_template([{"role": "user", "content": short_prompt(row["text"])}],
                                add_generation_prompt=True, enable_thinking=False, tokenize=False)
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    c_ids = tok(target(row) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p_ids + c_ids, len(p_ids)


def batches(rows, size, shuffle):
    enc = [encode(r) for r in rows]
    order = list(range(len(enc)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [enc[j] for j in order[i:i + size]]
        n = -(-max(len(ids) for ids, _ in chunk) // 32) * 32  # poucos formatos distintos: o ROCm escolhe kernel por formato
        ids = torch.full((len(chunk), n), tok.pad_token_id)
        mask = torch.zeros((len(chunk), n), dtype=torch.long)
        labels = torch.full((len(chunk), n), -100)
        for k, (x, p) in enumerate(chunk):
            ids[k, :len(x)] = torch.tensor(x)
            mask[k, :len(x)] = 1
            labels[k, p:len(x)] = torch.tensor(x[p:])
        yield ids.to(dev), mask.to(dev), labels.to(dev)


def loss_fn(ids, mask, labels):
    # Logits só nas posições da resposta: o vocabulário de 151k faria os logits do prompt inteiro
    # estourarem os 8 GB. É isso que permite batch 4 na RX 7600.
    h = base.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
    y = labels[:, 1:]
    keep = y != -100
    logits = base.lm_head(h[keep]).float()
    return torch.nn.functional.cross_entropy(logits, y[keep])


train = read_jsonl(f"{args.silver}/train.jsonl")
val = read_jsonl(f"{args.silver}/validation.jsonl")[:300]
if args.limit:
    train = train[:args.limit]
steps = math.ceil(len(train) / args.batch * args.epochs / args.accum)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))


@torch.no_grad()
def val_loss():
    model.eval()
    ls = [loss_fn(*b).item() for b in batches(val, 8, False)]
    model.train()
    return sum(ls) / len(ls)


print(f"treino {len(train)} | validação {len(val)} | {steps} passos de otimização")
model.train()
step, micro, t0 = 0, 0, time.time()
while step < steps:
    for b in batches(train, args.batch, True):
        (loss_fn(*b) / args.accum).backward()
        micro += 1
        if micro % args.accum:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        if step % 50 == 0 or step == steps:
            print(f"passo {step}/{steps} | val {val_loss():.4f} | {time.time() - t0:.0f}s", flush=True)
        if step >= steps:
            break

model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print(f"adaptador salvo em {args.out} | {time.time() - t0:.0f}s | pico VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
