# Open vs. paid teacher: distilling aspect sentiment for PT-BR reviews on an 8 GB AMD GPU

**Question:** to distill an LLM into a small model, does a free, local, open-weights teacher work as well as a paid API teacher?

I fine-tuned the same student (Qwen3-1.7B + LoRA) twice on 9,000 Brazilian e-commerce reviews (Olist). One run used labels from **Claude Haiku**, a paid API. The other used labels from **Qwen3-8B**, running locally on a consumer **Radeon RX 7600 (8 GB)**. Both students were then evaluated on the same public test set.

The task is aspect sentiment over five aspects: delivery, product, customer service, price and packaging. Each gets one of `positivo` / `negativo` / `neutro` / `ausente` (positive / negative / neutral / absent). The output is JSON that a support system can route on directly.

## Results

Public gold set: 329 reviews. Macro-F1 is computed per aspect over the classes present, then averaged across aspects, using the same metric as the [reference work](https://github.com/renatospessotto/review-intelligence-ptbr). The 95% confidence intervals come from 1,000 bootstrap resamples. "Needs action" means the review has at least one negative aspect.

| Model | Macro-F1 (95% CI) | "Needs action" recall | Reviews/s (Ryzen 5700X + RX 7600) |
|---|---|---|---|
| **Student, Haiku labels** | **0.784** (0.735–0.861) | **98.5%** | 6.6 |
| Student, Qwen3-8B labels | 0.690 (0.645–0.769) | 96.2% | 6.8 |
| TF-IDF + logistic regression, Haiku labels | 0.732 (0.671–0.769) | 94.6% | 14.7 (CPU) |
| Teacher Qwen3-8B (zero-shot, Ollama) | 0.709 (0.676–0.769) | 96.2% | 0.9 (one request at a time); 2.5 (4 in parallel) |
| TF-IDF + logistic regression, Qwen3-8B labels | 0.619 (0.580–0.650) | 95.4% | 17.0 (CPU) |
| Student before fine-tuning (zero-shot) | 0.472 (0.434–0.499) | 91.5% | 3.7 |

Paired bootstrap difference, Haiku student minus Qwen3-8B student: **+0.091** (95% CI +0.007 to +0.160).

Per aspect (macro-F1):

| Model | delivery | product | service | price | packaging |
|---|---|---|---|---|---|
| Student, Haiku labels | 0.70 | 0.81 | 0.83 | 0.97 | 0.61 |
| Student, Qwen3-8B labels | 0.70 | 0.70 | 0.74 | 0.72 | 0.60 |
| Teacher Qwen3-8B | 0.70 | 0.65 | 0.62 | 0.98 | 0.58 |
| TF-IDF, Haiku labels | 0.66 | 0.69 | 0.61 | 0.94 | 0.75 |

## What I found

1. **The student inherits its teacher's ceiling.** The Qwen3-8B student gets 97% of its teacher's F1 (0.690 vs 0.709) and runs about 3× faster. That compares 6.8 reviews/s at batch 16 with the teacher's 2.5/s at 4 parallel requests. With the stronger teacher, the same student reaches 0.784.
2. **The paid teacher won, but by less than it looks.** The +0.09 gap clears zero only barely. The gold set was annotated by Claude Opus plus a second model, so it may favour Claude-family labels. On delivery, the aspect that matters most for a logistics business, the two students tie at 0.70.
3. **A student can beat its teacher where the teacher is inconsistent.** On customer service and product, the Qwen3-8B student scores higher than Qwen3-8B itself (0.74 vs 0.62 and 0.70 vs 0.65). Fine-tuning averages out the teacher's noise. Price goes the other way (0.72 vs 0.98): the gold set has only 16 non-absent price reviews, and with that few examples the student did not learn the class.
4. **For triage, every option is already good.** "Needs action" recall is 96–98% for both students. The aspect-level detail is where they differ.
5. **The fully open pipeline costs no API calls.** Qwen3-8B labelled the 10,000 reviews in about 67 minutes through Ollama (Vulkan), with schema-constrained JSON. Each student trained in 44 minutes, in 4.3 GB of VRAM.

## Running on an 8 GB AMD card

PyTorch's ROCm 7.2 wheels run natively on the RX 7600 (gfx1102), with no `HSA_OVERRIDE_GFX_VERSION`. With plain LoRA, a batch of 4 sequences of 192 tokens ran out of the 8 GB. Three changes in [`src/train_lora.py`](src/train_lora.py) fixed that: batch 8 now peaks at 4.3 GB, and the short prompt alone made training 3.4× faster.

- **Logits only on answer tokens.** Qwen3's 151k-token vocabulary makes full-sequence logits the biggest tensor in the step. The loss runs `lm_head` only on the JSON answer positions.
- **A short prompt for the student.** The teacher needs the full annotation guide, about 400 tokens. Once fine-tuned, the student learns the schema and gets a one-line prompt, about 80 tokens. This cuts both training and inference cost.
- **Padding to multiples of 32.** ROCm picks a kernel for each tensor shape, so fewer distinct shapes means less time spent selecting kernels.

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2   # or the CUDA build
pip install -r requirements.txt
python src/get_data.py                                                   # public gold + Haiku silver
python src/teacher_label.py --model qwen3:8b --out data/silver-qwen3-8b  # needs `ollama serve`
python src/train_lora.py --silver data/aspectos/silver --out runs/aluno-haiku --batch 8 --accum 2
python src/train_lora.py --silver data/silver-qwen3-8b --out runs/aluno-qwen3-8b --batch 8 --accum 2
python src/evaluate.py --name aluno-haiku --adapter runs/aluno-haiku     # repeat for each model
python src/report.py
```

[`notebooks/kaggle_treino.ipynb`](notebooks/kaggle_treino.ipynb) is set up to run the same training on a free Kaggle T4, in fp32 because the T4 has no native bf16.

| File | What it does |
|---|---|
| [`src/aspects.py`](src/aspects.py) | Schema, prompts, JSON parsing and the metric |
| [`src/teacher_label.py`](src/teacher_label.py) | Open teacher via Ollama, resumable |
| [`src/train_lora.py`](src/train_lora.py) | LoRA training, on ROCm or CUDA |
| [`src/evaluate.py`](src/evaluate.py) | Student, teacher and TF-IDF on the gold set |
| [`src/report.py`](src/report.py) | Bootstrap CIs and paired difference |

## Limitations

- **The gold set is labelled by LLMs, not people.** It is the consensus of Claude Opus 5.5 and a second model, and only reviews where both agree on all five aspects are kept (329 of 400). Agreement-only sets skew toward easier reviews.
- **Small set.** Price and packaging have 16 and 13 non-absent gold reviews, so their per-aspect scores are noisy. Read the confidence intervals.
- **One run per arm.** Each student was trained once (seed 0), so I did not measure seed variance.
- **The teacher prompt was not tuned.** I wrote it once from the gold set's annotation guide, before seeing any result, to avoid tuning on the test set.
- **E-commerce, not food delivery.** Olist reviews are about product purchases.

## Data, credits and licenses

- Reviews: [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), CC BY-NC-SA 4.0.
- Gold set and Claude Haiku silver labels: [RenatoSpessotto/olist-aspectos-reviews-ptbr](https://huggingface.co/datasets/RenatoSpessotto/olist-aspectos-reviews-ptbr), CC BY-NC-SA 4.0. This project reuses their data and metric definition to measure one new variable: the teacher.
- Models: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (student) and [Qwen3-8B](https://ollama.com/library/qwen3) (teacher), both Apache 2.0.
- Code in this repository: [MIT](LICENSE). The data and adapters are derived from CC BY-NC-SA 4.0 data and inherit that license: non-commercial, share-alike.

## Resumo em português

Treinei o mesmo modelo pequeno (Qwen3-1.7B com LoRA) duas vezes. Uma vez com rótulos de um professor pago (Claude Haiku), outra com rótulos de um professor aberto rodando numa RX 7600 de 8 GB (Qwen3-8B). A tarefa é classificar o sentimento de 5 aspectos (entrega, produto, atendimento, preço e embalagem) em avaliações da Olist.

- O aluno do professor pago chegou a F1 0,784; o do professor aberto, 0,690.
- O aluno herda o teto do professor: atinge 97% do Qwen3-8B rodando cerca de 3 vezes mais rápido.
- Na entrega, os dois empatam (0,70).
- Os dois detectam mais de 96% das avaliações que pedem ação.

A pipeline aberta não tem custo de API e roda inteira numa placa de vídeo de consumo.
