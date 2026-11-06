# Professor aberto ou pago? Destilando sentimento por aspecto de avaliações em PT-BR numa GPU AMD de 8 GB

**Pergunta:** para destilar um LLM num modelo pequeno, um professor aberto, local e gratuito funciona tão bem quanto um professor pago via API?

Treinei o mesmo aluno (Qwen3-1.7B com LoRA) duas vezes, em 9.000 avaliações de compras online da Olist. Uma vez com rótulos do **Claude Haiku**, uma API paga. Outra com rótulos do **Qwen3-8B**, rodando localmente numa placa de vídeo de consumo, uma **Radeon RX 7600 de 8 GB**. Os dois alunos foram avaliados no mesmo conjunto de teste público.

**No Hugging Face:** [aluno do professor Haiku](https://huggingface.co/FMendonca/qwen3-1.7b-aspectos-olist-haiku), [aluno do professor Qwen3-8B](https://huggingface.co/FMendonca/qwen3-1.7b-aspectos-olist-qwen3-8b) e os [10 mil rótulos do Qwen3-8B](https://huggingface.co/datasets/FMendonca/olist-aspectos-silver-qwen3-8b).

A tarefa é classificar o sentimento de cinco aspectos: entrega, produto, atendimento, preço e embalagem. Cada aspecto recebe `positivo`, `negativo`, `neutro` ou `ausente`. A saída é um JSON que um sistema de atendimento pode usar direto para encaminhar o caso.

## Resultados

O conjunto ouro público tem 329 avaliações. O F1-macro é calculado por aspecto, sobre as classes presentes, e depois tirada a média entre os aspectos. É a mesma métrica do [trabalho de referência](https://github.com/renatospessotto/review-intelligence-ptbr). Os intervalos de confiança de 95% vêm de 1.000 reamostragens por bootstrap. "Pede ação" significa que a avaliação tem pelo menos um aspecto negativo.

| Modelo | F1-macro (IC 95%) | Recall "pede ação" | Avaliações/s (Ryzen 5700X + RX 7600) |
|---|---|---|---|
| **Aluno, rótulos do Haiku** | **0,784** (0,735–0,861) | **98,5%** | 6,6 |
| Aluno, rótulos do Qwen3-8B | 0,690 (0,645–0,769) | 96,2% | 6,8 |
| TF-IDF + regressão logística, rótulos do Haiku | 0,732 (0,671–0,769) | 94,6% | 14,7 (CPU) |
| Professor Qwen3-8B (zero-shot, Ollama) | 0,709 (0,676–0,769) | 96,2% | 0,9 (uma requisição por vez); 2,5 (4 em paralelo) |
| TF-IDF + regressão logística, rótulos do Qwen3-8B | 0,619 (0,580–0,650) | 95,4% | 17,0 (CPU) |
| Aluno antes do ajuste (zero-shot) | 0,472 (0,434–0,499) | 91,5% | 3,7 |

Diferença pareada por bootstrap, aluno do Haiku menos aluno do Qwen3-8B: **+0,091** (IC 95% de +0,007 a +0,160).

Por aspecto (F1-macro):

| Modelo | entrega | produto | atendimento | preço | embalagem |
|---|---|---|---|---|---|
| Aluno, rótulos do Haiku | 0,70 | 0,81 | 0,83 | 0,97 | 0,61 |
| Aluno, rótulos do Qwen3-8B | 0,70 | 0,70 | 0,74 | 0,72 | 0,60 |
| Professor Qwen3-8B | 0,70 | 0,65 | 0,62 | 0,98 | 0,58 |
| TF-IDF, rótulos do Haiku | 0,66 | 0,69 | 0,61 | 0,94 | 0,75 |

## O que encontrei

1. **O aluno herda o teto do professor.** O aluno do Qwen3-8B chega a 97% do F1 do professor (0,690 contra 0,709) e roda cerca de 3 vezes mais rápido: 6,8 avaliações/s com batch 16, contra 2,5/s do professor com 4 requisições em paralelo. Com o professor mais forte, o mesmo aluno chega a 0,784.
2. **O professor pago venceu, mas por menos do que parece.** A diferença de +0,09 mal passa de zero no intervalo de confiança. O conjunto ouro foi anotado pelo Claude Opus e por um segundo modelo, então pode favorecer rótulos da família Claude. Na entrega, o aspecto que mais importa para um negócio de logística, os dois alunos empatam em 0,70.
3. **O aluno pode superar o professor onde o professor é inconsistente.** Em atendimento e produto, o aluno do Qwen3-8B tira nota maior que o próprio Qwen3-8B (0,74 contra 0,62 e 0,70 contra 0,65). O ajuste fino tira a média do ruído do professor. Em preço acontece o contrário (0,72 contra 0,98): o ouro tem só 16 avaliações que falam de preço, e com tão poucos exemplos o aluno não aprendeu a classe.
4. **Para triagem, todas as opções já são boas.** Os dois alunos detectam de 96% a 98% das avaliações que pedem ação. A diferença entre eles está no detalhe por aspecto.
5. **A pipeline totalmente aberta não gasta nenhuma chamada de API.** O Qwen3-8B rotulou as 10 mil avaliações em cerca de 67 minutos via Ollama (Vulkan), com o JSON restrito ao esquema. Cada aluno treinou em 44 minutos, usando 4,3 GB de VRAM.

## Rodando numa placa AMD de 8 GB

Os pacotes do PyTorch para ROCm 7.2 rodam nativamente na RX 7600 (gfx1102), sem precisar de `HSA_OVERRIDE_GFX_VERSION`. Com LoRA simples, um batch de 4 sequências de 192 tokens estourava os 8 GB. Três mudanças em [`src/train_lora.py`](src/train_lora.py) resolveram: agora o batch 8 usa no máximo 4,3 GB, e só o prompt curto já deixou o treino 3,4 vezes mais rápido.

- **Logits só nos tokens da resposta.** O vocabulário de 151 mil tokens do Qwen3 faz dos logits da sequência inteira o maior tensor do passo. A loss roda o `lm_head` só nas posições do JSON de resposta.
- **Prompt curto para o aluno.** O professor precisa do guia de anotação completo, com cerca de 400 tokens. Depois do ajuste, o aluno já aprendeu o esquema e recebe um prompt de uma linha, com cerca de 80 tokens. Isso barateia o treino e a inferência.
- **Padding em múltiplos de 32.** O ROCm escolhe um kernel para cada formato de tensor, então menos formatos diferentes significa menos tempo escolhendo kernels.

## Como reproduzir

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2   # ou a versão CUDA
pip install -r requirements.txt
python src/get_data.py                                                   # ouro público + prata do Haiku
python src/teacher_label.py --model qwen3:8b --out data/silver-qwen3-8b  # precisa do `ollama serve`
python src/train_lora.py --silver data/aspectos/silver --out runs/aluno-haiku --batch 8 --accum 2
python src/train_lora.py --silver data/silver-qwen3-8b --out runs/aluno-qwen3-8b --batch 8 --accum 2
python src/evaluate.py --name aluno-haiku --adapter runs/aluno-haiku     # repita para cada modelo
python src/report.py
```

O [`notebooks/kaggle_treino.ipynb`](notebooks/kaggle_treino.ipynb) está preparado para rodar o mesmo treino numa T4 gratuita do Kaggle. Lá o modelo roda em fp32, porque a T4 não tem bf16 nativo.

| Arquivo | O que faz |
|---|---|
| [`src/aspects.py`](src/aspects.py) | Esquema, prompts, leitura do JSON e a métrica |
| [`src/teacher_label.py`](src/teacher_label.py) | Professor aberto via Ollama, retomável |
| [`src/train_lora.py`](src/train_lora.py) | Treino LoRA, em ROCm ou CUDA |
| [`src/evaluate.py`](src/evaluate.py) | Aluno, professor e TF-IDF no conjunto ouro |
| [`src/report.py`](src/report.py) | IC por bootstrap e diferença pareada |

## Limitações

- **O ouro foi rotulado por LLMs, não por pessoas.** É o consenso entre o Claude Opus 5.5 e um segundo modelo. Só ficaram as avaliações em que os dois concordam nos cinco aspectos (329 de 400), o que puxa o conjunto para casos mais fáceis.
- **Conjunto pequeno.** Preço e embalagem têm só 16 e 13 avaliações no ouro em que o aspecto aparece, então o F1 desses aspectos tem muito ruído. Leve em conta os intervalos de confiança.
- **Uma rodada por braço.** Cada aluno foi treinado uma vez (seed 0), então não medi a variação entre seeds.
- **O prompt do professor não foi ajustado.** Escrevi o prompt uma vez, a partir do guia de anotação do ouro e antes de ver qualquer resultado, para não ajustar no conjunto de teste.
- **E-commerce, não delivery de comida.** As avaliações da Olist são sobre compras de produtos.

## Dados, créditos e licenças

- Avaliações: [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), CC BY-NC-SA 4.0.
- Conjunto ouro e rótulos prata do Claude Haiku: [RenatoSpessotto/olist-aspectos-reviews-ptbr](https://huggingface.co/datasets/RenatoSpessotto/olist-aspectos-reviews-ptbr), CC BY-NC-SA 4.0. Este projeto reaproveita os dados e a definição da métrica para medir uma variável nova: o professor.
- Modelos: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (aluno) e [Qwen3-8B](https://ollama.com/library/qwen3) (professor), ambos Apache 2.0.
- O código deste repositório está sob [MIT](LICENSE). Os dados e os adaptadores derivam de dados CC BY-NC-SA 4.0 e herdam essa licença: uso não comercial e compartilhamento pela mesma licença.
