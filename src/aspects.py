"""Esquema, prompt, leitura dos dados e métrica. Tudo que professor, aluno e avaliação compartilham."""
import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ASPECTS = ["entrega", "produto", "atendimento", "preco", "embalagem"]
LABELS = ["positivo", "negativo", "neutro", "ausente"]
DATA = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

# Mesmas definições do guia de anotação do conjunto ouro (RenatoSpessotto/olist-aspectos-reviews-ptbr),
# para que professor e ouro rotulem a mesma coisa.
INSTRUCTION = """Classifique o sentimento de cada aspecto desta avaliação de compra online.
Rótulos: positivo (elogio ou satisfação), negativo (reclamação ou insatisfação), neutro (citado sem julgamento), ausente (não aparece).
Aspectos:
- entrega: prazo, atraso, rastreio, entrega errada ou incompleta, não recebimento. Estado da caixa não conta.
- produto: qualidade, defeito, se corresponde ao anúncio, funcionamento, elogio genérico ("adorei", "recomendo"). Preço não conta.
- atendimento: contato com a loja ou vendedor, resposta, troca, solução de problemas. Entregador vai em entrega.
- preco: preço, custo-benefício, frete caro ou barato.
- embalagem: caixa amassada, proteção, lacre, embalagem bonita.
Regras: "chegou tudo certo" = entrega positivo e produto ausente; "ainda não recebi" = entrega negativo; "recebi o produto errado" = entrega e produto negativo; ironia conta pelo sentido real; elogio genérico à loja vai em produto.
Responda só com JSON no formato {"entrega": "...", "produto": "...", "atendimento": "...", "preco": "...", "embalagem": "..."}.

Avaliação: """

JSON_SCHEMA = {
    "type": "object",
    "properties": {a: {"type": "string", "enum": LABELS} for a in ASPECTS},
    "required": ASPECTS,
}


def prompt(text: str) -> str:
    return INSTRUCTION + text.strip()


def short_prompt(text: str) -> str:
    # O aluno ajustado já aprendeu o esquema: sem as instruções, a sequência cai de ~400 para ~80 tokens,
    # o que deixa treino e inferência ~4x mais baratos.
    return "Aspectos (entrega, produto, atendimento, preco, embalagem) desta avaliação em JSON:\n" + text.strip()


def target(row: dict) -> str:
    return json.dumps({a: row[a] for a in ASPECTS}, ensure_ascii=False)


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse(output: str) -> dict | None:
    """Primeiro objeto JSON da saída, com os 5 aspectos e rótulos válidos; None se não houver."""
    m = re.search(r"\{.*?\}", output, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(d.get(a) in LABELS for a in ASPECTS):
        return None
    return {a: d[a] for a in ASPECTS}


def aspect_f1(gold: list[dict], pred: list[dict | None]) -> dict[str, float]:
    """F1-macro por aspecto sobre os rótulos presentes no ouro ou na predição, e a média entre aspectos.

    Mesma definição do repositório de referência, para os números serem comparáveis.
    Saída inválida conta como "ausente" em todos os aspectos.
    """
    pred = [p or {a: "ausente" for a in ASPECTS} for p in pred]
    out = {}
    for a in ASPECTS:
        y, p = [g[a] for g in gold], [q[a] for q in pred]
        present = sorted(set(y) | set(p))
        out[a] = float(f1_score(y, p, labels=present, average="macro", zero_division=0))
    out["macro"] = float(np.mean([out[a] for a in ASPECTS]))
    # Visão de negócio: o review pede ação (algum aspecto negativo)? Recall importa mais que precisão aqui.
    yn = [any(g[a] == "negativo" for a in ASPECTS) for g in gold]
    pn = [any(q[a] == "negativo" for a in ASPECTS) for q in pred]
    tp = sum(y and p for y, p in zip(yn, pn))
    out["acao_recall"] = tp / max(sum(yn), 1)
    out["acao_precisao"] = tp / max(sum(pn), 1)
    return out


if __name__ == "__main__":
    g = [{a: "ausente" for a in ASPECTS} | {"entrega": "negativo"}, {a: "positivo" for a in ASPECTS}]
    assert aspect_f1(g, g)["macro"] == 1.0
    assert aspect_f1(g, [None, None])["acao_recall"] == 0.0
    assert parse('lixo {"entrega": "negativo", "produto": "ausente", "atendimento": "ausente", "preco": "ausente", "embalagem": "ausente"} fim')["entrega"] == "negativo"
    assert parse('{"entrega": "ruim"}') is None
    print("ok")
