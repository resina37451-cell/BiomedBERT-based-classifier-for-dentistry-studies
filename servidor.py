import json
import os
import re
import threading
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

AREAS = [
    "Áreas não relacionadas à saúde",
    "Ciências Biológicas e da Saúde",
    "Medicina",
    "Odontologia",
]

TIPOS = [
    "Caso clínico / Série de casos",
    "Estudo em animais",
    "Estudo in vitro",
    "Não foi possível classificar",
    "Observacional",
    "RCT",
    "Revisão",
]

MODEL_DIR = os.path.expanduser("~/biomedbert/final com arquivos menores/modelo_treinado(1)")
STATUS = {"etapa": "idle", "progresso": 0, "mensagem": "Aguardando"}

# ── FIX 1: cache global dos modelos — carrega uma vez, reutiliza sempre ──────
_modelos = {}

def carregar_modelo(nome):
    """Carrega e cacheia tokenizer + modelo. Thread-safe via GIL."""
    if nome not in _modelos:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = f"{MODEL_DIR}/{nome}"
        print(f"  [cache] Carregando modelo '{nome}' pela primeira vez…")
        tok = AutoTokenizer.from_pretrained(path)
        mod = AutoModelForSequenceClassification.from_pretrained(path)
        mod.eval()
        _modelos[nome] = (tok, mod)
        print(f"  [cache] '{nome}' pronto.")
    return _modelos[nome]

# ── FIX 2: pré-carrega modelos ao iniciar (evita timeout na 1ª requisição) ───
def precarregar():
    if (os.path.exists(f"{MODEL_DIR}/area") and
            os.path.exists(f"{MODEL_DIR}/tipo")):
        print("  Pré-carregando modelos em background…")
        threading.Thread(target=lambda: [carregar_modelo("area"),
                                         carregar_modelo("tipo")],
                         daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────

def texto_artigo(art):
    titulo = art.get("Título completo do estudo", "") or ""
    resumo = art.get("Resumo", "") or ""
    kws    = art.get("Palavras-chave", []) or []
    kw_str = " ".join(kws) if isinstance(kws, list) else str(kws)
    return f"{titulo} [SEP] {resumo} {kw_str}".strip()


def justificativa_area(label, confianca):
    pct   = round(confianca * 100, 1)
    nivel = "alta" if confianca >= 0.85 else "moderada" if confianca >= 0.60 else "baixa"
    return (f"Confiança {nivel} ({pct}%) — BiomedBERT classificou como '{label}' "
            f"com base no conteúdo do título, abstract e palavras-chave.")


def justificativa_tipo(label, confianca):
    pct   = round(confianca * 100, 1)
    nivel = "alta" if confianca >= 0.85 else "moderada" if confianca >= 0.60 else "baixa"
    return (f"Confiança {nivel} ({pct}%) — BiomedBERT identificou padrão metodológico "
            f"de '{label}' no abstract do estudo.")


def parse_ris(conteudo):
    artigos, atual = [], {}
    for linha in conteudo.splitlines():
        linha = linha.strip()
        if not linha:
            continue
        if linha == "ER  -":
            if atual:
                artigos.append(atual)
                atual = {}
            continue
        m = re.match(r'^([A-Z][A-Z0-9])\s+-\s*(.*)', linha)
        if not m:
            continue
        tag, val = m.group(1), m.group(2).strip()
        if tag == "TI":
            atual["Título completo do estudo"] = val
        elif tag == "AB":
            atual["Resumo"] = atual.get("Resumo", "") + " " + val
        elif tag == "T2":
            atual["Periódico"] = val
        elif tag == "J2":
            atual["Abreviação"] = val
        elif tag == "SN":
            atual["ISSN"] = val
        elif tag == "AU":
            atual.setdefault("Autores", []).append(val)
        elif tag == "KW":
            atual.setdefault("Palavras-chave", []).append(val)
        elif tag == "DO":
            atual["DOI"] = val
        elif tag == "PY":
            atual["Ano"] = val
        elif tag == "CY":
            atual["Local"] = val
    return artigos


def parse_rdf(conteudo):
    artigos = []
    try:
        root = ET.fromstring(conteudo)
    except ET.ParseError:
        return artigos

    ns = {
        "bib":     "http://purl.org/net/biblio#",
        "dc":      "http://purl.org/dc/elements/1.1/",
        "dcterms": "http://purl.org/dc/terms/",
        "foaf":    "http://xmlns.com/foaf/0.1/",
        "prism":   "http://prismstandard.org/namespaces/1.2/basic/",
        "rdf":     "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "z":       "http://www.zotero.org/namespaces/export#",
        "vcard":   "http://nkd.org/vCard#",
    }

    journals = {}
    for j in root.findall(".//bib:Journal", ns):
        about = j.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", "")
        title = j.findtext("dc:title",             namespaces=ns, default="")
        alt   = j.findtext("dcterms:alternative",  namespaces=ns, default="")
        issn_raw = j.findtext("dc:identifier",     namespaces=ns, default="")
        issn  = issn_raw.replace("ISSN ", "").strip() if "ISSN" in issn_raw else ""
        journals[about] = {"Periódico": title, "Abreviação": alt, "ISSN": issn}

    for art in root.findall(".//bib:Article", ns):
        a = {}
        a["Título completo do estudo"] = (art.findtext("dc:title",        namespaces=ns) or "").strip()
        a["Resumo"]                    = (art.findtext("dcterms:abstract", namespaces=ns) or "").strip()
        a["Local"] = ""

        parte_ref = art.find("dcterms:isPartOf", ns)
        if parte_ref is not None:
            ref = parte_ref.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", "")
            if ref in journals:
                a.update(journals[ref])

        autores = []
        for pessoa in art.findall(".//foaf:Person", ns):
            sn = pessoa.findtext("foaf:surname",   namespaces=ns, default="")
            gn = pessoa.findtext("foaf:givenName", namespaces=ns, default="")
            if sn:
                autores.append(f"{sn}, {gn}".strip(", "))
        a["Autores"] = autores

        kws = [s.text.strip() for s in art.findall("dc:subject", ns) if s.text]
        a["Palavras-chave"] = kws

        data = art.findtext("dc:date", namespaces=ns, default="")
        a["Ano"] = data[:4] if data else ""

        if a.get("Título completo do estudo"):
            artigos.append(a)
    return artigos


# ── FIX 3: processamento em lotes para não explodir a RAM ────────────────────
def classificar_em_lotes(textos, tok, mod, lote=64):
    """Roda inferência em lotes de `lote` textos para economizar memória."""
    import torch
    preds_total, confs_total = [], []
    for i in range(0, len(textos), lote):
        bloco = textos[i:i + lote]
        enc = tok(bloco, truncation=True, padding=True,
                  max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = mod(**enc).logits
        preds_total.extend(torch.argmax(logits, dim=1).tolist())
        confs_total.extend(torch.softmax(logits, dim=1).max(dim=1).values.tolist())
    return preds_total, confs_total


def classificar_artigos(artigos):
    global STATUS
    resultados = []
    textos = [texto_artigo(a) for a in artigos]
    total  = len(artigos)

    for nome, labels in [("area", AREAS), ("tipo", TIPOS)]:
        tok, mod = carregar_modelo(nome)

        etapa_msg = "Classificando área principal" if nome == "area" else "Classificando tipo de estudo"
        STATUS = {"etapa": "classificando", "progresso": 20 if nome == "area" else 60,
                  "mensagem": f"{etapa_msg} ({total} artigos)…"}

        preds, confs = classificar_em_lotes(textos, tok, mod)

        if nome == "area":
            for i, art in enumerate(artigos):
                r = dict(art)
                r["Área Principal"]        = labels[preds[i]]
                r["Confiança Área"]        = round(confs[i], 4)
                r["Justificativa da Área"] = justificativa_area(labels[preds[i]], confs[i])
                resultados.append(r)
        else:
            for i, r in enumerate(resultados):
                r["Tipo de estudo"]        = labels[preds[i]]
                r["Confiança Tipo"]        = round(confs[i], 4)
                r["Justificativa do Tipo"] = justificativa_tipo(labels[preds[i]], confs[i])

    return resultados


def gerar_markdown(resultados):
    linhas = []
    linhas.append("# Resultado da Classificação BiomedBERT\n")
    linhas.append(f"Total de artigos classificados: **{len(resultados)}**\n")
    linhas.append("---\n")

    for i, r in enumerate(resultados, 1):
        titulo  = r.get("Título completo do estudo", "").strip()
        ano     = r.get("Ano", "").strip()
        resumo  = r.get("Resumo", "").strip()
        area    = r.get("Área Principal", "")
        tipo    = r.get("Tipo de estudo", "")
        conf_a  = r.get("Confiança Área", 0)
        conf_t  = r.get("Confiança Tipo", 0)
        just_a  = r.get("Justificativa da Área", "")
        just_t  = r.get("Justificativa do Tipo", "")
        periodo = r.get("Periódico", "")
        doi     = r.get("DOI", "")

        linhas.append(f"## {i}. {titulo}")
        linhas.append("")
        linhas.append("| Campo | Valor |")
        linhas.append("|---|---|")
        linhas.append(f"| **Ano** | {ano} |")
        linhas.append(f"| **Periódico** | {periodo} |")
        if doi:
            linhas.append(f"| **DOI** | {doi} |")
        linhas.append(f"| **Área Principal** | {area} |")
        linhas.append(f"| **Tipo de Estudo** | {tipo} |")
        linhas.append(f"| **Confiança Área** | {round(conf_a*100,1)}% |")
        linhas.append(f"| **Confiança Tipo** | {round(conf_t*100,1)}% |")
        linhas.append("")
        linhas.append("**Justificativa da Área:**")
        linhas.append(f"> {just_a}")
        linhas.append("")
        linhas.append("**Justificativa do Tipo:**")
        linhas.append(f"> {just_t}")
        linhas.append("")
        linhas.append("**Abstract:**")
        linhas.append("")
        linhas.append(resumo)
        linhas.append("")
        linhas.append("---")
        linhas.append("")

    return "\n".join(linhas)


# ── FIX 4: classificação roda em thread separada; resposta imediata ───────────
_lock_classificacao = threading.Lock()

@app.route("/status")
def status():
    return jsonify(STATUS)


@app.route("/classificar", methods=["POST"])
def classificar():
    global STATUS

    if not os.path.exists(f"{MODEL_DIR}/area") or not os.path.exists(f"{MODEL_DIR}/tipo"):
        return jsonify({"erro": "Modelo não encontrado. Verifique a pasta modelo_treinado."}), 400

    if STATUS.get("etapa") == "classificando":
        return jsonify({"erro": "Já existe uma classificação em andamento. Aguarde."}), 409

    arquivo = request.files.get("arquivo")
    if not arquivo:
        return jsonify({"erro": "Arquivo não enviado"}), 400

    nome_arquivo = arquivo.filename.lower()
    conteudo     = arquivo.read().decode("utf-8", errors="ignore")

    if nome_arquivo.endswith(".ris"):
        artigos = parse_ris(conteudo)
    elif nome_arquivo.endswith((".rdf", ".xml")):
        artigos = parse_rdf(conteudo)
    else:
        return jsonify({"erro": "Formato não suportado. Use .ris ou .rdf"}), 400

    if not artigos:
        return jsonify({"erro": "Nenhum artigo encontrado no arquivo"}), 400

    STATUS = {"etapa": "classificando", "progresso": 5,
              "mensagem": f"Iniciando classificação de {len(artigos)} artigos…"}

    # Roda em background; frontend faz polling em /status e busca /baixar quando pronto
    def tarefa():
        global STATUS
        with _lock_classificacao:
            try:
                resultados = classificar_artigos(artigos)
                STATUS = {"etapa": "idle", "progresso": 100,
                          "mensagem": f"✔ {len(resultados)} artigos classificados"}

                with open("./resultado_classificacao.json", "w", encoding="utf-8") as f:
                    json.dump(resultados, f, ensure_ascii=False, indent=2)
                with open("./resultado_classificacao.md", "w", encoding="utf-8") as f:
                    f.write(gerar_markdown(resultados))

                STATUS["total"] = len(resultados)

            except Exception as e:
                STATUS = {"etapa": "erro", "progresso": 0, "mensagem": str(e)}

    threading.Thread(target=tarefa, daemon=True).start()

    # Responde imediatamente — o frontend vai fazer polling
    return jsonify({"ok": True, "async": True, "total_enviado": len(artigos)})


@app.route("/baixar")
def baixar():
    formato = request.args.get("formato", "json")
    if formato == "md":
        path = "./resultado_classificacao.md"
        if not os.path.exists(path):
            return jsonify({"erro": "Nenhuma classificação disponível ainda."}), 400
        return send_file(path, as_attachment=True,
                         download_name="resultado_classificacao.md",
                         mimetype="text/markdown")
    else:
        path = "./resultado_classificacao.json"
        if not os.path.exists(path):
            return jsonify({"erro": "Nenhuma classificação disponível ainda."}), 400
        return send_file(path, as_attachment=True,
                         download_name="resultado_classificacao.json",
                         mimetype="application/json")


if __name__ == "__main__":
    precarregar()
    print("=" * 55)
    print("  BiomedBERT Classificador iniciado")
    print("  Abra o arquivo interface.html no navegador")
    print("=" * 55)
    # threaded=True para aguentar polling simultâneo do frontend
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
