# BiomedBERT-based Classifier for Dentistry Studies

Classificador de artigos científicos (área e tipo de estudo) baseado no modelo **BiomedBERT**, com interface web local para classificação.

---

## Sumário

- [Instalação local (Linux Mint)](#instalação-local-linux-mint)
- [Como usar a interface](#como-usar-a-interface)
- [Treinamento no Google Colab (com GPU)](#treinamento-no-google-colab-com-gpu)
- [Categorias do modelo](#categorias-do-modelo)
- [Estrutura do projeto](#estrutura-do-projeto)

---

## Instalação local (Linux Mint)

### 1. Instalar o pip
```bash
sudo apt install python3-pip
```

### 2. Instalar o ambiente virtual
```bash
sudo apt install python3-venv
```

### 3. Criar o ambiente virtual
```bash
python3 -m venv biomedbert-env
```

### 4. Ativar o ambiente virtual
```bash
source biomedbert-env/bin/activate
```
Você verá `(biomedbert-env)` no início da linha do terminal — isso indica que está ativo. **Esse comando precisa ser rodado toda vez que você abrir um novo terminal**, antes de usar o programa.

### 5. Instalar as bibliotecas necessárias
```bash
pip install transformers torch flask flask-cors
```

### 6. Verificar se funcionou
```bash
python3 -c "from transformers import AutoTokenizer; print('OK')"
```
Se aparecer `OK`, está tudo certo.

---

## Como usar a interface

1. Ative o ambiente virtual:
   ```bash
   source biomedbert-env/bin/activate
   ```
2. Entre na pasta do projeto e inicie o servidor:
   ```bash
   cd /caminho/da/pasta
   python3 servidor.py
   ```
3. Abra o arquivo `interface.html` no navegador (Firefox ou Chrome).

**Atalho** (ativa o ambiente, entra na pasta e inicia tudo de uma vez):
```bash
source ~/biomedbert-env/bin/activate && cd ~/biomedbert/final\ com\ arquivos\ menores && python3 servidor.py
```

### O que a interface faz
- Envia um arquivo `.ris` ou `.rdf` exportado do Zotero
- Classifica todos os artigos usando o modelo BiomedBERT treinado
- Permite baixar o resultado em **JSON** ou **Markdown**

> O modelo já está treinado e salvo na pasta `modelo_treinado(1)/`. Não é necessário retreinar para classificar novos arquivos.

---

## Treinamento no Google Colab (com GPU)

Link do Google Colab: **https://colab.research.google.com/**

### Como ativar a GPU no Colab

1. Abra um novo notebook em https://colab.research.google.com/
2. No menu superior, clique em **Ambiente de execução** (ou "Runtime", se estiver em inglês).
3. Clique em **Alterar o tipo de ambiente de execução** ("Change runtime type").
4. Em **Acelerador de hardware** ("Hardware accelerator"), selecione **GPU** (pode aparecer como T4 GPU).
5. Clique em **Salvar**.
6. Para confirmar que a GPU está ativa, rode em uma célula:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Sem GPU")
   ```
   Deve imprimir `True` e o nome da GPU (ex: `Tesla T4`).

### Antes de rodar o script

Faça upload dos arquivos `area completo.json` e `tipo completo.json` para a pasta `/content/` do Colab (ícone de pasta no menu lateral esquerdo → botão de upload).

### Script de treinamento

Cole o script abaixo em uma célula do Colab e execute:

```python
# Instalar dependências
!pip install transformers torch accelerate -q

import json, os, torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from torch.utils.data import Dataset

# ── Modelo ────────────────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
print(f"Modelo: {MODEL_NAME}")

# ── Categorias ────────────────────────────────────────────────────────────────
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

# ── Texto de entrada ──────────────────────────────────────────────────────────
def texto_artigo(art):
    titulo = art.get("Título completo do estudo", "") or ""
    resumo = art.get("Resumo", "") or ""
    kws    = art.get("Palavras-chave", []) or []
    kw_str = " ".join(kws) if isinstance(kws, list) else str(kws)
    return f"{titulo} [SEP] {resumo} {kw_str}".strip()

# ── Dataset ───────────────────────────────────────────────────────────────────
class ArticleDataset(Dataset):
    def __init__(self, textos, labels, tok):
        self.enc    = tok(textos, truncation=True, padding=True,
                         max_length=512, return_tensors="pt")
        self.labels = torch.tensor(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
os.makedirs("/content/modelo_treinado", exist_ok=True)

# ── Treinar ÁREA (6 épocas) ───────────────────────────────────────────────────
print("\n=== Carregando arquivo de ÁREA ===")
with open("/content/area completo.json", encoding="utf-8") as f:
    dados_area = json.load(f)
print(f"Artigos carregados: {len(dados_area)}")

textos_area, labels_area = [], []
nao_encontrados_area = set()
for art in dados_area:
    val = art.get("Área Principal", "")
    if val in AREAS:
        textos_area.append(texto_artigo(art))
        labels_area.append(AREAS.index(val))
    elif val:
        nao_encontrados_area.add(val)

print(f"Artigos válidos para treino de ÁREA: {len(textos_area)}")
if nao_encontrados_area:
    print(f"Valores não reconhecidos em ÁREA: {nao_encontrados_area}")

model_area = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(AREAS))

args_area = TrainingArguments(
    output_dir="/content/modelo_treinado/area_checkpoints",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    save_strategy="no",
    logging_steps=50,
)
Trainer(model=model_area, args=args_area,
        train_dataset=ArticleDataset(textos_area, labels_area, tokenizer)).train()
model_area.save_pretrained("/content/modelo_treinado/area", max_shard_size="90MB")
tokenizer.save_pretrained("/content/modelo_treinado/area")
print("ÁREA salvo.")

# ── Treinar TIPO (6 épocas) ───────────────────────────────────────────────────
print("\n=== Carregando arquivo de TIPO ===")
with open("/content/tipo completo.json", encoding="utf-8") as f:
    dados_tipo = json.load(f)
print(f"Artigos carregados: {len(dados_tipo)}")

textos_tipo, labels_tipo = [], []
nao_encontrados_tipo = set()
for art in dados_tipo:
    val = art.get("Tipo de estudo", "")
    if val in TIPOS:
        textos_tipo.append(texto_artigo(art))
        labels_tipo.append(TIPOS.index(val))
    elif val:
        nao_encontrados_tipo.add(val)

print(f"Artigos válidos para treino de TIPO: {len(textos_tipo)}")
if nao_encontrados_tipo:
    print(f"Valores não reconhecidos em TIPO: {nao_encontrados_tipo}")

model_tipo = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(TIPOS))

args_tipo = TrainingArguments(
    output_dir="/content/modelo_treinado/tipo_checkpoints",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    save_strategy="no",
    logging_steps=50,
)
Trainer(model=model_tipo, args=args_tipo,
        train_dataset=ArticleDataset(textos_tipo, labels_tipo, tokenizer)).train()
model_tipo.save_pretrained("/content/modelo_treinado/tipo", max_shard_size="90MB")
tokenizer.save_pretrained("/content/modelo_treinado/tipo")
print("TIPO salvo.")

# ── Zipar e baixar ────────────────────────────────────────────────────────────
import shutil
shutil.make_archive("/content/modelo_treinado", "zip", "/content/modelo_treinado")
from google.colab import files
files.download("/content/modelo_treinado.zip")
print("Download iniciado!")
```

Ao final, o Colab vai baixar automaticamente um arquivo `modelo_treinado.zip` contendo os dois modelos treinados (`area/` e `tipo/`). Extraia esse zip e coloque a pasta junto com `servidor.py` e `interface.html` para usar os modelos na interface local.

---

## Categorias do modelo

### Área Principal (4 categorias)
- Áreas não relacionadas à saúde
- Ciências Biológicas e da Saúde
- Medicina
- Odontologia

### Tipo de Estudo (7 categorias)
- Caso clínico / Série de casos
- Estudo em animais
- Estudo in vitro
- Não foi possível classificar
- Observacional
- RCT
- Revisão

---

## Estrutura do projeto

```
.
├── servidor.py                    # Backend Flask (classificação)
├── interface.html                 # Interface web
├── modelo_treinado(1)/            # Modelo treinado em shards de 90MB
│   ├── area/                      # Modelo de classificação de área
│   │   ├── model-00001-of-00005.safetensors
│   │   ├── ...
│   │   └── tokenizer.json
│   └── tipo/                      # Modelo de classificação de tipo
│       ├── model-00001-of-00005.safetensors
│       ├── ...
│       └── tokenizer.json
└── README.md
```
