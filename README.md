
# 🩺 ML na Saúde — Classificação de Câncer de Mama (Streamlit)

[![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b?logo=streamlit\&logoColor=white)](#-demo)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn\&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly\&logoColor=white)](https://plotly.com/python/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](#-licença)

Aplicação **interativa** em Streamlit demonstrando:

* **Aprendizagem Supervisionada** (Regressão Logística, Random Forest, SVM–RBF)
* **Aprendizagem Não Supervisionada** (K-Means + PCA em 2D)

A interface fica **dividida em duas colunas**:
▶️ **Esquerda:** modelos supervisionados (métricas, ROC, Matriz de Confusão, importância/coeficientes)
🧩 **Direita:** K-Means com PCA (2D), *Silhouette score*, resumo por cluster

O dataset é o **Breast Cancer Wisconsin** (via `scikit-learn`) e é **gerado localmente** (`data/breast_cancer.csv`).

---

## 📌 Sumário

* [Demo](#-demo)
* [Arquitetura & Pipeline](#-arquitetura--pipeline)
* [Recursos (o que você verá no app)](#-recursos-o-que-você-verá-no-app)
* [Pré-requisitos (macOS)](#-prérequisitos-macos)
* [Instalação Rápida (1 comando)](#-instalação-rápida-1-comando)
* [Instalação Passo a Passo (macOS)](#-instalação-passo-a-passo-macos)
* [Como Usar (guiado)](#-como-usar-guiado)
* [Gráficos & Dashboard (exemplos)](#-gráficos--dashboard-exemplos)
* [Planilha & Relatórios](#-planilha--relatórios)
* [Estrutura do Projeto](#-estrutura-do-projeto)
* [Deploy no Streamlit Cloud](#-deploy-no-streamlit-cloud)
* [Boas Práticas & Próximos Passos](#-boas-práticas--próximos-passos)
* [Troubleshooting (macOS)](#-troubleshooting-macos)
* [Licença](#-licença)

---

## 🚀 Demo

> Substitua pelos seus links reais quando publicar:

* **App (Streamlit Cloud):** `https://ml-saude-SEUUSUARIO.streamlit.app`
* **Repositório (GitHub):** `https://github.com/SEUUSUARIO/ml-saude-bc`

---


## ✨ Recursos (o que você verá no app)

* **Duas colunas lado a lado:**

  * **Supervisionado:** escolha do modelo, métricas (Accuracy, Precision, Recall, F1, ROC AUC), **Matriz de Confusão**, **Curva ROC**, **importância/coeficientes**, **validação cruzada (5-fold)**.
  * **Não supervisionado:** **K-Means** (k ajustável), **PCA (2D)** com gráficos interativos (Plotly), **Silhouette score**, **tabela** com tamanho e % de malignos por cluster.
* **EDA rápida:** dimensões, amostra, verificação de ausentes e balanceamento do target.
* **Relatório exportável:** download de **Markdown** com resumo do experimento.

---

## 🍎 Pré-requisitos (macOS)

* **Homebrew** (opcional, recomendado):

  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
* **Python 3.11** (recomendado para evitar problemas com wheels no 3.13):

  ```bash
  brew install python@3.11
  ```

---

## ⚡ Instalação Rápida (1 comando)

Dentro da pasta do projeto:

```bash
python3.11 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt && streamlit run app.py
```

O app abre em **[http://localhost:8501](http://localhost:8501)**.

---

## 🧩 Instalação Passo a Passo (macOS)

```bash
# 1) entrar no projeto
cd /caminho/para/ml-saude-bc

# 2) criar e ativar venv (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate

# 3) instalar dependências
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) rodar
streamlit run app.py
```

> Se preferir outra porta: `streamlit run app.py --server.port 8502`

---

## 🧭 Como Usar (guiado)

1. **Sidebar**: defina *seed*, **test_size** e se quer **padronizar features**.
2. **Dataset & EDA**: confira dimensões, amostra e balanceamento.
3. **Coluna esquerda (Supervisionado)**:

   * Selecione o modelo (LogReg, RF, SVM-RBF).
   * Ajuste hiperparâmetros nos *sliders*.
   * Analise **métricas**, **Matriz de Confusão**, **Curva ROC** (quando disponível) e **importâncias/coeficientes**.
   * Veja a **validação cruzada (5-fold)**.
4. **Coluna direita (Não supervisionado)**:

   * Ajuste **k** do K-Means (2–6).
   * Observe a projeção **PCA (2D)** por **cluster** e por **rótulo real**.
   * Confira o **Silhouette score** e a **tabela** com **tamanho** e **% de malignos** por cluster.
5. **Relatório**:

   * Selecione seções e **baixe o Markdown** (`relatorio_ml_saude.md`).

---

## 📑 Planilha & Relatórios

* **CSV gerado automaticamente**: `data/breast_cancer.csv` na primeira execução.
* **Relatório em Markdown**: use a aba **Relatório & Reprodutibilidade** para baixar `relatorio_ml_saude.md`.
* **Dica**: você pode abrir o CSV no Excel/Numbers/Google Sheets para análises tabulares adicionais.

> Se quiser, crie uma pasta `reports/` no repo e **acompanhe versões** dos relatórios (ex.: `reports/2025-10-29_relatorio.md`).

---

## 🧱 Estrutura do Projeto

```
.
├─ app.py                # UI do Streamlit (duas colunas: Superv. | Não Superv.)
├─ utils_ml.py           # EDA, métricas, matriz de confusão (Plotly), relatório
├─ requirements.txt      # Dependências (leves e compatíveis com Streamlit Cloud)
├─ README.md             # Este arquivo
├─ data/                 # CSV gerado: data/breast_cancer.csv
└─ assets/               # (opcional) screenshots para o README
```

---

## ☁️ Deploy no Streamlit Cloud

1. Suba o projeto no **GitHub** (branch `main`).
2. Acesse **[https://share.streamlit.io/](https://share.streamlit.io/)** e conecte sua conta GitHub.
3. Selecione o repositório e o arquivo principal **`app.py`**.
4. Confirme as dependências via `requirements.txt`.
5. Publique. Seu link ficará algo como:

   ```
   https://ml-saude-SEUUSUARIO.streamlit.app
   ```

---

## 🧭 Boas Práticas & Próximos Passos

* **Modelos adicionais**: XGBoost/LightGBM, tuning com **Optuna**.
* **Explicabilidade**: **SHAP** para explicações locais/globais.
* **Fairness**: métricas de equidade, avaliação de **custo assimétrico**.
* **Produção**: pipelines `sklearn`, versionamento de dados/modelos, **monitoramento de drift**.
* **Testes**: unitários para funções (ex.: `utils_ml.py`).

---

## 🆘 Troubleshooting (macOS)

**1) `ModuleNotFoundError: No module named 'plotly'`**
Ative a venv correta e reinstale:

```bash
source .venv/bin/activate
pip install plotly
```

**2) `ModuleNotFoundError: No module named 'utils_ml'`**
`app.py` e `utils_ml.py` precisam estar na **mesma pasta**.
Rode o app **dentro** do diretório do projeto:

```bash
cd /caminho/para/ml-saude-bc
streamlit run app.py
```

**3) Erros com scikit-learn no Python 3.13**
Use **Python 3.11**:

```bash
brew install python@3.11
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**4) Rodar em outra porta**

```bash
streamlit run app.py --server.port 8502
```

---

## 📄 Licença

Uso **acadêmico/educacional**. Sinta-se livre para forkar, adaptar e evoluir.

---

> **Contato & Suporte**
> Achou um bug, quer sugerir melhoria ou pedir uma feature?
> Abra uma **Issue** no GitHub com screenshot/log e passos para reproduzir.
