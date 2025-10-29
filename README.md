🩺 ML na Saúde — Classificação de Câncer de Mama (Streamlit)

Aplicação interativa em Streamlit demonstrando:

Aprendizagem Supervisionada (Regressão Logística, Random Forest, SVM–RBF)

Aprendizagem Não Supervisionada (K-Means + PCA em 2D)

A interface fica dividida em duas colunas:

▶️ Esquerda: modelos supervisionados (métricas, ROC, Matriz de Confusão, importância/coeficientes)

🧩 Direita: K-Means com PCA (2D), Silhouette score, resumo por cluster

O dataset é o Breast Cancer Wisconsin (via sklearn), e é gerado localmente em CSV na pasta data/.

🔗 Entregáveis

App (Streamlit Cloud): https://ml-saude-SEUUSUARIO.streamlit.app

Repositório (GitHub): https://github.com/SEUUSUARIO/ml-saude-bc

Substitua pelos seus links ao publicar.

📦 Estrutura do Projeto
.
├─ app.py                # UI do Streamlit (duas colunas: Superv. | Não Superv.)
├─ utils_ml.py           # EDA, métricas, matriz de confusão (Plotly), relatório
├─ requirements.txt      # Dependências (leves e compatíveis com Streamlit Cloud)
├─ README.md             # Este arquivo
└─ data/                 # Será criado o CSV: data/breast_cancer.csv


O arquivo data/breast_cancer.csv é criado automaticamente na primeira execução.

⚙️ Tecnologias

Python: 3.11+ (recomendado para compatibilidade de wheels)

Framework Web: Streamlit

ML/Data: scikit-learn, pandas, numpy

Visualizações: Plotly

🚀 Quick Start (macOS)

Abra o Terminal na pasta do projeto e rode um único comando:

python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt && streamlit run app.py


O app abrirá em http://localhost:8501.

Se aparecer erro para instalar scikit-learn no Python 3.13, use Python 3.11 (ver seção abaixo).

🍎 Setup no MacBook (passo a passo)
1) Pré-requisitos (Homebrew e Python)

Se ainda não tem o Homebrew:

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"


Instale o Python 3.11 (evita problemas de wheel no 3.13):

brew install python@3.11

2) Ambiente virtual (zsh)
python3.11 -m venv .venv
source .venv/bin/activate

3) Dependências
python -m pip install --upgrade pip
pip install -r requirements.txt

4) Rodar o app
streamlit run app.py

🧠 Funcionalidades
📥 Dataset & EDA

Carrega sklearn.datasets.load_breast_cancer

Gera data/breast_cancer.csv (para reprodutibilidade)

Mostra dimensões, amostra, valores ausentes e balanceamento do target

🎓 Coluna Esquerda — Supervisionado

Modelos: Regressão Logística | Random Forest | SVM (RBF)

Métricas: Accuracy, Precision, Recall, F1, ROC AUC

Gráficos: Matriz de Confusão (Plotly), Curva ROC

Interpretação: Importância (RF) e coeficientes (modelos lineares)

Validação Cruzada: 5-fold (accuracy)

🧩 Coluna Direita — Não Supervisionado

K-Means (k ajustável) + PCA (2D)

Silhouette score

Resumo por cluster: tamanho e % de malignos por cluster

📥 Relatório

Geração de Markdown (relatorio_ml_saude.md) com seções selecionáveis:

Introdução, Dataset, Pré-processamento, Resultados (sup/unsup), Conclusões

☁️ Publicação no Streamlit Cloud (macOS)

Suba o repositório no GitHub (ou faça fork).

Acesse https://share.streamlit.io/
 e conecte seu GitHub.

Selecione o repositório, branch e o arquivo principal app.py.

Confirme o uso do requirements.txt.

Publique e pegue o link (https://ml-saude-SEUUSUARIO.streamlit.app).

🧪 Comandos úteis (macOS)

Ativar venv

source .venv/bin/activate


Verificar Python/pip ativos (devem apontar para .venv)

which python
which pip


Teste rápido das importações

python -c "import streamlit, pandas, numpy, sklearn, plotly; print('OK')"


Rodar em porta alternativa

streamlit run app.py --server.port 8502

🛠️ Troubleshooting (macOS)
1) ModuleNotFoundError: No module named 'plotly'

Garanta que a venv está ativa (.venv no prompt)

Reinstale:

pip install plotly

2) ModuleNotFoundError: No module named 'utils_ml'

app.py e utils_ml.py devem estar na mesma pasta.

Rode o app dentro da pasta do projeto:

cd /caminho/para/ml-saude-bc
streamlit run app.py

3) Erros ao instalar scikit-learn (Python 3.13)

Prefira Python 3.11 (tem wheels prontos):

brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

4) TypeError: '<' not supported between instances of 'str' and 'int'

Corrigido no código final: índices de clusters unificados como str no resumo por cluster.

🧾 Requirements (já prontos)

Arquivo requirements.txt:

streamlit>=1.37
pandas>=2.1
numpy>=1.26
scikit-learn>=1.4
plotly>=5.22

📚 Metodologia (resumo)

Coleta/Criação do dataset a partir do sklearn → CSV local

Pré-processamento: train/test split estratificado, StandardScaler (opcional)

Treino/Validação: métricas + 5-fold CV

Interpretação: Matriz de Confusão, ROC, importâncias/coeficientes

Não supervisionado: K-Means + PCA (2D), Silhouette, tabela por cluster

Comunicação: gráficos interativos e relatório em Markdown

🤝 Contribuição

Contribuições são bem-vindas!
Sugestões de melhoria:

Hiperparâmetros com Optuna

Modelos adicionais (XGBoost/LightGBM)

Fairness & explicabilidade (SHAP)

Pipeline sklearn + model persistence (joblib)

📄 Licença

Projeto para fins acadêmicos/educacionais.
Sinta-se à vontade para forkar e adaptar.

👋 Dica final (MacBook)

Se usar Apple Silicon (M1/M2/M3), o fluxo com python@3.11 e pip atualizado tende a evitar compilações demoradas e incompatibilidades.
