# utils_ml.py
import os
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def ensure_data_csv(csv_path: str) -> pd.DataFrame:
    """
    Gera (se necessário) um CSV do Breast Cancer Wisconsin em data/breast_cancer.csv,
    lendo diretamente do sklearn.datasets. Retorna o DataFrame carregado do CSV.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        df = pd.concat([X, y], axis=1)
        # Converte nomes com espaços para snake_case (ajuda na manipulação)
        df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
        df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path)

def basic_eda(df: pd.DataFrame) -> None:
    st.write("**Dimensões:**", df.shape)
    st.write("**Amostra:**")
    st.dataframe(df.head())

    st.write("**Valores ausentes:**")
    nulls = df.isna().sum()
    st.dataframe(nulls[nulls > 0] if nulls.sum() > 0 else pd.Series({"Nenhum": 0}))

    st.write("**Balanceamento do target:**")
    bal = df["target"].value_counts().sort_index()
    st.bar_chart(bal.rename(index={0: "Benigno (0)", 1: "Maligno (1)"}))

def compute_supervised_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": None
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["roc_auc"] = None
    return metrics

def plot_confusion_matrix_plotly(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    z = cm.astype(int)
    x = labels
    y = labels
    fig = ff.create_annotated_heatmap(
        z=z, x=x, y=y, showscale=True, colorscale="Blues"
    )
    fig.update_layout(height=400, title="Matriz de Confusão")
    return fig

def build_md_report_supervised(selected_sections, metrics, model_name, test_size, scale_features, seed):
    parts = []
    if "Introdução" in selected_sections:
        parts.append(
            "# Relatório — ML na Saúde (Câncer de Mama)\n\n"
            "Este relatório resume um experimento de **classificação** para prever malignidade de tumores mamários "
            "a partir de medidas de imagens (dataset *Breast Cancer Wisconsin* via scikit-learn)."
        )
    if "Sobre o dataset" in selected_sections:
        parts.append(
            "## Sobre o dataset\n"
            "- Origem: `sklearn.datasets.load_breast_cancer`\n"
            "- Tarefa: classificação binária — 1: maligno, 0: benigno\n"
            "- Nº atributos: 30 numéricos + `target`\n"
        )
    if "Pré-processamento" in selected_sections:
        parts.append(
            "## Pré-processamento\n"
            f"- Split treino/teste com `test_size={test_size}` e `seed={seed}`.\n"
            f"- Padronização de features: {'ativada' if scale_features else 'desativada'} (StandardScaler).\n"
        )
    if "Resultados (supervisionado)" in selected_sections:
        parts.append(
            "## Resultados (supervisionado)\n"
            f"- Modelo: **{model_name}**\n"
            f"- Accuracy: **{metrics['accuracy']:.3f}**\n"
            f"- Precision: **{metrics['precision']:.3f}**\n"
            f"- Recall: **{metrics['recall']:.3f}**\n"
            f"- F1-score: **{metrics['f1']:.3f}**\n"
            f"- ROC AUC: **{metrics['roc_auc']:.3f}**\n" if metrics['roc_auc'] is not None else
            "## Resultados (supervisionado)\n- ROC AUC indisponível para o modelo selecionado.\n"
        )
    if "Resultados (não supervisionado)" in selected_sections:
        parts.append(
            "## Resultados (não supervisionado)\n"
            "Explorou-se **K-Means** com **PCA (2 componentes)** para visualização e inspeção de clusters.\n"
            "Relacione visualmente os clusters com o rótulo verdadeiro para investigar separabilidade."
        )
    if "Conclusões" in selected_sections:
        parts.append(
            "## Conclusões\n"
            "O dataset apresenta boa separabilidade e modelos lineares e de conjunto tendem a alcançar alto desempenho.\n"
            "Em produção, recomenda-se:\n"
            "- validação cruzada mais robusta, \n"
            "- avaliação de custo de erro assimétrico, \n"
            "- monitoramento de drift e fairness, \n"
            "- documentação de decisões e versionamento de dados/modelos."
        )
    return "\n\n".join(parts) + "\n"
