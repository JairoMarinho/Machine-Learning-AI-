# app.py
import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, silhouette_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from utils_ml import (
    ensure_data_csv, basic_eda, compute_supervised_metrics,
    plot_confusion_matrix_plotly, build_md_report_supervised
)

st.set_page_config(
    page_title="ML na Saúde — Câncer de Mama",
    page_icon="🩺",
    layout="wide"
)

# =========================
# Sidebar — Configurações
# =========================
st.sidebar.title("⚙️ Configurações")
seed = st.sidebar.number_input("Seed (reprodutibilidade)", 0, 9999, 42, step=1)
test_size = st.sidebar.slider("Proporção de teste", 0.1, 0.4, 0.2, 0.05)
scale_features = st.sidebar.checkbox("Padronizar features (StandardScaler)", True)

st.sidebar.markdown("---")
st.sidebar.caption("Projeto didático — ML aplicado à saúde (câncer de mama)")

# =========================
# Seção: Dataset
# =========================
st.title("🩺 Machine Learning na Saúde — Classificação de Câncer de Mama")
st.write(
    "Aplicação interativa que demonstra **aprendizagem supervisionada** e **não supervisionada** "
    "no dataset público *Breast Cancer Wisconsin* (sklearn)."
)

data_path = os.path.join("data", "breast_cancer.csv")
df = ensure_data_csv(data_path)   # cria/atualiza CSV local a partir do sklearn

with st.expander("📄 Dicionário de variáveis (resumo)", expanded=False):
    st.markdown("""
- **target**: 1 = **maligno**, 0 = **benigno**  
- Demais colunas: estatísticas de textura/forma dos núcleos celulares (média, erro padrão, pior caso) medidas em imagens;  
  exemplos: *mean radius, mean texture, worst symmetry*, etc.
    """)

st.subheader("👀 Visão geral do dataset")
basic_eda(df)

# =========================
# Split e pré-processamento
# =========================
X = df.drop(columns=["target"])
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed, stratify=y
)

scaler = None
if scale_features:
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns, index=X_test.index
    )

# ======= Duas colunas lado a lado =======
col_sup, col_unsup = st.columns(2)

# =========================
# Supervisionado (coluna esquerda)
# =========================
with col_sup:
    st.markdown("## 🎓 Aprendizado Supervisionado")

    st.markdown("### Escolha o modelo de classificação")
    model_name = st.selectbox(
        "Modelo",
        ["Regressão Logística", "Random Forest", "SVM (RBF)"]
    )

    if model_name == "Regressão Logística":
        C = st.slider("C (inverso da regularização)", 0.01, 5.0, 1.0, 0.01, key="logreg_C")
        clf = LogisticRegression(max_iter=500, C=C, random_state=seed)
    elif model_name == "Random Forest":
        n_estimators = st.slider("n_estimators", 50, 500, 200, 10, key="rf_n")
        max_depth = st.slider("max_depth", 1, 20, 8, 1, key="rf_d")
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=seed, n_jobs=-1
        )
    else:
        C = st.slider("C", 0.01, 5.0, 1.0, 0.01, key="svm_C")
        gamma = st.selectbox("gamma", ["scale", "auto"], key="svm_gamma")
        clf = SVC(C=C, gamma=gamma, probability=True, random_state=seed)

    # Treino
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    # Métricas
    metrics = compute_supervised_metrics(y_test, y_pred, y_proba)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with m2:
        st.metric("Recall", f"{metrics['recall']:.3f}")
        st.metric("F1-score", f"{metrics['f1']:.3f}")
    with m3:
        if metrics["roc_auc"] is not None:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        else:
            st.info("ROC AUC indisponível (modelo sem `predict_proba`).")

    # Matriz de confusão
    st.markdown("#### Matriz de confusão")
    cm_fig = plot_confusion_matrix_plotly(y_test, y_pred, ["Benigno (0)", "Maligno (1)"])
    st.plotly_chart(cm_fig, use_container_width=True)

    # Curva ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        roc_fig = px.area(
            roc_df, x="FPR", y="TPR",
            title=f"Curva ROC — AUC={metrics['roc_auc']:.3f}",
            labels={"FPR": "False Positive Rate", "TPR": "True Positive Rate"},
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    # Importâncias / Coeficientes
    st.markdown("#### Importância / Coeficientes dos atributos")
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig_imp = px.bar(imp.head(15), title="Top 15 importâncias (Random Forest)")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.dataframe(imp.rename("importância"))
    elif hasattr(clf, "coef_"):
        coefs = pd.Series(clf.coef_[0], index=X.columns).sort_values(key=lambda s: s.abs(), ascending=False)
        fig_coef = px.bar(coefs.head(15), title="Top 15 coeficientes (magnitude)")
        st.plotly_chart(fig_coef, use_container_width=True)
        st.dataframe(coefs.rename("coeficiente"))
    else:
        st.info("Este modelo não expõe importâncias/coeficientes diretamente.")

# =========================
# Não supervisionado (coluna direita)
# =========================
with col_unsup:
    st.markdown("## 🧩 Aprendizado Não Supervisionado")
    st.markdown("### K-Means + PCA (2D)")

    k = st.slider("Número de clusters (k)", 2, 6, 2, 1, key="k_unsup")
    pca = PCA(n_components=2, random_state=seed)
    X_all = X.copy()

    # Padroniza para PCA/KMeans se pedido
    X_scaled = X_all.copy()
    if scale_features:
        sc_all = StandardScaler().fit(X_all)
        X_scaled = pd.DataFrame(sc_all.transform(X_all), columns=X_all.columns, index=X_all.index)

    X_2d = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, clusters)
    st.metric("Silhouette score", f"{sil:.3f}")

    pca_df = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "Cluster": clusters.astype(str),  # manter como string para evitar mistura int/str
        "Alvo (target)": y.astype(int).map({0: "Benigno", 1: "Maligno"})
    })

    c1, c2 = st.columns(2)
    with c1:
        fig_c = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster",
            title="PCA colorido por **cluster**",
            hover_data=["Alvo (target)"]
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with c2:
        fig_t = px.scatter(
            pca_df, x="PC1", y="PC2", color="Alvo (target)",
            title="PCA colorido por **rótulo real**"
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # ---- Resumo por cluster (corrigido: índices como string) ----
    cluster_counts = (
        pd.Series(clusters).astype(str).value_counts().sort_index()
    )
    mal_rate = (
        pca_df.groupby("Cluster")["Alvo (target)"]
        .apply(lambda s: (s == "Maligno").mean() * 100)
        .sort_index()
    )
    cluster_summary = pd.DataFrame({
        "tam_cluster": cluster_counts,
        "taxa_maligno(%)": mal_rate
    })

    st.markdown("#### Medidas descritivas por cluster")
    st.dataframe(cluster_summary)

# =========================
# Relatório (abaixo das duas colunas)
# =========================
st.markdown("---")
st.markdown("### 📥 Relatório & Reprodutibilidade")
selected_sections = st.multiselect(
    "Selecione seções para incluir",
    ["Introdução", "Sobre o dataset", "Pré-processamento", "Resultados (supervisionado)",
     "Resultados (não supervisionado)", "Conclusões"],
    default=["Introdução", "Sobre o dataset", "Pré-processamento", "Resultados (supervisionado)", "Conclusões"]
)

report_md = build_md_report_supervised(
    selected_sections=selected_sections,
    metrics=metrics,
    model_name=model_name,
    test_size=test_size,
    scale_features=scale_features,
    seed=seed
)

b = io.BytesIO(report_md.encode("utf-8"))
st.download_button(
    "⬇️ Baixar relatório.md",
    data=b,
    file_name="relatorio_ml_saude.md",
    mime="text/markdown"
)

st.markdown("---")
st.caption("© 2025 — Projeto acadêmico. Dataset: Breast Cancer Wisconsin (sklearn).")
