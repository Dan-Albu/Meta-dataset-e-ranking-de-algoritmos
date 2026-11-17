"""
ETAPA 2 da Atividade Pratica 2 -- as tres analises da Secao 2.1.

  1. Relacao com o meta-alvo ..... importancia de cada meta-caracteristica
  2. Redundancia ................. as novas colunas repetem as classicas?
  3. Efeito no sistema ........... X_base contra X_estendida, mesmo
                                   protocolo LODO e mesmas metricas

Pre-requisitos (todos ja produzidos por etapas anteriores):
    results/P_matrix.csv        matriz de performances (Atividade 1)
    results/R_matrix.csv        matriz de ranks (Atividade 1)
    results/X_matrix.csv        X base, 13 colunas (Atividade 1)
    results/X_estendida.csv     X estendida, 38 colunas (build_extended_features.py)

A matriz P NAO muda entre as duas condicoes -- e o que permite atribuir
qualquer diferenca observada as meta-caracteristicas, conforme o enunciado
exige.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.base import clone

from evaluation import (
    spearman_por_abordagem, curva_de_perda, area_sob_curva,
    diagrama_diferenca_critica, plotar_curva_de_perda,
)
from harris import HarrisForest, normalize_performance_matrix

_SRC = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_SRC), "results")

# Colunas por origem -- usado para separar "classicas" de "novas" nas
# analises de importancia e redundancia.
COLUNAS_BASE = [
    "nr_inst", "nr_attr", "inst_to_attr",
    "mean_of_means", "mean_of_stds", "mean_skewness", "mean_kurtosis",
    "mean_abs_correlation", "prop_high_correlation",
    "target_skewness", "target_cv", "mean_relevance", "prop_rare_instances",
]
COLUNAS_GRUPO_A = ["C1", "C2", "C3", "C4", "L1", "L2", "L3",
                   "S1", "S2", "S3", "S4", "T2"]


def origem_da_coluna(coluna):
    """Classifica cada meta-caracteristica pela origem, para os graficos."""
    if coluna in COLUNAS_BASE:
        return "base"
    if coluna in COLUNAS_GRUPO_A:
        return "complexidade (Grupo A)"
    return "por regiao (Grupo C)"


# ======================================================================
# ANALISE 1 -- relacao com o meta-alvo
# ======================================================================

def importancia_por_algoritmo(X, P, random_state=42):
    """
    ANALISE 1: treina uma floresta por ALGORITMO BASE, prevendo a
    performance daquele algoritmo a partir das meta-caracteristicas, e
    extrai o vetor de importancias.

    Por que uma floresta por algoritmo, e nao uma multi-saida: o
    enunciado pede explicitamente para examinar "se o perfil de
    importancia muda de um algoritmo para outro, como em De Amorim et al.
    (2024), onde cada classificador base depende de familias diferentes
    de meta-atributos". Um modelo multi-saida devolveria um unico vetor
    e apagaria justamente essa diferenca.

    Returns
    -------
    pd.DataFrame : linhas = meta-caracteristicas, colunas = algoritmos,
        valores = importancia (Gini) daquela meta-caracteristica para
        prever a performance daquele algoritmo.
    """
    X_limpo = X.fillna(X.median())
    escalado = StandardScaler().fit_transform(X_limpo)

    importancias = {}
    for algoritmo in P.columns:
        modelo = RandomForestRegressor(
            n_estimators=500, random_state=random_state
        )
        modelo.fit(escalado, P[algoritmo].to_numpy())
        importancias[algoritmo] = modelo.feature_importances_

    return pd.DataFrame(importancias, index=X_limpo.columns)


def plotar_importancias(df_importancias, caminho, titulo, top=15):
    """
    Grafico de barras horizontais das meta-caracteristicas mais
    importantes (media entre algoritmos), coloridas por origem.
    """
    media = df_importancias.mean(axis=1).sort_values(ascending=False).head(top)
    cores = {"base": "#5c6673",
             "complexidade (Grupo A)": "#b8562f",
             "por regiao (Grupo C)": "#2f5d6b"}

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(media) + 1.5))
    ax.barh(
        range(len(media)), media.to_numpy()[::-1],
        color=[cores[origem_da_coluna(c)] for c in media.index[::-1]],
    )
    ax.set_yticks(range(len(media)))
    ax.set_yticklabels(media.index[::-1], fontsize=9)
    ax.set_xlabel("Importancia media (Gini)")
    ax.set_title(titulo)
    manipuladores = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cores.values()]
    ax.legend(manipuladores, cores.keys(), fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  grafico de importancias salvo em {caminho}")


# ======================================================================
# ANALISE 2 -- redundancia
# ======================================================================

def analisar_redundancia(X, limiar=0.8):
    """
    ANALISE 2: as novas colunas trazem informacao nova, ou repetem o que
    as classicas ja diziam?

    Instrumento escolhido: correlacao de Spearman entre pares de
    meta-caracteristicas, seguida de agrupamento hierarquico das
    redundantes. Spearman (e nao Pearson) porque as meta-caracteristicas
    tem escalas e distribuicoes muito diferentes entre si, e o que
    interessa e a associacao monotonica.

    Parameters
    ----------
    limiar : float
        |correlacao| acima da qual duas meta-caracteristicas sao
        consideradas redundantes.

    Returns
    -------
    dict com:
      "correlacao"  : matriz de correlacao absoluta
      "pares"       : pares redundantes, ordenados por correlacao
      "grupos"      : agrupamento das redundantes
      "resumo"      : quantas novas sao redundantes com alguma classica
    """
    X_limpo = X.fillna(X.median())
    correlacao = X_limpo.corr(method="spearman").abs()

    # Pares redundantes (so triangulo superior, sem a diagonal)
    mascara = np.triu(np.ones(correlacao.shape, dtype=bool), k=1)
    pares = (
        correlacao.where(mascara)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "a", "level_1": "b", 0: "correlacao"})
    )
    pares = pares[pares["correlacao"] >= limiar].sort_values(
        "correlacao", ascending=False
    )
    pares["origem_a"] = pares["a"].map(origem_da_coluna)
    pares["origem_b"] = pares["b"].map(origem_da_coluna)

    # Agrupamento hierarquico: distancia = 1 - |correlacao|
    distancia = (1 - correlacao).to_numpy(copy=True)
    np.fill_diagonal(distancia, 0)
    ligacao = linkage(squareform(distancia, checks=False), method="average")
    rotulos = fcluster(ligacao, t=1 - limiar, criterion="distance")
    grupos = pd.Series(rotulos, index=correlacao.index, name="grupo")

    # Quantas NOVAS sao redundantes com alguma CLASSICA (base)?
    novas = [c for c in X_limpo.columns if origem_da_coluna(c) != "base"]
    redundantes_com_base = []
    for coluna in novas:
        maxima = correlacao.loc[coluna, COLUNAS_BASE].max()
        if maxima >= limiar:
            parceira = correlacao.loc[coluna, COLUNAS_BASE].idxmax()
            redundantes_com_base.append(
                {"nova": coluna, "classica": parceira,
                 "correlacao": round(maxima, 3)}
            )

    resumo = {
        "total_novas": len(novas),
        "novas_redundantes_com_base": len(redundantes_com_base),
        "grupos_encontrados": int(grupos.nunique()),
        "colunas_totais": len(X_limpo.columns),
    }

    return {
        "correlacao": correlacao,
        "pares": pares,
        "grupos": grupos,
        "redundantes_com_base": pd.DataFrame(redundantes_com_base),
        "resumo": resumo,
    }


def plotar_correlacao(correlacao, caminho):
    """Mapa de calor da correlacao absoluta entre meta-caracteristicas."""
    fig, ax = plt.subplots(figsize=(11, 9))
    imagem = ax.imshow(correlacao.values, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(correlacao)))
    ax.set_yticks(range(len(correlacao)))
    ax.set_xticklabels(correlacao.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(correlacao.index, fontsize=7)
    plt.colorbar(imagem, ax=ax, label="|correlacao de Spearman|")
    ax.set_title("Redundancia entre meta-caracteristicas")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  mapa de correlacao salvo em {caminho}")


# ======================================================================
# ANALISE 3 -- efeito no sistema de recomendacao
# ======================================================================

def _predizer_lodo(X_meta, alvo, random_state=42):
    """LODO no nivel meta -- identico ao usado na Atividade 1."""
    X_limpo = X_meta.fillna(X_meta.median())
    predicoes = []
    for treino, teste in LeaveOneOut().split(X_limpo):
        escala = StandardScaler()
        X_treino = escala.fit_transform(X_limpo.iloc[treino])
        X_teste = escala.transform(X_limpo.iloc[teste])
        modelo = RandomForestRegressor(random_state=random_state)
        modelo.fit(X_treino, alvo.iloc[treino])
        predicoes.append(modelo.predict(X_teste)[0])
    return pd.DataFrame(predicoes, index=alvo.index, columns=alvo.columns)


def rodar_abordagens(X_meta, P, R, lambda_harris=0.5,
                     n_trees=50, max_depth=4, random_state=42):
    """
    Reexecuta as tres abordagens que USAM X (Abordagem 1, Abordagem 2 e
    HARRIS) com uma dada matriz de meta-caracteristicas.

    As abordagens de agregacao (AR, MR, SW) nao usam X -- seus resultados
    sao identicos nas duas condicoes, e por isso nao sao reexecutadas
    aqui. Elas entram na comparacao final como referencia fixa.

    Returns
    -------
    dict {nome_abordagem: ranking predito}
    """
    X_limpo = X_meta.fillna(X_meta.median())

    # Abordagem 1: preve P, converte em ranking depois
    P_predita = _predizer_lodo(X_limpo, P, random_state)
    R_a1 = P_predita.rank(axis=1, ascending=True, method="average")

    # Abordagem 2: preve R diretamente
    R_bruta = _predizer_lodo(X_limpo, R, random_state)
    R_a2 = R_bruta.rank(axis=1, ascending=True, method="average")

    # HARRIS: exige P normalizada por linha
    P_norm = normalize_performance_matrix(P.values)
    predicoes = []
    for treino, teste in LeaveOneOut().split(X_limpo):
        escala = StandardScaler()
        X_treino = escala.fit_transform(X_limpo.iloc[treino])
        X_teste = escala.transform(X_limpo.iloc[teste])
        floresta = HarrisForest(
            n_trees=n_trees, lambda_param=lambda_harris,
            max_depth=max_depth, random_state=random_state,
        )
        floresta.fit(X_treino, P_norm[treino])
        predicoes.append(floresta.predict(X_teste)[0])
    R_harris = pd.DataFrame(
        predicoes, index=P.index, columns=P.columns
    ).rank(axis=1, ascending=True, method="average")

    return {"A1": R_a1, "A2": R_a2, f"HARRIS(L={lambda_harris})": R_harris}


if __name__ == "__main__":
    P = pd.read_csv(os.path.join(RESULTS, "P_matrix.csv"), index_col=0)
    R = pd.read_csv(os.path.join(RESULTS, "R_matrix.csv"), index_col=0)
    X_base = pd.read_csv(os.path.join(RESULTS, "X_matrix.csv"), index_col=0)
    X_est = pd.read_csv(os.path.join(RESULTS, "X_estendida.csv"), index_col=0)

    print(f"X base:      {X_base.shape[0]} x {X_base.shape[1]}")
    print(f"X estendida: {X_est.shape[0]} x {X_est.shape[1]}")

    # ---------------- ANALISE 1 ----------------
    print("\n=== ANALISE 1: relacao com o meta-alvo ===")
    importancias = importancia_por_algoritmo(X_est, P)
    importancias.to_csv(os.path.join(RESULTS, "importancia_meta_features.csv"))

    media = importancias.mean(axis=1).sort_values(ascending=False)
    print("\nTop 10 meta-caracteristicas (importancia media):")
    for coluna in media.head(10).index:
        print(f"  {coluna:24} {media[coluna]:.4f}  [{origem_da_coluna(coluna)}]")

    por_origem = media.groupby(
        [origem_da_coluna(c) for c in media.index]
    ).sum()
    print("\nImportancia total por origem:")
    print(por_origem.sort_values(ascending=False).round(4).to_string())

    print("\nMeta-caracteristica mais importante para CADA algoritmo:")
    print("(se o perfil muda entre algoritmos, confirma De Amorim et al. 2024)")
    for algoritmo in importancias.columns:
        topo = importancias[algoritmo].idxmax()
        print(f"  {algoritmo:5} -> {topo:24} ({importancias.loc[topo, algoritmo]:.4f})")

    plotar_importancias(
        importancias,
        os.path.join(RESULTS, "importancia_meta_features.png"),
        "Importancia das meta-caracteristicas (media entre algoritmos)",
    )

    # ---------------- ANALISE 2 ----------------
    print("\n=== ANALISE 2: redundancia ===")
    redundancia = analisar_redundancia(X_est, limiar=0.8)
    print(f"\nResumo: {redundancia['resumo']}")

    if len(redundancia["redundantes_com_base"]):
        print("\nNovas meta-caracteristicas redundantes com alguma classica:")
        print(redundancia["redundantes_com_base"].to_string(index=False))
    else:
        print("\nNenhuma nova meta-caracteristica e redundante com as classicas.")

    print(f"\nPares mais redundantes (|rho| >= 0.8): "
          f"{len(redundancia['pares'])} pares")
    print(redundancia["pares"].head(12).to_string(index=False))

    redundancia["correlacao"].to_csv(
        os.path.join(RESULTS, "correlacao_meta_features.csv"))
    redundancia["pares"].to_csv(
        os.path.join(RESULTS, "pares_redundantes.csv"), index=False)
    plotar_correlacao(
        redundancia["correlacao"],
        os.path.join(RESULTS, "correlacao_meta_features.png"))

    # ---------------- ANALISE 3 ----------------
    print("\n=== ANALISE 3: efeito no sistema de recomendacao ===")
    print("Rodando abordagens com X base...")
    resultado_base = rodar_abordagens(X_base, P, R)
    print("Rodando abordagens com X estendida...")
    resultado_est = rodar_abordagens(X_est, P, R)

    # Agregacoes nao usam X -- entram como referencia fixa
    agregados = pd.read_csv(
        os.path.join(RESULTS, "rankings_agregados.csv"), index_col=0)

    comparacao = {}
    for nome, ranking in resultado_base.items():
        comparacao[f"{nome}_base"] = ranking
    for nome, ranking in resultado_est.items():
        comparacao[f"{nome}_est"] = ranking

    spearman = spearman_por_abordagem(R, comparacao)
    spearman.to_csv(os.path.join(RESULTS, "spearman_base_vs_estendida.csv"))

    P_norm = pd.DataFrame(
        normalize_performance_matrix(P.values), index=P.index, columns=P.columns)
    curva = curva_de_perda(P_norm, comparacao, menor_e_melhor=True)
    curva.to_csv(os.path.join(RESULTS, "curva_base_vs_estendida.csv"))
    areas = area_sob_curva(curva)

    resumo = pd.DataFrame({
        "spearman": spearman.mean(),
        "area_curva_perda": areas,
    })
    print("\nComparacao X base contra X estendida:")
    print(resumo.round(4).to_string())

    print("\nDiferenca (estendida - base), por abordagem:")
    for nome in resultado_base:
        ds = spearman[f"{nome}_est"].mean() - spearman[f"{nome}_base"].mean()
        da = areas[f"{nome}_est"] - areas[f"{nome}_base"]
        sinal_s = "melhorou" if ds > 0 else "piorou"
        sinal_a = "melhorou" if da < 0 else "piorou"
        print(f"  {nome:18} Spearman {ds:+.4f} ({sinal_s}) | "
              f"area {da:+.4f} ({sinal_a})")

    plotar_curva_de_perda(
        curva, os.path.join(RESULTS, "curva_base_vs_estendida.png"))
    diagrama_diferenca_critica(
        spearman,
        os.path.join(RESULTS, "cd_base_vs_estendida.png"))
