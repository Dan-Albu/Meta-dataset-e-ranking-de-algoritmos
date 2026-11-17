"""
ANALISE ADICIONAL -- selecao de meta-atributos.

Nao e exigida pelo enunciado, mas responde a pergunta que os resultados
da Secao 2.1 deixam em aberto: a X estendida piorou o sistema, e o
enunciado atribui isso a dimensionalidade ("Seu meta-dataset tem poucas
dezenas de linhas... e exatamente o cenario em que a aula disse que a
selecao deixa de ser opcional"). Se essa explicacao estiver certa,
SELECIONAR um subconjunto das colunas deve recuperar -- ou superar -- o
desempenho da X base.

Tres estrategias de selecao, todas aplicadas a X estendida (38 colunas):

  topk_importancia : as k colunas de maior importancia media (Analise 1)
  antirredundancia : uma coluna por grupo de redundancia (Analise 2),
                     escolhendo a de maior importancia dentro do grupo
  so_novas         : apenas Grupo A + Grupo C, descartando a base --
                     testa se as novas SOZINHAS bastam

ARMADILHA A DECLARAR NO RELATORIO
---------------------------------
topk_importancia e antirredundancia usam a importancia/correlacao
calculadas sobre os 30 datasets, ou seja, incluindo o dataset que sera
deixado de fora no LODO. Isso e VAZAMENTO: o desempenho medido fica
otimista. A forma correta seria selecionar dentro de cada dobra do LODO,
usando apenas os 29 de treino.

A versao sem vazamento esta implementada em selecionar_dentro_do_lodo()
e e a que deve ser reportada. A versao com vazamento e mantida para
quantificar o tamanho do efeito -- a diferenca entre as duas e, em si,
um resultado interessante sobre meta-aprendizagem com poucas linhas.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

from evaluation import (
    spearman_por_abordagem, curva_de_perda, area_sob_curva,
    diagrama_diferenca_critica,
)
from harris import HarrisForest, normalize_performance_matrix
from analysis_etapa2 import (
    importancia_por_algoritmo, analisar_redundancia, origem_da_coluna,
    COLUNAS_BASE,
)

_SRC = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_SRC), "results")


# ----------------------------------------------------------------------
# Estrategias de selecao
# ----------------------------------------------------------------------

def selecionar_topk(X, P, k=8, random_state=42):
    """As k meta-caracteristicas de maior importancia media."""
    importancias = importancia_por_algoritmo(X, P, random_state=random_state)
    media = importancias.mean(axis=1).sort_values(ascending=False)
    return list(media.head(k).index)


def selecionar_antirredundancia(X, P, limiar=0.8, random_state=42):
    """
    Uma meta-caracteristica por grupo de redundancia.

    Agrupa por correlacao (Analise 2) e mantem, de cada grupo, a de maior
    importancia media (Analise 1). Reduz a dimensionalidade sem descartar
    nenhuma FAMILIA de informacao -- diferente do topk, que pode manter
    varias colunas do mesmo bloco redundante.
    """
    redundancia = analisar_redundancia(X, limiar=limiar)
    grupos = redundancia["grupos"]
    importancias = importancia_por_algoritmo(X, P, random_state=random_state)
    media = importancias.mean(axis=1)

    escolhidas = []
    for grupo in sorted(grupos.unique()):
        colunas = grupos[grupos == grupo].index
        escolhidas.append(media.loc[colunas].idxmax())
    return escolhidas


def selecionar_so_novas(X, P=None, **kwargs):
    """Apenas Grupo A + Grupo C, sem as 13 colunas da X base."""
    return [c for c in X.columns if origem_da_coluna(c) != "base"]


ESTRATEGIAS = {
    "topk8": lambda X, P: selecionar_topk(X, P, k=8),
    "topk13": lambda X, P: selecionar_topk(X, P, k=13),
    "antirredundancia": lambda X, P: selecionar_antirredundancia(X, P),
    "so_novas": selecionar_so_novas,
}


# ----------------------------------------------------------------------
# Execucao das abordagens
# ----------------------------------------------------------------------

def _rodar_com_colunas(X, colunas, P, R, lambda_harris=0.5,
                       n_trees=50, max_depth=4, random_state=42):
    """Roda A1, A2 e HARRIS restritos a um subconjunto de colunas."""
    X_sub = X[colunas].fillna(X[colunas].median())
    P_norm = normalize_performance_matrix(P.values)

    pred_p, pred_r, pred_h = [], [], []
    for treino, teste in LeaveOneOut().split(X_sub):
        escala = StandardScaler()
        X_treino = escala.fit_transform(X_sub.iloc[treino])
        X_teste = escala.transform(X_sub.iloc[teste])

        modelo = RandomForestRegressor(random_state=random_state)
        modelo.fit(X_treino, P.iloc[treino])
        pred_p.append(modelo.predict(X_teste)[0])

        modelo = RandomForestRegressor(random_state=random_state)
        modelo.fit(X_treino, R.iloc[treino])
        pred_r.append(modelo.predict(X_teste)[0])

        floresta = HarrisForest(
            n_trees=n_trees, lambda_param=lambda_harris,
            max_depth=max_depth, random_state=random_state,
        )
        floresta.fit(X_treino, P_norm[treino])
        pred_h.append(floresta.predict(X_teste)[0])

    em_ranking = lambda pred: pd.DataFrame(
        pred, index=P.index, columns=P.columns
    ).rank(axis=1, ascending=True, method="average")

    return {
        "A1": em_ranking(pred_p),
        "A2": em_ranking(pred_r),
        f"HARRIS(L={lambda_harris})": em_ranking(pred_h),
    }


def selecionar_dentro_do_lodo(X, P, R, k=8, lambda_harris=0.5,
                              n_trees=50, max_depth=4, random_state=42):
    """
    VERSAO SEM VAZAMENTO: a selecao e refeita em cada dobra do LODO,
    usando apenas os 29 datasets de treino.

    E mais cara (uma selecao por dobra) e mais pessimista que selecionar
    uma vez sobre os 30 -- mas e a unica que mede generalizacao de
    verdade. E esta que deve ser reportada como resultado.

    Returns
    -------
    (predicoes, colunas_por_dobra) : os rankings preditos e quais colunas
        foram escolhidas em cada dobra. A ESTABILIDADE dessa escolha e
        informativa: se cada dobra seleciona um conjunto diferente, a
        selecao esta capturando ruido, nao estrutura.
    """
    X_limpo = X.fillna(X.median())
    P_norm = normalize_performance_matrix(P.values)

    pred_p, pred_r, pred_h, colunas_por_dobra = [], [], [], []

    for treino, teste in LeaveOneOut().split(X_limpo):
        X_treino_completo = X_limpo.iloc[treino]
        P_treino = P.iloc[treino]

        # selecao usando SOMENTE os 29 de treino
        importancias = importancia_por_algoritmo(
            X_treino_completo, P_treino, random_state=random_state
        )
        colunas = list(
            importancias.mean(axis=1).sort_values(ascending=False).head(k).index
        )
        colunas_por_dobra.append(colunas)

        escala = StandardScaler()
        X_treino = escala.fit_transform(X_treino_completo[colunas])
        X_teste = escala.transform(X_limpo.iloc[teste][colunas])

        modelo = RandomForestRegressor(random_state=random_state)
        modelo.fit(X_treino, P_treino)
        pred_p.append(modelo.predict(X_teste)[0])

        modelo = RandomForestRegressor(random_state=random_state)
        modelo.fit(X_treino, R.iloc[treino])
        pred_r.append(modelo.predict(X_teste)[0])

        floresta = HarrisForest(
            n_trees=n_trees, lambda_param=lambda_harris,
            max_depth=max_depth, random_state=random_state,
        )
        floresta.fit(X_treino, P_norm[treino])
        pred_h.append(floresta.predict(X_teste)[0])

    em_ranking = lambda pred: pd.DataFrame(
        pred, index=P.index, columns=P.columns
    ).rank(axis=1, ascending=True, method="average")

    predicoes = {
        "A1": em_ranking(pred_p),
        "A2": em_ranking(pred_r),
        f"HARRIS(L={lambda_harris})": em_ranking(pred_h),
    }
    return predicoes, colunas_por_dobra


def estabilidade_da_selecao(colunas_por_dobra):
    """
    Com que frequencia cada meta-caracteristica e escolhida nas 30 dobras?

    Selecao estavel (sempre as mesmas colunas) indica estrutura real.
    Selecao instavel indica que a importancia esta capturando ruido -- o
    que seria evidencia direta a favor da explicacao de dimensionalidade.
    """
    contagem = {}
    for colunas in colunas_por_dobra:
        for coluna in colunas:
            contagem[coluna] = contagem.get(coluna, 0) + 1
    serie = pd.Series(contagem).sort_values(ascending=False)
    return (serie / len(colunas_por_dobra)).round(3)


if __name__ == "__main__":
    P = pd.read_csv(os.path.join(RESULTS, "P_matrix.csv"), index_col=0)
    R = pd.read_csv(os.path.join(RESULTS, "R_matrix.csv"), index_col=0)
    X_base = pd.read_csv(os.path.join(RESULTS, "X_matrix.csv"), index_col=0)
    X_est = pd.read_csv(os.path.join(RESULTS, "X_estendida.csv"), index_col=0)

    P_norm = pd.DataFrame(
        normalize_performance_matrix(P.values), index=P.index, columns=P.columns
    )

    resultados = {}

    print("Referencias (sem selecao)...")
    for nome, ranking in _rodar_com_colunas(
        X_base, list(X_base.columns), P, R).items():
        resultados[f"{nome}_base"] = ranking
    for nome, ranking in _rodar_com_colunas(
        X_est, list(X_est.columns), P, R).items():
        resultados[f"{nome}_est"] = ranking

    print("\nSelecao COM vazamento (para quantificar o efeito)...")
    for etiqueta, funcao in ESTRATEGIAS.items():
        colunas = funcao(X_est, P)
        print(f"  {etiqueta}: {len(colunas)} colunas -> {colunas[:6]}...")
        for nome, ranking in _rodar_com_colunas(X_est, colunas, P, R).items():
            resultados[f"{nome}_{etiqueta}"] = ranking

    print("\nSelecao SEM vazamento (dentro do LODO, k=8)...")
    predicoes, colunas_por_dobra = selecionar_dentro_do_lodo(X_est, P, R, k=8)
    for nome, ranking in predicoes.items():
        resultados[f"{nome}_lodo8"] = ranking

    estabilidade = estabilidade_da_selecao(colunas_por_dobra)
    print("\nEstabilidade da selecao (fracao das 30 dobras em que a coluna entrou):")
    print(estabilidade.head(12).to_string())
    estabilidade.to_csv(os.path.join(RESULTS, "estabilidade_selecao.csv"))

    spearman = spearman_por_abordagem(R, resultados)
    curva = curva_de_perda(P_norm, resultados, menor_e_melhor=True)
    areas = area_sob_curva(curva)

    resumo = pd.DataFrame({
        "spearman": spearman.mean(), "area": areas
    }).sort_values("spearman", ascending=False)
    resumo.to_csv(os.path.join(RESULTS, "resumo_selecao.csv"))

    print("\n=== RESUMO ===")
    print(resumo.round(4).to_string())

    print("\n=== POR ABORDAGEM: a selecao recuperou o desempenho? ===")
    for nome in ["A1", "A2", "HARRIS(L=0.5)"]:
        base = spearman[f"{nome}_base"].mean()
        est = spearman[f"{nome}_est"].mean()
        print(f"\n{nome}")
        print(f"  X base          {base:.4f}")
        print(f"  X estendida     {est:.4f}  ({est - base:+.4f})")
        for etiqueta in list(ESTRATEGIAS) + ["lodo8"]:
            chave = f"{nome}_{etiqueta}"
            if chave in spearman:
                valor = spearman[chave].mean()
                marca = " <-- supera a base" if valor > base else ""
                print(f"  {etiqueta:16} {valor:.4f}  ({valor - base:+.4f}){marca}")

    diagrama_diferenca_critica(
        spearman, os.path.join(RESULTS, "cd_selecao.png"))
