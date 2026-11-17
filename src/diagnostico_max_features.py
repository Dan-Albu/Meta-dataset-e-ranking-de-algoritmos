"""
DIAGNOSTICO -- por que a X estendida piorou o sistema?

A Analise 3 da Secao 2.1 mostrou que a X estendida (38 colunas) piorou as
tres abordagens em relacao a X base (13 colunas). O enunciado antecipa
esse resultado e o atribui a dimensionalidade. Este script investiga se a
causa e a dimensionalidade EM SI ou a CONFIGURACAO do meta-modelo.

A SUSPEITA
----------
O RandomForestRegressor do scikit-learn usa max_features=1.0 por padrao:
cada divisao avalia TODAS as colunas. Com 13 colunas, sao 13 candidatas
por split; com 38, sao 38. Como o nivel meta tem apenas 29 linhas de
treino, quase o triplo de candidatas significa quase o triplo de chances
de uma divisao espuria vencer por acaso.

Pior: com max_features=1.0 a floresta deixa de ser aleatoria no sentido
de Breiman. As arvores ficam correlacionadas entre si, e o ganho do
ensemble diminui -- exatamente quando a decorrelacao seria mais
necessaria.

Teste preliminar com a Abordagem 2 (Spearman medio):

    max_features    X base(13)   X estendida(38)
        1.0            0.7388        0.7200   (-0.0188)
        sqrt           0.7345        0.7521   (+0.0176)   <-- inverte
        0.33           0.7400        0.7289   (-0.0111)

Com sqrt, o sinal do efeito INVERTE: a X estendida passa a ser melhor.

NOTA SOBRE O HARRIS
-------------------
O HarrisForest ja usa max_features="sqrt" internamente (ver harris.py,
parametro da classe). Por isso foi a abordagem MENOS prejudicada pela
X estendida (-0,013 contra -0,120 da Abordagem 1). O problema esta
restrito as Abordagens 1 e 2, que usam RandomForestRegressor com o
padrao do scikit-learn.

Este script roda as tres abordagens sob as duas configuracoes, nas duas
matrizes, e produz a comparacao completa com teste estatistico.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

from evaluation import (
    spearman_por_abordagem, curva_de_perda, area_sob_curva,
    diagrama_diferenca_critica, plotar_curva_de_perda,
)
from harris import HarrisForest, normalize_performance_matrix

_SRC = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_SRC), "results")


def rodar_abordagens(X, P, R, max_features=1.0, lambda_harris=0.5,
                     n_trees=50, max_depth=4, random_state=42):
    """
    Roda A1, A2 e HARRIS com uma configuracao de max_features.

    Parameters
    ----------
    max_features : float, str ou None
        Repassado ao RandomForestRegressor das Abordagens 1 e 2 E ao
        HarrisForest. Valores uteis: 1.0 (padrao do sklearn, todas as
        colunas), "sqrt" (raiz do total, padrao do Breiman), 0.33.
    """
    X_limpo = X.fillna(X.median())
    P_norm = normalize_performance_matrix(P.values)

    pred_p, pred_r, pred_h = [], [], []
    for treino, teste in LeaveOneOut().split(X_limpo):
        escala = StandardScaler()
        X_treino = escala.fit_transform(X_limpo.iloc[treino])
        X_teste = escala.transform(X_limpo.iloc[teste])

        modelo = RandomForestRegressor(
            random_state=random_state, max_features=max_features)
        modelo.fit(X_treino, P.iloc[treino])
        pred_p.append(modelo.predict(X_teste)[0])

        modelo = RandomForestRegressor(
            random_state=random_state, max_features=max_features)
        modelo.fit(X_treino, R.iloc[treino])
        pred_r.append(modelo.predict(X_teste)[0])

        # O HarrisForest aceita "sqrt" ou um inteiro; converte a fracao.
        if isinstance(max_features, float):
            mf_harris = max(1, int(max_features * X_treino.shape[1]))
        else:
            mf_harris = max_features
        floresta = HarrisForest(
            n_trees=n_trees, lambda_param=lambda_harris, max_depth=max_depth,
            max_features=mf_harris, random_state=random_state,
        )
        floresta.fit(X_treino, P_norm[treino])
        pred_h.append(floresta.predict(X_teste)[0])

    em_ranking = lambda pred: pd.DataFrame(
        pred, index=P.index, columns=P.columns
    ).rank(axis=1, ascending=True, method="average")

    return {
        "A1": em_ranking(pred_p),
        "A2": em_ranking(pred_r),
        "HARRIS": em_ranking(pred_h),
    }


def comparar_sementes(X, P, R, max_features, sementes=(0, 1, 2, 3, 4),
                      lambda_harris=0.5):
    """
    Repete o experimento com varias sementes e devolve media e
    desvio-padrao do Spearman por abordagem.

    Motivacao: todo o restante do projeto roda com random_state=42 uma
    unica vez, sem estimativa de variancia. Diferencas da ordem de 0,02
    no Spearman podem nao sobreviver a troca de semente, e afirmar que
    uma configuracao supera outra exige saber isso.
    """
    acumulado = {}
    for semente in sementes:
        resultado = rodar_abordagens(
            X, P, R, max_features=max_features,
            lambda_harris=lambda_harris, random_state=semente,
        )
        spearman = spearman_por_abordagem(R, resultado).mean()
        for nome, valor in spearman.items():
            acumulado.setdefault(nome, []).append(valor)

    return pd.DataFrame({
        nome: {"media": np.mean(vals), "desvio": np.std(vals)}
        for nome, vals in acumulado.items()
    }).T


if __name__ == "__main__":
    P = pd.read_csv(os.path.join(RESULTS, "P_matrix.csv"), index_col=0)
    R = pd.read_csv(os.path.join(RESULTS, "R_matrix.csv"), index_col=0)
    X_base = pd.read_csv(os.path.join(RESULTS, "X_matrix.csv"), index_col=0)
    X_est = pd.read_csv(os.path.join(RESULTS, "X_estendida.csv"), index_col=0)
    P_norm = pd.DataFrame(
        normalize_performance_matrix(P.values), index=P.index, columns=P.columns)

    CONFIGS = {"padrao(1.0)": 1.0, "sqrt": "sqrt"}

    # ---------------- comparacao principal ----------------
    resultados = {}
    for etiqueta, mf in CONFIGS.items():
        for nome_x, X in [("base", X_base), ("est", X_est)]:
            print(f"Rodando {nome_x} com max_features={etiqueta}...")
            for abordagem, ranking in rodar_abordagens(
                    X, P, R, max_features=mf).items():
                resultados[f"{abordagem}_{nome_x}_{etiqueta}"] = ranking

    spearman = spearman_por_abordagem(R, resultados)
    curva = curva_de_perda(P_norm, resultados, menor_e_melhor=True)
    areas = area_sob_curva(curva)

    resumo = pd.DataFrame({"spearman": spearman.mean(), "area": areas})
    resumo.to_csv(os.path.join(RESULTS, "diagnostico_max_features.csv"))

    print("\n=== EFEITO DA X ESTENDIDA SOB CADA CONFIGURACAO ===")
    print(f"{'abordagem':>10} {'config':>12} {'base':>9} {'estendida':>10} "
          f"{'diferenca':>11}")
    for abordagem in ["A1", "A2", "HARRIS"]:
        for etiqueta in CONFIGS:
            base = spearman[f"{abordagem}_base_{etiqueta}"].mean()
            est = spearman[f"{abordagem}_est_{etiqueta}"].mean()
            marca = "  <-- inverte" if est > base else ""
            print(f"{abordagem:>10} {etiqueta:>12} {base:>9.4f} "
                  f"{est:>10.4f} {est - base:>+11.4f}{marca}")

    print("\n=== MELHOR CONFIGURACAO GLOBAL ===")
    print(resumo.sort_values("spearman", ascending=False).head(6).round(4).to_string())

    # ---------------- estabilidade entre sementes ----------------
    print("\n=== VARIANCIA ENTRE SEMENTES (5 repeticoes) ===")
    print("Verifica se as diferencas sobrevivem a troca de semente.")
    for etiqueta, mf in CONFIGS.items():
        for nome_x, X in [("base", X_base), ("est", X_est)]:
            tabela = comparar_sementes(X, P, R, max_features=mf)
            print(f"\nX {nome_x}, max_features={etiqueta}:")
            print(tabela.round(4).to_string())

    plotar_curva_de_perda(
        curva, os.path.join(RESULTS, "curva_diagnostico.png"))
    diagrama_diferenca_critica(
        spearman, os.path.join(RESULTS, "cd_diagnostico.png"))
