"""
ETAPA 2 da Atividade Pratica 2 -- construcao da matriz X estendida.

Roda uma vez e salva em results/:
    X_estendida.csv        as tres origens juntas (base + Grupo A + Grupo C)
    custo_extracao.csv     tempo de extracao por dataset e por grupo

A matriz X BASE (results/X_matrix.csv) permanece intocada -- ela e o
X_base da comparacao exigida pela Secao 2.1, item 3 do enunciado.
A matriz P tambem nao e tocada: o enunciado exige que ela e o protocolo
permanecam identicos aos da Atividade 1, para que qualquer diferenca
observada seja atribuivel as novas meta-caracteristicas.

ORIGENS DA MATRIZ ESTENDIDA
  X base (13 colunas) ... ja calculada na Atividade 1: gerais/dimensao,
                          estatisticas/derivadas e especificas do dominio.
  Grupo A (12 colunas) .. medidas de complexidade para regressao de
                          Lorena et al. (2018) -- familia "conceito e
                          complexidade", ausente do X base.
  Grupo C (13 colunas) .. decomposicao por regiao de relevancia (recorte
                          proposto): 6 medidas restritas a regiao rara,
                          6 razoes rara/global, e n_raras.

Total: 38 colunas para 30 linhas. O enunciado alerta que esse e
exatamente o cenario em que a selecao de meta-atributos deixa de ser
opcional -- ponto a discutir na interpretacao.
"""

import os
import time
import numpy as np
import pandas as pd

from metrics import relevance, LIMIAR_RARO
from complexity import (
    extrair_complexidade,
    extrair_complexidade_por_regiao,
    MEDIDAS,
)

_SRC = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_SRC)
DATA_FOLDER = os.path.join(_RAIZ, "data")
RESULTS_FOLDER = os.path.join(_RAIZ, "results")

# Medidas do Grupo A calculadas tambem por regiao (Grupo C). Subconjunto
# das 12: as que fazem sentido restritas a um subconjunto de instancias e
# que representam as tres familias (correlacao, linearidade, suavidade).
MEDIDAS_POR_REGIAO = ("C1", "C2", "L1", "L2", "S1", "S3")

# Datasets acima deste tamanho sao subamostrados nas medidas O(n^2).
# Ver nota sobre custo no relatorio.
MAX_AMOSTRAS = 2000


def carregar_datasets(pasta):
    """Mesma convencao da Atividade 1: o alvo e a PRIMEIRA coluna."""
    datasets = []
    for arquivo in sorted(os.listdir(pasta)):
        if not arquivo.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(pasta, arquivo))
        y = df.iloc[:, 0].to_numpy(dtype=float)
        X = df.iloc[:, 1:].to_numpy(dtype=float)
        datasets.append((arquivo.replace(".csv", ""), X, y))
    return datasets


def construir_x_estendida(pasta_dados=None, pasta_resultados=None):
    """
    Extrai Grupo A e Grupo C de cada dataset, concatena com a X base e
    registra o tempo gasto em cada etapa.

    Returns
    -------
    (X_estendida, custos) : dois DataFrames.
    """
    pasta_dados = pasta_dados or DATA_FOLDER
    pasta_resultados = pasta_resultados or RESULTS_FOLDER

    X_base = pd.read_csv(
        os.path.join(pasta_resultados, "X_matrix.csv"), index_col=0
    )

    linhas_a, linhas_c, custos = {}, {}, {}

    for nome, X, y in carregar_datasets(pasta_dados):
        print(f"Extraindo de {nome} (n={len(y)}, d={X.shape[1]})...")

        t0 = time.time()
        grupo_a = extrair_complexidade(X, y, max_amostras=MAX_AMOSTRAS)
        t_a = time.time() - t0

        t0 = time.time()
        phi = relevance(y, y_reference=y)
        grupo_c = extrair_complexidade_por_regiao(
            X, y, phi,
            limiar_phi=LIMIAR_RARO,
            medidas_regiao=MEDIDAS_POR_REGIAO,
            max_amostras=MAX_AMOSTRAS,
            complexidade_global={k: grupo_a[k] for k in MEDIDAS_POR_REGIAO},
        )
        t_c = time.time() - t0

        linhas_a[nome] = grupo_a
        linhas_c[nome] = grupo_c
        custos[nome] = {
            "n": len(y),
            "d": X.shape[1],
            "segundos_grupo_A": round(t_a, 2),
            "segundos_grupo_C": round(t_c, 2),
            "segundos_total": round(t_a + t_c, 2),
            "subamostrado": len(y) > MAX_AMOSTRAS,
        }
        print(f"    grupo A: {t_a:.1f}s | grupo C: {t_c:.1f}s")

    df_a = pd.DataFrame(linhas_a).T
    df_c = pd.DataFrame(linhas_c).T
    df_custos = pd.DataFrame(custos).T

    # Alinha pelo indice da X base (mesma ordem de datasets)
    X_estendida = pd.concat(
        [X_base, df_a.loc[X_base.index], df_c.loc[X_base.index]], axis=1
    )

    return X_estendida, df_custos


if __name__ == "__main__":
    X_est, custos = construir_x_estendida()

    X_est.to_csv(os.path.join(RESULTS_FOLDER, "X_estendida.csv"))
    custos.to_csv(os.path.join(RESULTS_FOLDER, "custo_extracao.csv"))

    print(f"\nX estendida: {X_est.shape[0]} linhas x {X_est.shape[1]} colunas")
    print(f"  (X base tinha {X_est.shape[1] - len(MEDIDAS) - len(MEDIDAS_POR_REGIAO) * 2 - 1} colunas)")

    print("\nCusto de extracao:")
    print(f"  total: {custos['segundos_total'].sum():.1f}s")
    print(f"  mediana por dataset: {custos['segundos_total'].median():.1f}s")
    print(f"  mais caro: {custos['segundos_total'].idxmax()} "
          f"({custos['segundos_total'].max():.1f}s)")
    print(f"  datasets subamostrados: {custos['subamostrado'].sum()} de {len(custos)}")

    # Colunas com valores faltantes -- tipicamente datasets sem regiao
    # rara suficiente para o Grupo C. Quantos sao e em si um achado.
    faltantes = X_est.isna().sum()
    if faltantes.any():
        print("\nColunas com valores faltantes (datasets sem regiao rara):")
        print(faltantes[faltantes > 0].to_string())
