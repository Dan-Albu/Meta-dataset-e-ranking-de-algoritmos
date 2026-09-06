"""
SECAO 3 da atividade: avaliacao experimental.

Compara as seis abordagens (AR, MR, SW, Abordagem 1, Abordagem 2, HARRIS)
em modo leave-one-dataset-out, com as tres metricas que o enunciado pede:

  1. Correlacao de Spearman entre o ranking recomendado e o verdadeiro
     -> spearman_por_abordagem()
  2. Curva de perda por numero de testes + area sob a curva
     -> curva_de_perda() e area_sob_curva()
  3. Grafico de diferenca critica (Friedman + pos-teste de Nemenyi)
     -> diagrama_diferenca_critica()

Base teorica: Cap. 3 do livro (Brazdil et al.), Secoes 3.3 (normalizacao),
3.4 (correlacao e curvas de perda) e 3.5 (comparacao com testes
estatisticos).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem janela grafica -- salva direto em arquivo
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, friedmanchisquare
import scikit_posthocs as sp


def spearman_por_abordagem(R_verdadeira, rankings_preditos):
    """
    METRICA 1: correlacao de Spearman entre o ranking recomendado por cada
    abordagem e o ranking verdadeiro de cada dataset.

    Parameters
    ----------
    R_verdadeira : pd.DataFrame
        A matriz R real (datasets x algoritmos).
    rankings_preditos : dict {nome_abordagem: ranking}
        Para abordagens TREINADAS: um pd.DataFrame (um ranking por dataset).
        Para abordagens de AGREGACAO: um pd.Series (o mesmo ranking fixo
        para todos os datasets) -- a funcao detecta e trata os dois casos.

    Returns
    -------
    pd.DataFrame : linhas = datasets, colunas = abordagens, valores =
        correlacao de Spearman naquele dataset.
    """
    colunas = R_verdadeira.columns
    resultados = {}

    for nome, predito in rankings_preditos.items():
        correlacoes = []
        for ds in R_verdadeira.index:
            if isinstance(predito, pd.Series):
                # agregacao: ranking fixo, igual para todo dataset
                ranking_recomendado = predito[colunas]
            else:
                ranking_recomendado = predito.loc[ds]
            correlacoes.append(
                spearmanr(R_verdadeira.loc[ds], ranking_recomendado).statistic
            )
        resultados[nome] = correlacoes

    return pd.DataFrame(resultados, index=R_verdadeira.index)


def curva_de_perda(P_normalizada, rankings_preditos, menor_e_melhor=True):
    """
    METRICA 2: curva de perda por numero de testes.

    A ideia (Cap. 3 do livro): o sistema de metalearning devolve uma ORDEM
    de teste. Voce testa os algoritmos nessa ordem e, a cada teste, mede o
    quao longe ainda esta do melhor algoritmo possivel naquele dataset.

    Formula do enunciado, para metricas em que MAIOR e melhor:
        perda_i(t) = max_j P_ij - max_{a in primeiros t} P_ia

    Como a SERA e uma metrica de ERRO (menor = melhor), invertemos a ordem
    da diferenca, exatamente como o enunciado instrui:
        perda_i(t) = min_{a in primeiros t} P_ia - min_j P_ij

    A perda cai monotonicamente e chega a zero em t = numero de algoritmos
    (testou todos, entao achou o melhor). Um bom metodo faz a curva cair
    rapido -- idealmente ja em t=1.

    IMPORTANTE: use a matriz P NORMALIZADA por linha. Sem isso, um dataset
    de escala gigante (california ~1e12) domina completamente a media entre
    datasets, e a curva perde significado. O Cap. 3 (Secao 3.3) lista essa
    normalizacao 0-1 como transformacao padrao justamente para permitir
    comparacoes entre datasets.

    Parameters
    ----------
    P_normalizada : pd.DataFrame
        Performances normalizadas por linha para [0, 1].
    rankings_preditos : dict {nome_abordagem: ranking}
        Mesmo formato de spearman_por_abordagem().
    menor_e_melhor : bool
        True para metricas de erro (SERA, RMSE); False para metricas em
        que maior e melhor (acuracia, F1).

    Returns
    -------
    pd.DataFrame : linhas = t (1..m), colunas = abordagens, valores =
        perda MEDIA sobre todos os datasets apos t testes.
    """
    algoritmos = list(P_normalizada.columns)
    m = len(algoritmos)
    resultados = {}

    for nome, predito in rankings_preditos.items():
        perdas_por_t = np.zeros(m)

        for ds in P_normalizada.index:
            performances = P_normalizada.loc[ds]

            if isinstance(predito, pd.Series):
                ranking_recomendado = predito[algoritmos]
            else:
                ranking_recomendado = predito.loc[ds]

            # ordem de teste = algoritmos ordenados pela posicao recomendada
            ordem = ranking_recomendado.sort_values().index.tolist()

            if menor_e_melhor:
                melhor_possivel = performances.min()
            else:
                melhor_possivel = performances.max()

            for t in range(1, m + 1):
                testados = ordem[:t]
                if menor_e_melhor:
                    melhor_ate_agora = performances[testados].min()
                    perda = melhor_ate_agora - melhor_possivel
                else:
                    melhor_ate_agora = performances[testados].max()
                    perda = melhor_possivel - melhor_ate_agora
                perdas_por_t[t - 1] += perda

        resultados[nome] = perdas_por_t / len(P_normalizada)

    return pd.DataFrame(resultados, index=range(1, m + 1))


def area_sob_curva(curva):
    """
    Area sob a curva de perda, calculada como o enunciado define: "a media
    das perdas para t = 1, ..., m".

    Interpretacao: quanto MENOR, melhor -- significa que a abordagem chega
    perto do melhor algoritmo com menos testes.

    Returns
    -------
    pd.Series : area (media das perdas) por abordagem.
    """
    return curva.mean(axis=0)


def diagrama_diferenca_critica(spearman_df, caminho_saida, alpha=0.05):
    """
    METRICA 3: teste de Friedman seguido do pos-teste de Nemenyi,
    apresentado num diagrama de diferenca critica (Cap. 3, Secao 3.5).

    Como ler o diagrama: cada abordagem aparece posicionada pelo seu rank
    medio (mais a esquerda = melhor). Abordagens ligadas por uma barra
    horizontal NAO tem diferenca estatisticamente significativa entre si.

    Sequencia estatistica correta (e por que ela importa):
      1. Friedman testa a hipotese nula de que TODAS as abordagens sao
         equivalentes. Se p >= alpha, paramos: nao ha evidencia de
         diferenca, e comparacoes par a par nao se justificam.
      2. So se Friedman rejeitar, aplicamos Nemenyi para descobrir QUAIS
         pares diferem. Fazer o pos-teste sem o teste global aumenta a
         chance de falsos positivos por multiplas comparacoes.

    Parameters
    ----------
    spearman_df : pd.DataFrame
        Saida de spearman_por_abordagem() (datasets x abordagens).
    caminho_saida : str
        Onde salvar a figura .png.
    alpha : float
        Nivel de significancia.

    Returns
    -------
    dict com 'friedman_p', 'significativo' (bool) e 'ranks_medios'.
    """
    # Rank medio de cada abordagem por dataset (1 = melhor Spearman).
    # ascending=False porque Spearman ALTO e melhor.
    ranks = spearman_df.rank(axis=1, ascending=False)
    ranks_medios = ranks.mean(axis=0)

    estatistica, p_valor = friedmanchisquare(
        *[spearman_df[c] for c in spearman_df.columns]
    )
    significativo = p_valor < alpha

    print(f"Teste de Friedman: estatistica={estatistica:.4f}, p={p_valor:.4f}")
    if significativo:
        print(f"  p < {alpha}: ha diferenca significativa entre as abordagens.")
        print("  Aplicando o pos-teste de Nemenyi para identificar quais pares.")
    else:
        print(f"  p >= {alpha}: NAO ha evidencia de diferenca entre as")
        print("  abordagens. O diagrama abaixo vai mostrar todas conectadas")
        print("  -- o que e, em si, um resultado legitimo a reportar.")

    matriz_nemenyi = sp.posthoc_nemenyi_friedman(spearman_df.values)
    matriz_nemenyi.index = spearman_df.columns
    matriz_nemenyi.columns = spearman_df.columns

    plt.figure(figsize=(10, 3))
    sp.critical_difference_diagram(ranks_medios, matriz_nemenyi)
    plt.title(
        f"Diagrama de diferenca critica -- Friedman p={p_valor:.4f}"
        f" ({'significativo' if significativo else 'nao significativo'})"
    )
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Diagrama salvo em {caminho_saida}")

    return {
        "friedman_p": p_valor,
        "significativo": significativo,
        "ranks_medios": ranks_medios,
    }


def plotar_curva_de_perda(curva, caminho_saida):
    """
    Desenha a curva de perda media de todas as abordagens num unico
    grafico, para comparacao visual.
    """
    plt.figure(figsize=(8, 5))
    for coluna in curva.columns:
        plt.plot(curva.index, curva[coluna], marker="o", label=coluna)
    plt.xlabel("Numero de algoritmos testados (t)")
    plt.ylabel("Perda media (P normalizada)")
    plt.title("Curva de perda por numero de testes")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.xticks(curva.index)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curva de perda salva em {caminho_saida}")
