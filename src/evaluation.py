import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem janela grafica -- salva direto em arquivo
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, friedmanchisquare
import scikit_posthocs as sp


def spearman_por_abordagem(R_verdadeira, rankings_preditos):

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
    return curva.mean(axis=0)


def diagrama_diferenca_critica(spearman_df, caminho_saida, alpha=0.05):
 
    # Rank medio de cada abordagem por dataset (1 = melhor Spearman). ---ascending=False porque Spearman ALTO e melhor.
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
