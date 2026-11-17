"""
ETAPA 1 -- construcao da matriz de perdas K x n.

Avalia as MESMAS K configuracoes em TODOS os n datasets, sob o mesmo
protocolo de validacao. O resultado e o meta-conhecimento desta
atividade: a versao completa da tabela d1-d4 vista em aula.

Saida: results/hpo/loss_matrix.csv -- K linhas (configuracoes) por n
colunas (datasets), como o enunciado especifica.

POR QUE UM CONJUNTO FIXO DE CONFIGURACOES
-----------------------------------------
Com a matriz completa, a busca no dataset alvo pode ser SIMULADA:
avaliar uma configuracao passa a ser consultar uma celula, nao treinar
um modelo. Isso permite repetir cada busca com varias sementes a custo
praticamente nulo -- essencial, porque a diferenca entre os metodos e
da ordem da variancia entre sementes, como ja se viu na Atividade 2 --,
e da a perda otima L* de cada dataset de forma exata.

O preco e que a busca fica restrita as K configuracoes. O Capitulo 6
do livro aponta exatamente essa limitacao para o warm-start, ao final da
Secao 6.4.1: "these initialization procedures are limited to
configurations that have already been examined in the past and the
existing metadata captures this". A discussao desse preco esta no
relatorio.

PROTOCOLO
---------
Identico ao da Atividade 1 em tudo o que nao seja o numero de folds:
    - validacao cruzada com embaralhamento, semente 42
    - StandardScaler em X, ajustado apenas no treino de cada fold
    - StandardScaler em y via TransformedTargetRegressor
    - SERA como funcao de perda, com relevancia phi calculada sobre o
      alvo do dataset INTEIRO (phi e propriedade do problema, nao da
      particao)

A reducao de 10 para 5 folds e consequencia do custo: K x n x folds
treinos, contra os n x 5 algoritmos x 10 folds da Atividade 1. Com
K = 60 e 5 folds sao 9.000 treinos, seis vezes o que a Atividade 1
exigiu. Como a comparacao entre metodos de busca se faz sobre a MESMA
matriz, a reducao afeta a precisao de cada celula, nao a validade da
comparacao.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from hpo_space import gerar_configuracoes, como_kwargs, K
from metrics import relevance, sera

warnings.filterwarnings("ignore")

_SRC = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_SRC)
DATA_FOLDER = os.path.join(_RAIZ, "data")
RESULTS_FOLDER = os.path.join(_RAIZ, "results", "hpo")

N_FOLDS = 5
RANDOM_STATE = 42


def carregar_datasets(pasta=None):
    """Alvo na PRIMEIRA coluna, convencao de todo o projeto."""
    pasta = pasta or DATA_FOLDER
    for arquivo in sorted(os.listdir(pasta)):
        if not arquivo.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(pasta, arquivo))
        y = df.iloc[:, 0].to_numpy(dtype=float)
        X = df.iloc[:, 1:].to_numpy(dtype=float)
        yield arquivo.replace(".csv", ""), X, y


def avaliar_configuracao(X, y, kwargs, n_folds=N_FOLDS,
                         random_state=RANDOM_STATE):
    """
    SERA media de uma configuracao em um dataset, por validacao cruzada.

    A relevancia do fold de teste e calculada contra o alvo do dataset
    inteiro (y_reference=y), nao contra o proprio fold: a regiao rara e
    definida pelo problema, e deixa-la variar por particao tornaria as
    celulas da matriz incomparaveis entre si.

    Returns
    -------
    (perda, segundos_de_treino) -- SERA media e tempo medio de ajuste por
    fold. O tempo e medido porque o orcamento T do enunciado conta
    AVALIACOES, tratando todas as configuracoes como igualmente caras --
    o que e falso aqui: n_estimators varia de 20 a 500 e max_depth de 2 a
    12, de modo que as configuracoes diferem em mais de uma ordem de
    grandeza em custo. Sem medir, nao ha como saber se a vantagem de um
    metodo de warm-start vem de escolher configuracoes melhores ou apenas
    mais caras.

    As duas referencias do contexto que fundamentam a medicao:
    PIPES (Maia et al., 2025, Secao III, Eq. 2) define o meta-dataset
    como S* = {(fi, Sj, H, tau_train, tau_test, epsilon)} -- performance
    E tempos de treino e teste em cada celula, "allowing researchers to
    perform [...] cost-benefit analysis"; e a Aula 7 apresenta a medida
    A3R' = P / T^Q, que pondera performance por tempo de execucao.
    """
    modelo = TransformedTargetRegressor(
        regressor=XGBRegressor(**kwargs), transformer=StandardScaler()
    )
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    perdas, tempos = [], []
    for treino, teste in kf.split(X):
        escala = StandardScaler()
        X_treino = escala.fit_transform(X[treino])
        X_teste = escala.transform(X[teste])

        estimador = clone(modelo)
        t0 = time.perf_counter()
        estimador.fit(X_treino, y[treino])
        tempos.append(time.perf_counter() - t0)

        y_predito = estimador.predict(X_teste)

        phi_teste = relevance(y[teste], y_reference=y)
        perdas.append(sera(y[teste], y_predito, phi_teste))

    return float(np.mean(perdas)), float(np.mean(tempos))


def construir(pasta_dados=None, configuracoes=None, datasets=None,
              n_folds=N_FOLDS, random_state=RANDOM_STATE,
              caminho_checkpoint=None, verbose=True):
    """
    Percorre a grade configuracao x dataset.

    O checkpoint e salvo a cada dataset concluido. Com uma execucao de
    uma a duas horas, perder tudo por uma interrupcao seria caro; ao
    reiniciar, os datasets ja medidos sao pulados.

    Returns
    -------
    (matriz_perdas, matriz_tempos, custos) -- duas matrizes K x n, de
    perda e de tempo medio de treino por celula, e o custo por dataset.

    A matriz de tempos permite a analise de custo-beneficio da Aula 7
    (A3R'); um dataset retomado de um checkpoint anterior a essa medicao
    aparece na matriz de perdas e nao na de tempos, e as analises que
    usam tempo trabalham apenas com as colunas disponiveis.
    """
    configuracoes = (gerar_configuracoes() if configuracoes is None
                     else configuracoes)
    kwargs_por_config = {
        nome: como_kwargs(linha, random_state=random_state)
        for nome, linha in configuracoes.iterrows()
    }

    colunas, tempos_col, custos = {}, {}, {}

    if caminho_checkpoint and os.path.exists(caminho_checkpoint):
        parcial = pd.read_csv(caminho_checkpoint, index_col=0)
        colunas = {c: parcial[c] for c in parcial.columns}
        if verbose:
            print(f"Checkpoint: {len(colunas)} datasets ja medidos "
                  f"({', '.join(list(colunas)[:4])}...)")

    caminho_tempos = (caminho_checkpoint.replace("loss_matrix", "tempo_matrix")
                      if caminho_checkpoint else None)
    if caminho_tempos and os.path.exists(caminho_tempos):
        parcial_t = pd.read_csv(caminho_tempos, index_col=0)
        tempos_col = {c: parcial_t[c] for c in parcial_t.columns}

    for nome, X, y in carregar_datasets(pasta_dados):
        if datasets and nome not in datasets:
            continue
        if nome in colunas:
            continue

        t0 = time.time()
        perdas, tempos = {}, {}
        for config, kwargs in kwargs_por_config.items():
            perdas[config], tempos[config] = avaliar_configuracao(
                X, y, kwargs, n_folds=n_folds, random_state=random_state)

        colunas[nome] = pd.Series(perdas)
        tempos_col[nome] = pd.Series(tempos)
        decorrido = time.time() - t0
        custos[nome] = {"n": len(y), "d": X.shape[1],
                        "segundos": round(decorrido, 1)}

        if verbose:
            serie, serie_t = colunas[nome], tempos_col[nome]
            print(f"{nome:26} n={len(y):<6} {decorrido / 60:>5.1f} min  "
                  f"melhor={serie.min():.4g} ({serie.idxmin()})  "
                  f"pior/melhor={serie.max() / max(serie.min(), 1e-12):.1f}x  "
                  f"treino {serie_t.min():.3f}-{serie_t.max():.3f}s "
                  f"({serie_t.max() / max(serie_t.min(), 1e-12):.0f}x)",
                  flush=True)

        if caminho_checkpoint:
            pd.DataFrame(colunas).to_csv(caminho_checkpoint)
            pd.DataFrame(tempos_col).to_csv(caminho_tempos)

    matriz = pd.DataFrame(colunas)
    matriz.index.name = "configuracao"
    matriz_tempos = pd.DataFrame(tempos_col)
    matriz_tempos.index.name = "configuracao"
    return matriz, matriz_tempos, pd.DataFrame(custos).T


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="subconjunto, para medir custo antes da rodada completa")
    parser.add_argument("--sufixo", default="")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    configuracoes = gerar_configuracoes()
    configuracoes.to_csv(os.path.join(RESULTS_FOLDER, "configuracoes.csv"))

    print(f"Matriz de perdas: K={len(configuracoes)} configuracoes "
          f"x datasets, {args.folds} folds\n")

    inicio = time.time()
    matriz, matriz_tempos, custos = construir(
        configuracoes=configuracoes, datasets=args.datasets,
        n_folds=args.folds,
        caminho_checkpoint=os.path.join(
            RESULTS_FOLDER, f"loss_matrix{args.sufixo}.parcial.csv"),
    )

    matriz.to_csv(os.path.join(RESULTS_FOLDER, f"loss_matrix{args.sufixo}.csv"))
    matriz_tempos.to_csv(
        os.path.join(RESULTS_FOLDER, f"tempo_matrix{args.sufixo}.csv"))
    custos.to_csv(os.path.join(RESULTS_FOLDER, f"custo_matriz{args.sufixo}.csv"))

    print(f"\nMatriz: {matriz.shape[0]} configuracoes x {matriz.shape[1]} datasets")
    print(f"Tempo total: {(time.time() - inicio) / 60:.1f} min")

    # Diagnostico que decide se a comparacao da Etapa 2 tem o que medir:
    # com dispersao baixa, qualquer metodo acha logo uma boa configuracao
    # e os cinco empatam.
    razao = (matriz.max() / matriz.min().clip(lower=1e-12))
    print(f"\nDispersao (pior/melhor configuracao), por dataset:")
    print(f"  mediana {razao.median():.2f}x | "
          f"minimo {razao.min():.2f}x | maximo {razao.max():.2f}x")

    padrao_rank = matriz.rank(axis=0).loc["cfg_00"]
    print(f"\nRank da configuracao PADRAO entre as {len(matriz)}:")
    print(f"  mediana {padrao_rank.median():.0f} | "
          f"melhor {padrao_rank.min():.0f} | pior {padrao_rank.max():.0f}")
    print(f"  foi a melhor em {(padrao_rank == 1).sum()} de "
          f"{matriz.shape[1]} datasets")
