"""
Secao 2.1 -- as tres analises da Atividade Pratica 3.

1. Curvas de perda, area sob a curva, e comparacao em tres orcamentos
   com teste de Friedman e diagrama de diferenca critica.
2. O papel das meta-caracteristicas: metodo 4 contra metodo 5, X base
   contra X estendida, e variacao do numero de vizinhos k.
3. Quando o warm-start falha: ganho do metodo 4 sobre o 3 em funcao da
   distancia ao vizinho mais proximo.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, spearmanr

from evaluation import diagrama_diferenca_critica
from hpo_space import N_INIT, T
from warmstart import (executar_todos, normalizar_meta_features,
                       diagnosticar_iniciais, METODOS)

warnings.filterwarnings("ignore")

_SRC = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_SRC)
RESULTS = os.path.join(_RAIZ, "results")
HPO = os.path.join(RESULTS, "hpo")

CORES = {
    "padrao": "#5c6673", "aleatoria": "#8a939e",
    "smbo_frio": "#2f5d6b", "smbo_similar": "#b8562f",
    "smbo_independente": "#8a6a2f",
}


# ======================================================================
# ANALISE 1 -- curvas de perda
# ======================================================================

def curva_media(resultados, coluna="perda_norm"):
    """
    Perda media do incumbente por metodo e avaliacao, sobre datasets e
    sementes.

    coluna="perda_norm" da a curva do enunciado (linear entre L* e a pior
    da matriz); coluna="perda_rank" da a versao por ranking, imune a
    dispersao extrema de um dataset. Ver a nota em warmstart.executar_todos.
    """
    return (resultados.groupby(["metodo", "avaliacao"])[coluna]
            .mean().unstack(level=0))


def area_sob_curva(curva):
    """
    Media da curva no intervalo do orcamento.

    O Capitulo 3 (Secao 3.6.2) trata essa area como a caracterizacao
    natural de uma curva de perda: um unico numero representando a perda
    media ao longo do intervalo, de modo que um metodo que desce rapido
    e um que desce devagar ate o mesmo ponto final deixam de parecer
    equivalentes.
    """
    return curva.mean()


def comparar_em_orcamentos(resultados, orcamentos, pasta_saida=None,
                           sufixo="", coluna="perda_norm", datasets=None):
    """
    Friedman e diagrama de diferenca critica em cada orcamento.

    A pergunta do enunciado: o warm-start ajuda so no comeco ou a
    vantagem persiste ate o fim? Tres orcamentos respondem isso -- logo
    apos os pontos iniciais, no meio, e em T.

    O diagrama espera que MAIOR seja melhor, e perda e o contrario;
    passa-se a perda negada.

    OS BLOCOS DO TESTE. Os blocos sao os DATASETS, com a perda media
    sobre as sementes em cada celula -- nao os pares (dataset, semente).
    Sementes sao replicas da mesma busca no mesmo problema, nao problemas
    independentes; tratar as 10 sementes de 30 datasets como 300 blocos
    multiplicaria o poder do teste por um fator que nao corresponde a
    informacao disponivel, e quase tudo sairia significativo. E tambem o
    que fizemos na Atividade 2, onde os blocos foram os 30 datasets.

    datasets : lista, opcional
        Restringe os blocos a esses datasets alvo. Serve ao recorte sem
        quase-duplicatas (ver pares_quase_duplicados).
    """
    if datasets is not None:
        resultados = resultados[resultados["dataset"].isin(datasets)]

    tabelas = {}
    for orcamento in orcamentos:
        fatia = resultados[resultados["avaliacao"] == orcamento]
        por_dataset = (fatia.groupby(["dataset", "metodo"])[coluna]
                       .mean().unstack(level="metodo"))

        estatistica, p = friedmanchisquare(
            *[por_dataset[c] for c in por_dataset.columns])

        tabelas[orcamento] = {
            "tabela": por_dataset,
            "friedman_p": float(p),
            "ranks_medios": por_dataset.rank(axis=1).mean().sort_values(),
        }

        if pasta_saida:
            diagrama_diferenca_critica(
                -por_dataset,
                os.path.join(pasta_saida, f"cd_orcamento{orcamento}{sufixo}.png"),
            )

    return tabelas


def plotar_curvas(curva, caminho, titulo="Curvas de perda", n_init=N_INIT):
    """Curva de perda normalizada por numero de avaliacoes."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for metodo in curva.columns:
        ax.plot(curva.index, curva[metodo], marker="o", markersize=3.5,
                label=metodo, color=CORES.get(metodo), linewidth=1.8)

    ax.axvline(n_init, color="#b4b2a9", linestyle="--", linewidth=1)
    ax.annotate("fim dos pontos iniciais", xy=(n_init, ax.get_ylim()[1] * 0.92),
                xytext=(n_init + 0.4, ax.get_ylim()[1] * 0.92),
                fontsize=8, color="#5f5e5a")

    ax.set_xlabel("Numero de avaliacoes")
    ax.set_ylabel("Perda normalizada do incumbente\n(0 = otimo, 1 = pior da matriz)")
    ax.set_title(titulo)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================================
# ANALISE 2 -- o papel das meta-caracteristicas
# ======================================================================

def variar_vizinhos(matriz_perdas, configuracoes, X_meta,
                    valores_k=(1, 3, 5), orcamento=T, n_init=N_INIT,
                    sementes=range(10)):
    """
    O metodo 4 com diferentes numeros de vizinhos, nos DOIS arranjos.

    arranjo="ninit_fixo"  -- ninit constante, rodizio entre os k vizinhos.
                             E o que o enunciado pede ("os mesmos T e
                             ninit para todos"), mas com k < ninit o
                             metodo desce no ranking de cada vizinho, de
                             modo que variar k confunde numero de
                             vizinhos com profundidade no ranking.
    arranjo="acoplado"    -- ninit = k, uma configuracao por vizinho. E o
                             procedimento canonico: o exemplo da Aula 8
                             usa dois vizinhos e dois pontos iniciais, e
                             o Auto-sklearn 1.0 toma k configuracoes dos
                             k datasets mais proximos. Aqui variar k
                             tambem muda a divisao do orcamento entre
                             pontos iniciais e passos guiados pelo modelo.

    Rodar os dois custa quase nada -- a busca e simulada sobre a matriz --
    e permite separar o efeito do numero de vizinhos dos dois
    confundidores. Se as duas series concordarem na direcao, a conclusao
    sobre a similaridade nao depende do arranjo.
    """
    linhas = []
    for k in valores_k:
        for arranjo, n in (("ninit_fixo", n_init), ("acoplado", k)):
            resultado = executar_todos(
                matriz_perdas, configuracoes, X_meta,
                orcamento=orcamento, n_init=n, k_vizinhos=k,
                sementes=sementes, metodos=["smbo_similar"],
            )
            resultado["k"] = k
            resultado["arranjo"] = arranjo
            resultado["n_init"] = n
            linhas.append(resultado)
    return pd.concat(linhas, ignore_index=True)


def resumir_variacao_k(variacao, orcamento=T, coluna="perda_norm"):
    """Area sob a curva por (arranjo, k), e a perda final em T."""
    area = (variacao.groupby(["arranjo", "k", "avaliacao"])[coluna]
            .mean().groupby(level=["arranjo", "k"]).mean()
            .rename("area_sob_curva"))
    final = (variacao[variacao["avaliacao"] == orcamento]
             .groupby(["arranjo", "k"])[coluna].mean()
             .rename(f"perda_em_T{orcamento}"))
    return pd.concat([area, final], axis=1)


def comparar_meta_features(matriz_perdas, configuracoes, conjuntos_X,
                           orcamento=T, n_init=N_INIT, k_vizinhos=3,
                           sementes=range(10)):
    """
    O metodo 4 com diferentes conjuntos de meta-caracteristicas.

    Se o problema da Atividade 2 foi mantido, compara X base e X
    estendida. A X estendida acrescenta as 12 medidas de complexidade de
    Lorena et al. e a decomposicao por regiao de relevancia; se a
    similaridade entre datasets melhorar com ela, as meta-caracteristicas
    da Atividade 2 encontram aqui um uso que nao tiveram la.
    """
    linhas = []
    for nome, X_meta in conjuntos_X.items():
        resultado = executar_todos(
            matriz_perdas, configuracoes, X_meta,
            orcamento=orcamento, n_init=n_init, k_vizinhos=k_vizinhos,
            sementes=sementes, metodos=["smbo_similar"],
        )
        resultado["conjunto_X"] = nome
        linhas.append(resultado)
    return pd.concat(linhas, ignore_index=True)


# ======================================================================
# ANALISE 3 -- quando o warm-start falha
# ======================================================================

def ganho_por_dataset(resultados, orcamento=T,
                      metodo="smbo_similar", referencia="smbo_frio"):
    """
    Ganho do warm-start sobre a partida fria, dataset a dataset.

    Positivo significa que o warm-start terminou com perda menor.
    """
    fatia = resultados[resultados["avaliacao"] == orcamento]
    tabela = (fatia.groupby(["dataset", "metodo"])["perda_norm"]
              .mean().unstack(level="metodo"))
    return (tabela[referencia] - tabela[metodo]).rename("ganho")


def distancia_ao_vizinho(X_meta, datasets, k_vizinhos=1):
    """
    Distancia L1 de cada dataset ao seu vizinho mais proximo, sob o
    mesmo LODO usado na busca -- normalizacao excluindo o alvo.

    E a variavel que o enunciado pede para relacionar com o ganho: se o
    warm-start depende da similaridade, deve render menos quando o
    vizinho mais proximo esta longe.
    """
    X = X_meta.loc[datasets].to_numpy(dtype=float)
    distancias = {}
    for i, nome in enumerate(datasets):
        treino, alvo = normalizar_meta_features(X, i)
        d = np.abs(treino - alvo).sum(axis=1)
        distancias[nome] = float(np.sort(d)[:k_vizinhos].mean())
    return pd.Series(distancias, name="dist_vizinho")


def analisar_falhas(resultados, X_meta, orcamento=T):
    """
    Junta ganho e distancia, e correlaciona.

    Returns
    -------
    (tabela, rho, p) -- tabela por dataset e a correlacao de Spearman
    entre distancia ao vizinho e ganho do warm-start.
    """
    datasets = sorted(resultados["dataset"].unique())
    ganho = ganho_por_dataset(resultados, orcamento=orcamento)
    distancia = distancia_ao_vizinho(X_meta, datasets)

    tabela = pd.concat([ganho, distancia], axis=1).dropna()
    rho, p = spearmanr(tabela["dist_vizinho"], tabela["ganho"])
    return tabela.sort_values("ganho"), float(rho), float(p)


def pares_quase_duplicados(X_meta, datasets, limiar=3.0):
    """
    Identifica os pares de datasets quase indistinguiveis no espaco de
    meta-caracteristicas, e o recorte de alvos sem quase-duplicata.

    POR QUE ISSO IMPORTA. O LODO e respeitado -- o alvo sai da matriz e
    da normalizacao --, mas se o vizinho mais proximo for praticamente o
    mesmo problema sob outro nome, o metodo 4 recebe uma vantagem que
    nao se reproduz num dataset genuinamente novo. O resultado nao e
    invalido; e otimista, e precisa ser declarado.

    O PRECEDENTE E DA PROPRIA BIBLIOGRAFIA. PIPES (Maia et al., 2025,
    Secao IV-A) parte de 1.248 datasets e exclui 867: "to reduce the
    number of datasets belonging to the same family and to remove
    repeated datasets that had been logged with different names". MetaML
    (Maia et al., 2025, Secao 5.5) repete o corte na comparacao com o
    estado da arte: "Other 47 datasets were excluded to reduce the number
    of datasets belonging to the same family, hence increasing the
    diversity within the test bed".

    O desenho do MetaML e mais fino que um corte simples, e e o que se
    adota aqui: as familias ficam nos 290 datasets do meta-dataset de
    treino e sao aparadas apenas no test bed de 152 usado para avaliar.
    Sob LODO cada dataset e metadado n-1 vezes e alvo uma vez, de modo
    que o analogo e manter os 30 como metadados SEMPRE e reportar a
    comparacao tambem sobre o subconjunto de alvos sem quase-duplicata.

    limiar : float
        Distancia L1 abaixo da qual dois datasets sao tratados como
        quase-duplicatas. O padrao 3,0 separa, no nosso conjunto, o
        bloco a1/a2/a3/a7 e os pares compactiv/cpu_small e
        treasury/mortgage do resto -- todos abaixo de 2,7, contra um
        proximo valor bem acima.

    Returns
    -------
    (pares, alvos_isolados) -- DataFrame dos pares abaixo do limiar, com
    a distancia, e a lista de datasets cujo vizinho mais proximo esta
    acima dele.
    """
    X = X_meta.loc[datasets].to_numpy(dtype=float)
    linhas, isolados = [], []

    for i, nome in enumerate(datasets):
        treino, alvo = normalizar_meta_features(X, i)
        d = np.abs(treino - alvo).sum(axis=1)
        outros = [n for j, n in enumerate(datasets) if j != i]
        ordem = np.argsort(d)

        if d[ordem[0]] >= limiar:
            isolados.append(nome)
        else:
            for pos in ordem:
                if d[pos] < limiar:
                    linhas.append({"dataset": nome,
                                   "vizinho": outros[pos],
                                   "distancia": float(d[pos])})

    pares = pd.DataFrame(linhas).sort_values("distancia") if linhas \
        else pd.DataFrame(columns=["dataset", "vizinho", "distancia"])
    return pares, isolados


# ======================================================================
# ANALISE COMPLEMENTAR -- o orcamento em tempo, nao em avaliacoes
# ======================================================================

def curva_por_tempo(resultados, pontos=40, coluna="perda_norm"):
    """
    A mesma curva de perda, mas com o orcamento medido em SEGUNDOS de
    treino acumulado em vez de numero de avaliacoes.

    POR QUE. O enunciado conta avaliacoes, o que trata todas as
    configuracoes como igualmente caras. Nao sao: n_estimators vai de 20
    a 500 e max_depth de 2 a 12, de modo que uma configuracao pode custar
    mais de uma ordem de grandeza a mais que outra. Um metodo que
    privilegia configuracoes grandes aparece melhor por avaliacao sem ser
    melhor por segundo gasto.

    A Aula 7 da a resposta da disciplina a esse problema -- a medida
    A3R' = P / T^Q, que pondera performance por tempo de execucao, com Q
    controlando o peso do tempo -- e PIPES guarda tau_train em cada celula
    do meta-dataset justamente para permitir "cost-benefit analysis"
    (Secao VII). Aqui a ponderacao e feita do modo mais direto: reler as
    mesmas buscas numa grade de orcamento em segundos.

    Requer que executar_todos tenha recebido matriz_tempos.

    Returns
    -------
    DataFrame indexado pelo orcamento em segundos, uma coluna por metodo,
    ou None se nao houver tempos registrados.
    """
    if ("segundos_acumulados" not in resultados.columns
            or resultados["segundos_acumulados"].isna().all()):
        return None

    validos = resultados.dropna(subset=["segundos_acumulados"])
    teto = validos.groupby(["dataset", "metodo", "semente"])[
        "segundos_acumulados"].max().groupby("dataset").min()

    grade_global = np.geomspace(
        max(validos["segundos_acumulados"].min(), 1e-3),
        float(teto.max()), pontos)

    registros = []
    for (dataset, metodo, semente), grupo in validos.groupby(
            ["dataset", "metodo", "semente"]):
        grupo = grupo.sort_values("segundos_acumulados")
        tempos = grupo["segundos_acumulados"].to_numpy()
        perdas = grupo[coluna].to_numpy()

        # degrau: em cada orcamento, o incumbente ja alcancado
        posicoes = np.searchsorted(tempos, grade_global, side="right") - 1
        for orcamento, pos in zip(grade_global, posicoes):
            if pos < 0 or orcamento > teto[dataset]:
                continue
            registros.append({"segundos": float(orcamento), "metodo": metodo,
                              "dataset": dataset, "semente": semente,
                              coluna: float(perdas[pos])})

    if not registros:
        return None

    longo = pd.DataFrame(registros)
    return (longo.groupby(["metodo", "segundos"])[coluna]
            .mean().unstack(level=0))


def custo_das_escolhas(resultados, matriz_tempos, orcamento=T):
    """
    Quanto custou, em segundos de treino, o que cada metodo escolheu.

    Separa as duas explicacoes possiveis para uma vantagem observada:
    escolher configuracoes melhores, ou escolher configuracoes mais
    caras. Se o metodo vencedor gastou sistematicamente mais segundos no
    mesmo numero de avaliacoes, parte da vantagem e orcamento disfarcado.
    """
    if "segundos_acumulados" not in resultados.columns:
        return None
    fatia = resultados[resultados["avaliacao"] == orcamento].dropna(
        subset=["segundos_acumulados"])
    if fatia.empty:
        return None

    return (fatia.groupby("metodo")
            .agg(segundos_medios=("segundos_acumulados", "mean"),
                 segundos_mediana=("segundos_acumulados", "median"),
                 perda_final=("perda_norm", "mean"))
            .sort_values("perda_final"))


def plotar_ganho_vs_distancia(tabela, caminho):
    """Dispersao do ganho contra a distancia ao vizinho mais proximo."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cores = ["#b8562f" if g < 0 else "#2f5d6b" for g in tabela["ganho"]]
    ax.scatter(tabela["dist_vizinho"], tabela["ganho"], c=cores, s=45)
    ax.axhline(0, color="#888780", linewidth=1)

    # rotula os casos em que o warm-start foi pior que partir do zero
    for nome, linha in tabela[tabela["ganho"] < 0].iterrows():
        ax.annotate(nome, (linha["dist_vizinho"], linha["ganho"]),
                    fontsize=7, color="#993c1d",
                    xytext=(4, -2), textcoords="offset points")

    ax.set_xlabel("Distancia L1 ao vizinho mais proximo")
    ax.set_ylabel("Ganho do warm-start sobre a partida fria")
    ax.set_title("Quando o warm-start falha")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sufixo", default="")
    parser.add_argument("--sementes", type=int, default=10)
    parser.add_argument("--orcamento", type=int, default=T)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--aquisicao", choices=["ei", "lcb"], default="ei",
                        help="ei = Eq. 6.7 do livro; lcb = mu-2sigma da Aula 8")
    args = parser.parse_args()

    os.makedirs(HPO, exist_ok=True)
    sementes = range(args.sementes)

    matriz = pd.read_csv(
        os.path.join(HPO, f"loss_matrix{args.sufixo}.csv"), index_col=0)
    configuracoes = pd.read_csv(
        os.path.join(HPO, "configuracoes.csv"), index_col=0)
    X_base = pd.read_csv(os.path.join(RESULTS, "X_matrix.csv"), index_col=0)

    caminho_tempos = os.path.join(HPO, f"tempo_matrix{args.sufixo}.csv")
    matriz_tempos = (pd.read_csv(caminho_tempos, index_col=0)
                     if os.path.exists(caminho_tempos) else None)

    caminho_estendida = os.path.join(RESULTS, "X_estendida.csv")
    X_estendida = (pd.read_csv(caminho_estendida, index_col=0)
                   if os.path.exists(caminho_estendida) else None)

    print(f"Matriz: {matriz.shape[0]} configuracoes x {matriz.shape[1]} datasets")
    print(f"T={args.orcamento}  ninit={N_INIT}  k={args.k}  "
          f"sementes={args.sementes}\n")

    # ---------------- ANALISE 1 ----------------
    print("=" * 62)
    print("ANALISE 1 -- curvas de perda")
    print("=" * 62)

    resultados = executar_todos(
        matriz, configuracoes, X_base,
        orcamento=args.orcamento, n_init=N_INIT, k_vizinhos=args.k,
        sementes=sementes, verbose=False,
        aquisicao=args.aquisicao, matriz_tempos=matriz_tempos,
    )
    resultados.to_csv(
        os.path.join(HPO, f"resultados_busca{args.sufixo}.csv"), index=False)

    curva = curva_media(resultados)
    curva.to_csv(os.path.join(HPO, f"curva_perda{args.sufixo}.csv"))
    plotar_curvas(curva, os.path.join(HPO, f"curva_perda{args.sufixo}.png"))

    areas = area_sob_curva(curva).sort_values()
    print("\nArea sob a curva (menor = melhor):")
    print(areas.round(4).to_string())

    # A mesma area pela curva de RANKING, imune a dispersao extrema de um
    # dataset (ver a nota em warmstart.executar_todos).
    curva_rank = curva_media(resultados, coluna="perda_rank")
    curva_rank.to_csv(os.path.join(HPO, f"curva_rank{args.sufixo}.csv"))
    print("\nArea sob a curva, normalizacao por RANKING:")
    print(area_sob_curva(curva_rank).sort_values().round(4).to_string())

    print("\nPerda normalizada em cada orcamento:")
    orcamentos = [N_INIT, args.orcamento // 2, args.orcamento]
    print(curva.loc[orcamentos].round(4).to_string())

    tabelas = comparar_em_orcamentos(
        resultados, orcamentos, pasta_saida=HPO, sufixo=args.sufixo)
    for orcamento, dados in tabelas.items():
        print(f"\nOrcamento {orcamento}: Friedman p={dados['friedman_p']:.4g}")
        print("  ranks medios: " + ", ".join(
            f"{m}={r:.2f}" for m, r in dados["ranks_medios"].items()))
    print(f"\n  (blocos = {matriz.shape[1]} datasets, media sobre as "
          f"{args.sementes} sementes)")

    # ---------------- ANALISE 2 ----------------
    print("\n" + "=" * 62)
    print("ANALISE 2 -- o papel das meta-caracteristicas")
    print("=" * 62)

    em_T = resultados[resultados["avaliacao"] == args.orcamento]
    comparacao = em_T.groupby("metodo")["perda_norm"].agg(["mean", "std"])
    print("\nMetodo 4 (similaridade) contra metodo 5 (independente):")
    print(comparacao.loc[["smbo_similar", "smbo_independente", "smbo_frio"]]
          .round(4).to_string())

    print("\nDiagnostico dos pontos iniciais de cada metodo:")
    diag = diagnosticar_iniciais(matriz, configuracoes, X_base.loc[
        matriz.columns].to_numpy(dtype=float),
        n_init=N_INIT, k_vizinhos=args.k)
    diag.to_csv(os.path.join(HPO, f"diagnostico_iniciais{args.sufixo}.csv"))
    print(diag.round(3).to_string())
    print("  dispersao_media: distancia media entre os pontos iniciais")
    print("  configuracoes_distintas: quantas o metodo usa em todos os alvos")

    print("\nVariando o numero de vizinhos k, nos dois arranjos:")
    por_k = variar_vizinhos(matriz, configuracoes, X_base,
                            orcamento=args.orcamento, sementes=sementes)
    por_k.to_csv(os.path.join(HPO, f"variacao_k{args.sufixo}.csv"), index=False)
    print(resumir_variacao_k(por_k, orcamento=args.orcamento)
          .round(4).to_string())

    if X_estendida is not None:
        print("\nX base contra X estendida:")
        por_X = comparar_meta_features(
            matriz, configuracoes,
            {"X_base": X_base, "X_estendida": X_estendida},
            orcamento=args.orcamento, k_vizinhos=args.k, sementes=sementes)
        por_X.to_csv(
            os.path.join(HPO, f"comparacao_X{args.sufixo}.csv"), index=False)
        resumo_X = (por_X[por_X["avaliacao"] == args.orcamento]
                    .groupby("conjunto_X")["perda_norm"].agg(["mean", "std"]))
        print(resumo_X.round(4).to_string())

    # ---------------- ANALISE 3 ----------------
    print("\n" + "=" * 62)
    print("ANALISE 3 -- quando o warm-start falha")
    print("=" * 62)

    tabela, rho, p = analisar_falhas(resultados, X_base,
                                     orcamento=args.orcamento)
    tabela.to_csv(os.path.join(HPO, f"ganho_vs_distancia{args.sufixo}.csv"))
    plotar_ganho_vs_distancia(
        tabela, os.path.join(HPO, f"ganho_vs_distancia{args.sufixo}.png"))

    piores = tabela[tabela["ganho"] < 0]
    print(f"\nWarm-start foi PIOR que partir do zero em "
          f"{len(piores)} de {len(tabela)} datasets:")
    if len(piores):
        print(piores.round(4).to_string())

    print(f"\nCorrelacao entre distancia ao vizinho e ganho: "
          f"rho={rho:.3f} (p={p:.4g})")
    print("  (rho negativo sustenta a hipotese: quanto mais longe o "
          "vizinho,\n   menos o warm-start rende)")

    # Recorte sem quase-duplicatas, seguindo PIPES e MetaML
    pares, isolados = pares_quase_duplicados(X_base, list(matriz.columns))
    pares.to_csv(os.path.join(HPO, f"pares_duplicados{args.sufixo}.csv"),
                 index=False)
    print(f"\nPares quase-duplicados (L1 < 3,0): {len(pares)}")
    if len(pares):
        print(pares.head(10).round(2).to_string(index=False))

    if isolados and len(isolados) >= 5:
        print(f"\nMesma comparacao, so nos {len(isolados)} alvos SEM "
              f"quase-duplicata:")
        tabelas_rec = comparar_em_orcamentos(
            resultados, [args.orcamento], datasets=isolados)
        dados = tabelas_rec[args.orcamento]
        print(f"  Friedman p={dados['friedman_p']:.4g}")
        print("  ranks medios: " + ", ".join(
            f"{m}={r:.2f}" for m, r in dados["ranks_medios"].items()))
        print("  (os 30 datasets seguem como metadados de treino em todos "
              "os casos;\n   o recorte vale so para os alvos avaliados)")

    # ---------------- COMPLEMENTAR: orcamento em tempo ----------------
    por_tempo = curva_por_tempo(resultados)
    if por_tempo is not None:
        print("\n" + "=" * 62)
        print("COMPLEMENTAR -- orcamento em segundos, nao em avaliacoes")
        print("=" * 62)
        por_tempo.to_csv(os.path.join(HPO, f"curva_tempo{args.sufixo}.csv"))
        print("\nArea sob a curva por tempo (menor = melhor):")
        print(por_tempo.mean().sort_values().round(4).to_string())

        custo = custo_das_escolhas(resultados, matriz_tempos,
                                   orcamento=args.orcamento)
        if custo is not None:
            print(f"\nCusto do que cada metodo escolheu, em {args.orcamento} "
                  f"avaliacoes:")
            print(custo.round(3).to_string())
            print("  (se o metodo de menor perda tambem gastou mais segundos,"
                  "\n   parte da vantagem e orcamento disfarcado -- Aula 7, A3R')")
    elif matriz_tempos is None:
        print("\n(sem tempo_matrix.csv: a analise de custo-beneficio exige "
              "rodar\n build_loss_matrix.py com a versao que mede tempo "
              "por celula)")
