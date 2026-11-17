"""
PRATICA 4 -- as tres analises da Secao 2.1.

    1. Desempenho          -- dataset a dataset e no agregado, com perda
                              relativa ao melhor observado; Friedman +
                              Nemenyi + diagrama de diferenca critica,
                              SEPARADAMENTE para cada orcamento.
    2. Desempenho e custo  -- tempo de cada metodo, desempenho contra
                              custo, custo online separado do offline, e
                              o ponto de equilibrio do investimento.
    3. O que o AutoML      -- o modelo final por dataset, se os
       escolheu              algoritmos estao entre os m da matriz P, e
                              de onde vem o ganho.

O ORACULO NAO ENTRA NO TESTE DE FRIEDMAN
----------------------------------------
O enunciado e explicito: "Nao e um metodo, mas e o teto da selecao de
algoritmos e ajuda a ler os demais." Um teste de significancia compara
METODOS -- procedimentos que alguem poderia executar num dataset novo.
O oraculo nao e executavel: ele olha o resultado de teste dos cinco
algoritmos e escolhe o menor depois do fato. Incluir no Friedman
inflaria o numero de tratamentos com algo que ganha por construcao e
deslocaria os ranks de todos os demais. Ele entra como linha de
referencia nas tabelas e nos graficos.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr

from evaluation import diagrama_diferenca_critica
from p4_protocolo import RESULTS_FOLDER, perda_relativa

try:  # usado so para converter a tempo_matrix, que e media por fold
    from build_loss_matrix import N_FOLDS
except Exception:  # pragma: no cover
    N_FOLDS = 5

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_RAIZ, "results")

ROTULOS = {
    "melhor_medio": "1. melhor algoritmo medio",
    "mtl": "2. recomendacao MtL",
    "warmstart": "3. MtL + warm-start",
    "oraculo": "5. oraculo (teto)",
}


def montar_tabela(baselines, automl, orcamento, variante="auto_data",
                  coluna_mtl="sera_mtl_base"):
    """
    A tabela de SERA no teste: linhas = datasets, colunas = metodos,
    com o AutoML num orcamento especifico.
    """
    fatia = automl[(automl["orcamento"] == orcamento)
                   & (automl["variante"] == variante)]
    fatia = fatia.set_index("dataset")["sera"]

    tabela = pd.DataFrame({
        "melhor_medio": baselines["sera_melhor_medio"],
        "mtl": baselines[coluna_mtl],
        "warmstart": baselines["sera_warmstart"],
        f"automl_{orcamento}s": fatia,
        "oraculo": baselines["sera_oraculo"],
    })
    return tabela.dropna()


# ======================================================================
# ANALISE 1 -- desempenho
# ======================================================================

def analise_desempenho(baselines, automl, orcamentos, variante="auto_data",
                       coluna_mtl="sera_mtl_base", pasta_saida=None,
                       sufixo=""):
    """
    Friedman + Nemenyi + diagrama de diferenca critica em cada orcamento.

    Os blocos sao os datasets, como na Pratica 3 e na Aula 5. Nao ha
    sementes a mediar aqui: cada metodo produz um resultado por dataset,
    porque a particao e fixa e os metodos sao deterministas dada a
    semente.
    """
    saida = {}
    for orcamento in orcamentos:
        tabela = montar_tabela(baselines, automl, orcamento, variante,
                               coluna_mtl)
        relativa = perda_relativa(tabela)
        metodos = [c for c in tabela.columns if c != "oraculo"]

        est, p = friedmanchisquare(*[relativa[c].to_numpy() for c in metodos])
        ranks = relativa[metodos].rank(axis=1).mean().sort_values()

        if pasta_saida:
            diagrama_diferenca_critica(
                -relativa[metodos],
                os.path.join(pasta_saida, f"cd_{orcamento}s{sufixo}.png"))

        saida[orcamento] = {
            "tabela_sera": tabela, "perda_relativa": relativa,
            "friedman_p": float(p), "friedman_stat": float(est),
            "ranks_medios": ranks,
            "perda_media": relativa.mean().sort_values(),
            "vitorias": tabela[metodos].idxmin(axis=1).value_counts(),
        }
    return saida


# ======================================================================
# ANALISE 2 -- desempenho e custo
# ======================================================================

def custos_online(baselines, automl, orcamentos, variante="auto_data"):
    """Custo online por metodo, em segundos por dataset."""
    linhas = {
        "melhor_medio": baselines["tempo_melhor_medio"],
        "mtl": baselines["tempo_mtl_base"],
        "warmstart": baselines["tempo_warmstart"],
    }
    for orcamento in orcamentos:
        fatia = automl[(automl["orcamento"] == orcamento)
                       & (automl["variante"] == variante)]
        linhas[f"automl_{orcamento}s"] = fatia.set_index("dataset")["segundos"]
    return pd.DataFrame(linhas)


def custo_offline(caminho_custo_matriz=None, n_datasets_alvo=30):
    """
    O custo pago uma vez, que a Analise 2 manda separar do online.

    A fonte preferida e a `custo_matriz.csv` da Atividade 3: relogio de
    parede por dataset, cobrindo as K configuracoes com todos os folds,
    o ajuste, a predicao e o calculo da SERA. E o custo que de fato foi
    pago para construir o meta-conhecimento.

    A `tempo_matrix.csv` NAO e equivalente e so entra como alternativa.
    Cada celula ali e o tempo MEDIO DE AJUSTE POR FOLD (docstring de
    `avaliar_configuracao` em `build_loss_matrix.py`), de modo que a
    soma crua subestima o custo em um fator N_FOLDS. Em `cpu_small`, por
    exemplo, a soma crua da 28,1 s contra 144,0 s de relogio de parede.
    Usada como alternativa, ela e multiplicada por N_FOLDS e ainda assim
    constitui piso, nao medicao: exclui predicao, escalonamento e metrica.

    Nenhuma das duas e obrigatoria. Se faltarem, a funcao devolve um
    dicionario vazio e o relatorio registra a lacuna -- um custo offline
    nao medido e uma limitacao a declarar, nunca uma excecao que derruba
    as Analises 2 e 3.

    Devolve (componentes, nota), onde `nota` descreve a procedencia: se o
    numero foi medido sobre os 30 datasets ou extrapolado de uma medicao
    parcial. Um valor extrapolado nao pode ser apresentado como medido.
    """
    pasta = os.path.join(RESULTS, "hpo")
    componentes, nota = {}, "custo offline nao medido"

    caminho = caminho_custo_matriz or os.path.join(pasta, "custo_matriz.csv")
    if os.path.exists(caminho):
        custo = pd.read_csv(caminho, index_col=0)
        coluna = _coluna_de_tempo(custo)
        if coluna is not None:
            medido = float(pd.to_numeric(custo[coluna],
                                         errors="coerce").sum())
            cobertos = int(len(custo))
            if cobertos >= n_datasets_alvo:
                componentes["matriz_de_perdas"] = medido
                nota = (f"relogio de parede, {cobertos} datasets "
                        f"(coluna '{coluna}')")
            elif cobertos > 0:
                componentes["matriz_de_perdas"] = (
                    medido * n_datasets_alvo / cobertos)
                nota = (f"EXTRAPOLADO: {medido/60:.1f} min de relogio de "
                        f"parede em {cobertos} de {n_datasets_alvo} "
                        f"datasets, escalados por {n_datasets_alvo}/{cobertos}")
            return componentes, nota
        nota = (f"{os.path.basename(caminho)} sem coluna de tempo "
                f"reconhecivel (colunas: {list(custo.columns)})")

    caminho_tempos = os.path.join(pasta, "tempo_matrix.csv")
    if os.path.exists(caminho_tempos):
        tempos = pd.read_csv(caminho_tempos, index_col=0)
        tempos = tempos.apply(pd.to_numeric, errors="coerce")
        # x N_FOLDS: as celulas sao a media por fold, nao o total.
        medido = float(np.nansum(tempos.to_numpy(dtype=float))) * N_FOLDS
        cobertos = int(tempos.shape[1])
        if cobertos > 0:
            fator = max(1.0, n_datasets_alvo / cobertos)
            componentes["matriz_de_perdas"] = medido * fator
            escala = ("" if cobertos >= n_datasets_alvo
                      else f", escalados por {n_datasets_alvo}/{cobertos}")
            nota = (f"PISO estimado da tempo_matrix: ajuste x {N_FOLDS} "
                    f"folds em {cobertos} datasets{escala}; exclui "
                    f"predicao, escalonamento e metrica")

    return componentes, nota


def _coluna_de_tempo(tabela):
    """
    Qual coluna guarda segundos, sem depender do nome exato.

    O `build_loss_matrix.py` ja gravou essa coluna com nomes diferentes
    ao longo das atividades. Procura por nome conhecido e, se nenhum
    casar, cai na ultima coluna numerica -- que e onde o tempo sempre
    ficou no layout `n, d, segundos`.
    """
    conhecidos = ("segundos", "seconds", "tempo", "tempo_s",
                  "segundos_treino", "custo")
    for nome in conhecidos:
        for col in tabela.columns:
            if str(col).strip().lower() == nome:
                return col
    numericas = [c for c in tabela.columns
                 if pd.api.types.is_numeric_dtype(tabela[c])]
    return numericas[-1] if numericas else None


def ponto_de_equilibrio(custo_offline_total, economia_por_dataset):
    """
    Quantos datasets novos para o investimento offline se pagar.

    A pergunta do enunciado: "Quantos datasets novos seriam necessarios
    para que o custo offline se pagasse?"

    A economia por dataset e a diferenca entre o que o AutoML gasta para
    igualar a meta-aprendizagem e o que a meta-aprendizagem gasta online.
    Se a economia for negativa -- o AutoML chega la mais barato --, o
    investimento nunca se paga, e a funcao devolve infinito.
    """
    if economia_por_dataset <= 0:
        return float("inf")
    return custo_offline_total / economia_por_dataset


def plotar_desempenho_custo(baselines, automl, orcamentos, caminho,
                            variante="auto_data", coluna_mtl="sera_mtl_base"):
    """Desempenho contra custo, o grafico que a Analise 2 pede."""
    pontos = []
    tabela_ref = montar_tabela(baselines, automl, orcamentos[-1], variante,
                               coluna_mtl)
    relativa = perda_relativa(tabela_ref)

    for chave, rotulo, tempo in (
            ("melhor_medio", "1. melhor medio", baselines["tempo_melhor_medio"]),
            ("mtl", "2. recomendacao MtL", baselines[coluna_mtl.replace("sera", "tempo")]),
            ("warmstart", "3. MtL + warm-start", baselines["tempo_warmstart"])):
        pontos.append((rotulo, float(tempo.median()),
                       float(relativa[chave].mean())))

    for orcamento in orcamentos:
        tab = montar_tabela(baselines, automl, orcamento, variante, coluna_mtl)
        rel = perda_relativa(tab)
        fatia = automl[(automl["orcamento"] == orcamento)
                       & (automl["variante"] == variante)]
        pontos.append((f"AutoML {orcamento}s",
                       float(fatia["segundos"].median()),
                       float(rel[f"automl_{orcamento}s"].mean())))

    fig, eixo = plt.subplots(figsize=(7.2, 4.6))
    for rotulo, custo, perda in pontos:
        marcador = "o" if rotulo.startswith("AutoML") else "s"
        eixo.scatter(custo, perda, s=70, marker=marcador, zorder=3)
        eixo.annotate(rotulo, (custo, perda), textcoords="offset points",
                      xytext=(7, 4), fontsize=8)

    automl_pts = sorted([(c, p) for r, c, p in pontos if r.startswith("AutoML")])
    if len(automl_pts) > 1:
        eixo.plot([c for c, _ in automl_pts], [p for _, p in automl_pts],
                  "--", lw=1, alpha=0.5, zorder=2)

    eixo.set_xscale("log")
    eixo.set_xlabel("custo online por dataset (s, mediana) -- escala log")
    eixo.set_ylabel("perda relativa media ao melhor observado")
    eixo.set_title("Desempenho contra custo")
    eixo.grid(alpha=0.3, zorder=1)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return pd.DataFrame(pontos, columns=["metodo", "segundos", "perda_media"])


# ======================================================================
# ANALISE 3 -- o que o AutoML escolheu
# ======================================================================

# Correspondencia entre os estimadores do FLAML e os m = 5 da matriz P.
# lgbm, extra_tree, catboost, histgb e os lineares NAO tem equivalente
# -- e e justamente isso que a analise quer evidenciar.
EQUIVALENTES = {"rf": "RF", "xgboost": "XGB", "xgb_limitdepth": "XGB",
                "kneighbor": "KNN", "extra_tree": None, "lgbm": None,
                "catboost": None, "histgb": None, "enet": None,
                "lassolars": None, "sgd": None, "lrl1": None, "lrl2": None}


def analise_escolhas(baselines, automl, orcamento, variante="auto_data",
                     coluna_alg_mtl="alg_mtl_base"):
    """
    O que o AutoML escolheu, e se isso esta no nosso espaco.

    Responde as tres perguntas do enunciado: quais algoritmos aparecem;
    se estao entre os m da matriz P; e se concordam com o recomendado
    pela meta-aprendizagem.
    """
    fatia = automl[(automl["orcamento"] == orcamento)
                   & (automl["variante"] == variante)].set_index("dataset")

    tabela = pd.DataFrame({
        "escolhido_automl": fatia["estimador"],
        "equivalente_em_P": fatia["estimador"].map(
            lambda e: EQUIVALENTES.get(e, None)),
        "recomendado_mtl": baselines[coluna_alg_mtl],
        "oraculo": baselines["alg_oraculo"],
        "modelos_testados": fatia["modelos_testados"],
    }).dropna(subset=["escolhido_automl"])

    tabela["dentro_do_espaco"] = tabela["equivalente_em_P"].notna()
    tabela["concorda_com_mtl"] = (
        tabela["equivalente_em_P"] == tabela["recomendado_mtl"])
    return tabela


def ganho_do_automl(baselines, automl, orcamento, variante="auto_data",
                    coluna_mtl="sera_mtl_base"):
    """
    Ganho do AutoML sobre a melhor solucao por meta-aprendizagem, por
    dataset, com o contexto que a Analise 3 manda relacionar.
    """
    fatia = automl[(automl["orcamento"] == orcamento)
                   & (automl["variante"] == variante)].set_index("dataset")

    # Alinhamento explicito. O AutoML pode cobrir menos datasets que os
    # baselines -- numa execucao parcial, ou quando uma execucao falha e
    # vira NaN -- e comparar Series de indices diferentes levanta
    # ValueError em vez de alinhar. O reindex pelos datasets COMUNS
    # torna a funcao utilizavel tambem em execucoes por blocos.
    comuns = baselines.index.intersection(fatia.index)
    base = baselines.loc[comuns]
    sera_automl = fatia.loc[comuns, "sera"]

    melhor_mtl = base[[coluna_mtl, "sera_warmstart"]].min(axis=1)
    ganho = (melhor_mtl - sera_automl) / melhor_mtl

    contexto = pd.DataFrame({"ganho_relativo": ganho})
    contexto["superou_oraculo"] = sera_automl < base["sera_oraculo"]

    caminho = os.path.join(RESULTS, "hpo", "ganho_vs_distancia_q.csv")
    if os.path.exists(caminho):
        dist = pd.read_csv(caminho, index_col=0)
        if "dist_vizinho" in dist.columns:
            contexto["dist_vizinho_A3"] = dist["dist_vizinho"]

    L = os.path.join(RESULTS, "hpo", "loss_matrix.csv")
    if os.path.exists(L):
        matriz = pd.read_csv(L, index_col=0)
        contexto["dispersao_matriz"] = (matriz.max() / matriz.min())

    return contexto.dropna(subset=["ganho_relativo"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sufixo", default="")
    parser.add_argument("--variante", default="auto_data")
    args = parser.parse_args()

    P4 = RESULTS_FOLDER
    baselines = pd.read_csv(
        os.path.join(P4, f"baselines{args.sufixo}.csv"), index_col=0)
    automl = pd.read_csv(os.path.join(P4, f"automl{args.sufixo}.csv"))
    orcamentos = sorted(automl["orcamento"].unique())

    # Qual matriz de meta-caracteristicas usar no metodo 2: medida.
    rel_base = ((baselines["sera_mtl_base"] - baselines["sera_oraculo"])
                / baselines["sera_oraculo"]).mean()
    rel_ext = ((baselines["sera_mtl_ext"] - baselines["sera_oraculo"])
               / baselines["sera_oraculo"]).mean()
    coluna_mtl = "sera_mtl_base" if rel_base <= rel_ext else "sera_mtl_ext"
    print(f"Metodo 2 usa {coluna_mtl} (perda ao oraculo: X_base "
          f"{rel_base:.4f} vs X_estendida {rel_ext:.4f})\n")

    print("=" * 66)
    print("ANALISE 1 -- DESEMPENHO")
    print("=" * 66)
    res = analise_desempenho(baselines, automl, orcamentos, args.variante,
                             coluna_mtl, pasta_saida=P4, sufixo=args.sufixo)
    for orcamento, d in res.items():
        print(f"\nAutoML com {orcamento}s  (Friedman p={d['friedman_p']:.3g})")
        print("  rank medio:  " + " | ".join(
            f"{m}={r:.2f}" for m, r in d["ranks_medios"].items()))
        print("  perda media: " + " | ".join(
            f"{m}={v:.4f}" for m, v in d["perda_media"].items()))
        print("  vitorias:    " + d["vitorias"].to_dict().__str__())

    print("\n" + "=" * 66)
    print("ANALISE 2 -- DESEMPENHO E CUSTO")
    print("=" * 66)
    custos = custos_online(baselines, automl, orcamentos, args.variante)
    custos.to_csv(os.path.join(P4, f"custos_online{args.sufixo}.csv"))
    print("\ncusto online por dataset (s):")
    print(custos.median().round(2).to_string())

    offline, nota_offline = custo_offline()
    total_offline = sum(offline.values())
    if offline:
        print("\ncusto offline: " + ", ".join(
            f"{k}={v/60:.1f} min" for k, v in offline.items()))
        print(f"  procedencia: {nota_offline}")
    else:
        print(f"\ncusto offline indisponivel -- {nota_offline}")
        print("  a Analise 2 segue sem o ponto de equilibrio; a lacuna")
        print("  deve ser declarada no relatorio, nao preenchida.")

    pontos = plotar_desempenho_custo(
        baselines, automl, orcamentos,
        os.path.join(P4, f"desempenho_custo{args.sufixo}.png"),
        args.variante, coluna_mtl)
    pontos.to_csv(os.path.join(P4, f"desempenho_custo{args.sufixo}.csv"),
                  index=False)
    print("\n" + pontos.round(4).to_string(index=False))

    melhor_mtl_perda = min(res[orcamentos[0]]["perda_media"][m]
                           for m in ("mtl", "warmstart"))
    supera = [o for o in orcamentos
              if res[o]["perda_media"][f"automl_{o}s"] < melhor_mtl_perda]
    print(f"\norcamento em que o AutoML passa a superar a MtL: "
          f"{supera[0] if supera else 'nenhum dos testados'}s")
    if supera and total_offline > 0:
        custo_mtl = custos["warmstart"].median()
        economia = supera[0] - custo_mtl
        n = ponto_de_equilibrio(total_offline, economia)
        print(f"  economia por dataset novo: {economia:.1f}s")
        print(f"  datasets novos para o offline se pagar: "
              f"{n:.0f}" if np.isfinite(n) else "  nunca se paga")
    elif supera:
        print("  ponto de equilibrio nao calculado: custo offline ausente")

    print("\n" + "=" * 66)
    print("ANALISE 3 -- O QUE O AUTOML ESCOLHEU")
    print("=" * 66)
    for orcamento in orcamentos:
        esc = analise_escolhas(baselines, automl, orcamento, args.variante)
        esc.to_csv(os.path.join(P4, f"escolhas_{orcamento}s{args.sufixo}.csv"))
        print(f"\n{orcamento}s -- algoritmos escolhidos:")
        print("  " + esc["escolhido_automl"].value_counts().to_dict().__str__())
        print(f"  dentro dos m=5 da matriz P: {esc['dentro_do_espaco'].sum()}"
              f"/{len(esc)}")
        print(f"  concordam com a recomendacao MtL: "
              f"{esc['concorda_com_mtl'].sum()}/{len(esc)}")

    ganhos = ganho_do_automl(baselines, automl, orcamentos[-1], args.variante,
                             coluna_mtl)
    ganhos.to_csv(os.path.join(P4, f"ganhos_automl{args.sufixo}.csv"))
    print(f"\nGanho do AutoML ({orcamentos[-1]}s) sobre a melhor MtL:")
    print(f"  superou o ORACULO em {int(ganhos['superou_oraculo'].sum())} "
          f"de {len(ganhos)} datasets")
    print("     (o oraculo e o teto da selecao entre os m=5; supera-lo so e")
    print("      possivel buscando FORA desse espaco)")
    for col in ("dist_vizinho_A3", "dispersao_matriz"):
        if col in ganhos.columns:
            sub = ganhos.dropna(subset=[col])
            if len(sub) > 3:
                rho, p = spearmanr(sub[col], sub["ganho_relativo"])
                print(f"  Spearman(ganho, {col}) = {rho:+.3f} (p={p:.3g})")
