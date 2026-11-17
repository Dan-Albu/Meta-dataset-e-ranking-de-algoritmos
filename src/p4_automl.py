"""
PRATICA 4 -- metodo 4: o sistema de AutoML.

SISTEMA ESCOLHIDO: FLAML
------------------------
A escolha foi por viabilidade medida, nao por reputacao. O Auto-sklearn
seria o mais interessante do ponto de vista pedagogico -- e o sistema
que o proprio enunciado cita por carregar meta-conhecimento do OpenML --
mas ele exige Python antigo e scikit-learn fixado em versao antiga, e
nao instala no ambiente deste projeto (Python 3.13).

O FLAML instala limpo e, mais importante, oferece os quatro controles
que esta atividade exige:

  time_budget     orcamento de tempo exato, para os tres orcamentos
  metric          metrica customizada -- precisamos de SERA, nao de RMSE
  estimator_list  restringir a busca a um conjunto de algoritmos
  starting_points o liga-desliga da meta-aprendizagem interna

Os dois ultimos sao o que torna a atividade respondivel com experimento
em vez de especulacao:

ESTIMATOR_LIST responde a pergunta central da Analise 3 -- "o ganho vem
de escolher melhor entre os mesmos algoritmos, ou de buscar em um espaco
que a sua solucao nao alcancava?". Rodando o FLAML uma vez restrito aos
algoritmos equivalentes aos m = 5 da matriz P e uma vez irrestrito, a
diferenca entre as duas isola o efeito do espaco de busca.

STARTING_POINTS e o liga-desliga que o enunciado pede no quadro sobre
sobreposicao com o meta-conhecimento do sistema. E ele reproduz, dentro
do FLAML, a dicotomia da Atividade 3:
    "data"   -> defaults DEPENDENTES do dataset: o FLAML extrai
                meta-caracteristicas do alvo e consulta um portfolio
                aprendido offline. E o nosso metodo 4 (warm-start por
                similaridade).
    "static" -> defaults INDEPENDENTES dos dados. E o nosso metodo 5
                (portfolio estatico).

O META-CONHECIMENTO DO FLAML, E A SOBREPOSICAO
----------------------------------------------
O portfolio de regressao embarcado (flaml/default/all/regression.json)
usa QUATRO meta-caracteristicas: NumberOfInstances, NumberOfFeatures,
NumberOfClasses (zero em regressao) e PercentageOfNumericFeatures --
contra as 13 da nossa X_base. Essa pobreza e criticada por MetaML (Maia
et al., 2025, Secao 3.4), que avalia o mesmo sistema: "its meta-model
relies on only four simple meta-features [...] These simple meta-features
are unlikely to provide enough predictive power if a wider search space
focusing on the full pipeline was to be considered."

O campo configsource do portfolio nomeia os datasets de origem: houses,
house_8L, house_16H, poker, Albert, mv, bng_echomonths, higgs,
bng_pharynx, connect-4, bank-marketing, dilbert, particulate-matter --
todos do OpenML. Nenhum coincide em nome com os nossos 30, e nenhum
coincide exatamente em (n, d). O unico caso suspeito e california
(20.640 x 8), que encontra um vizinho do portfolio em (~20.505, 8): a
reconstrucao dos valores brutos a partir das features padronizadas tem
precisao de cerca de +-140 linhas, e houses, no OpenML, E o California
housing. Ver verificar_sobreposicao().

A METRICA
---------
O FLAML otimiza o que lhe dermos. Entregar RMSE e avaliar em SERA
tornaria a comparacao injusta com o AutoML -- ele estaria perseguindo
outro objetivo. A metrica customizada abaixo e SERA com phi do TREINO,
pela mesma razao que o warm-start usa: a regiao rara do teste nao pode
entrar na definicao do que a busca persegue.
"""

import json
import os
import time

import numpy as np
import pandas as pd

from metrics import relevance, sera
from p4_protocolo import (RANDOM_STATE, RESULTS_FOLDER, avaliar_predicao,
                          carregar, listar_datasets, particionar)

# Os estimadores do FLAML mais proximos dos m = 5 da nossa matriz P.
# Nao ha correspondencia perfeita: o FLAML nao expoe SVR nem arvore
# unica para regressao, e traz lgbm e extra_tree, que nao temos. A
# intersecao honesta e {rf, xgboost, kneighbor}, e a declaracao desse
# descompasso e parte da resposta da Analise 3.
ESTIMADORES_RESTRITOS = ["rf", "xgboost", "kneighbor"]

ORCAMENTOS_PADRAO = (10, 60, 300)


def _corrigir_portfolio_do_flaml():
    """
    Contorna um defeito do proprio FLAML, nao do nosso codigo.

    O portfolio de regressao embarcado traz DUAS configuracoes (indices
    16, de rf, e 20, de extra_tree) com o hiperparametro `criterion`,
    que e de CLASSIFICACAO -- o arquivo all/regression.json carrega
    configuracoes que pertenciam ao portfolio de classificacao. Ao
    validar um ponto inicial contra o espaco de busca de regressao, o
    FLAML faz `search_space['criterion']` sem protecao e levanta
    KeyError, abortando a execucao inteira.

    O erro so aparece quando as meta-caracteristicas do dataset alvo
    levam a consulta a escolher uma dessas duas configuracoes, de modo
    que atinge alguns datasets e nao outros: na validacao, derrubou
    cpu_small em todas as execucoes com starting_points="data" e
    passou nos demais. Como atinge apenas o modo "data", ele
    silenciaria justamente a variante COM meta-aprendizagem -- a que o
    enunciado manda comparar contra a variante sem.

    A correcao trata configuracao com chave desconhecida como ponto
    inicial invalido, que e a semantica correta: aquela configuracao
    nao pertence a este espaco de busca. O FLAML passa a proxima do
    portfolio, de modo que a meta-aprendizagem segue ativa -- perde-se
    apenas as duas configuracoes contaminadas.
    """
    from flaml.automl.state import SearchState

    if getattr(SearchState, "_p4_corrigido", False):
        return
    original = SearchState.valid_starting_point

    def valido_com_guarda(self, starting_point, search_space):
        try:
            return original(self, starting_point, search_space)
        except KeyError:
            return False

    SearchState.valid_starting_point = valido_com_guarda
    SearchState._p4_corrigido = True


def metrica_sera(X_val, y_val, estimator, labels, X_train, y_train,
                 weight_val=None, weight_train=None, *args, **kwargs):
    """
    SERA como metrica interna do FLAML.

    A assinatura e imposta pelo FLAML. phi vem de y_train: a validacao
    interna e do sistema, e o teste nao entra aqui.
    """
    predito = estimator.predict(X_val)
    valor = float(sera(y_val, predito, relevance(y_val, y_reference=y_train)))
    return valor, {"sera": valor}


def verificar_sobreposicao(datasets=None, pasta=None):
    """
    A verificacao que a Etapa 1 exige: os nossos datasets estao no
    meta-conhecimento do FLAML?

    O portfolio guarda os vizinhos por meta-caracteristicas PADRONIZADAS,
    sem nome. Desfazendo a padronizacao com o bloco preprocessing
    (center/scale) recuperam-se os valores brutos, que se comparam aos
    nossos. A precisao da reconstrucao e limitada pelos decimais
    guardados: um erro de 1e-4 na feature padronizada desloca
    NumberOfInstances em cerca de 46 linhas, o que a coluna
    precisao_n quantifica.
    """
    import flaml.default as fd

    caminho = os.path.join(os.path.dirname(fd.__file__), "all", "regression.json")
    with open(caminho) as fp:
        portfolio = json.load(fp)

    centro = np.array(portfolio["preprocessing"]["center"], dtype=float)
    escala = np.array(portfolio["preprocessing"]["scale"], dtype=float)
    nomes = portfolio["meta_feature_names"]

    vizinhos = pd.DataFrame(
        [np.array(v["features"], dtype=float) * escala + centro
         for v in portfolio["neighbors"]], columns=nomes).drop_duplicates()
    vizinhos["NumberOfInstances"] = vizinhos["NumberOfInstances"].round().astype(int)
    vizinhos["NumberOfFeatures"] = vizinhos["NumberOfFeatures"].round().astype(int)

    datasets = datasets or listar_datasets(pasta)
    linhas = []
    for nome in datasets:
        X, y = carregar(nome, pasta)
        n, d = len(y), X.shape[1]
        mesma_d = vizinhos[vizinhos["NumberOfFeatures"] == d]
        if len(mesma_d):
            k = (mesma_d["NumberOfInstances"] - n).abs().idxmin()
            dif = int(abs(vizinhos.loc[k, "NumberOfInstances"] - n))
            viz_n = int(vizinhos.loc[k, "NumberOfInstances"])
        else:
            dif, viz_n = None, None
        linhas.append({"dataset": nome, "n": n, "d": d,
                       "vizinho_mesma_d_n": viz_n, "dif_n": dif})

    tabela = pd.DataFrame(linhas).set_index("dataset")
    tabela["suspeito"] = tabela["dif_n"].notna() & (tabela["dif_n"] <= 200)
    return tabela, portfolio.get("configsource", []), vizinhos


def rodar_flaml(X_treino, y_treino, X_teste, y_teste, y_completo,
                orcamento, starting_points="data", estimator_list="auto",
                ensemble=False, random_state=RANDOM_STATE, n_jobs=1):
    """
    Uma execucao do FLAML, do ajuste a medicao no teste.

    O escalonamento NAO e aplicado aqui, ao contrario de treinar_avaliar:
    o FLAML trata o pre-processamento internamente e normaliza o que
    cada estimador precisa. Padronizar antes seria interferir no
    pipeline do sistema, que e justamente o objeto da comparacao.

    Returns
    -------
    dict com a SERA de teste, o tempo real, o estimador escolhido, a
    configuracao vencedora e quantos modelos couberam no orcamento.
    """
    from flaml import AutoML

    _corrigir_portfolio_do_flaml()
    automl = AutoML()
    t0 = time.perf_counter()
    automl.fit(
        X_train=np.asarray(X_treino, dtype=float),
        y_train=np.asarray(y_treino, dtype=float),
        task="regression", metric=metrica_sera,
        time_budget=orcamento, starting_points=starting_points,
        estimator_list=estimator_list, ensemble=ensemble,
        seed=random_state, n_jobs=n_jobs, verbose=0,
    )
    segundos = time.perf_counter() - t0

    predito = automl.predict(np.asarray(X_teste, dtype=float))
    return {
        "sera": avaliar_predicao(y_teste, predito, y_completo),
        "segundos": segundos,
        "estimador": automl.best_estimator,
        "config": json.dumps(automl.best_config, default=str),
        "modelos_testados": len(automl.config_history),
        "ensemble": bool(ensemble),
    }


def rodar(datasets=None, orcamentos=ORCAMENTOS_PADRAO, verbose=True,
          variantes=None):
    """
    O metodo 4 em todas as combinacoes.

    variantes : lista de (rotulo, starting_points, estimator_list)
        O padrao cobre as tres perguntas da atividade:
          auto_data    -> o sistema como ele vem (meta-aprendizagem ON)
          auto_static  -> meta-aprendizagem OFF, o controle do enunciado
          restrito     -> busca limitada aos algoritmos da matriz P,
                          que isola o efeito do espaco de busca
    """
    variantes = variantes or [
        ("auto_data", "data", "auto"),
        ("auto_static", "static", "auto"),
        ("restrito_data", "data", ESTIMADORES_RESTRITOS),
    ]
    datasets = datasets or listar_datasets()
    linhas = []

    for nome in datasets:
        X, y = carregar(nome)
        X_tr, X_te, y_tr, y_te = particionar(X, y)
        for rotulo, inicio, estimadores in variantes:
            for orcamento in orcamentos:
                try:
                    r = rodar_flaml(X_tr, y_tr, X_te, y_te, y, orcamento,
                                    starting_points=inicio,
                                    estimator_list=estimadores)
                except Exception as erro:          # noqa: BLE001
                    r = {"sera": np.nan, "segundos": np.nan,
                         "estimador": f"ERRO: {type(erro).__name__}: {erro}",
                         "config": "", "modelos_testados": 0,
                         "ensemble": False}
                linhas.append({"dataset": nome, "variante": rotulo,
                               "orcamento": orcamento, **r})
                if verbose:
                    print(f"{nome:26} {rotulo:14} {orcamento:>4}s -> "
                          f"SERA={r['sera']:>11.1f}  {r['estimador']:<16} "
                          f"({r['modelos_testados']:>3} modelos)", flush=True)

    return pd.DataFrame(linhas)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--orcamentos", nargs="*", type=int,
                        default=list(ORCAMENTOS_PADRAO))
    parser.add_argument("--sufixo", default="")
    parser.add_argument("--so-sobreposicao", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    tabela, fontes, vizinhos = verificar_sobreposicao(args.datasets)
    tabela.to_csv(os.path.join(RESULTS_FOLDER, f"sobreposicao{args.sufixo}.csv"))
    print("ETAPA 1 -- sobreposicao com o meta-conhecimento do FLAML")
    print(f"  o portfolio veio de {len(set(f.split('/')[-1] for f in fontes))} "
          f"datasets do OpenML:")
    print("   ", ", ".join(sorted({f.split("/")[-1] for f in fontes})))
    suspeitos = tabela[tabela["suspeito"]]
    print(f"  nossos datasets com vizinho de mesmo d e n a menos de 200: "
          f"{len(suspeitos)}")
    if len(suspeitos):
        print(suspeitos[["n", "d", "vizinho_mesma_d_n", "dif_n"]].to_string())

    if args.so_sobreposicao:
        raise SystemExit(0)

    print(f"\nMETODO 4 -- FLAML, orcamentos {args.orcamentos}s\n")
    inicio = time.time()
    resultados = rodar(datasets=args.datasets, orcamentos=args.orcamentos)
    resultados.to_csv(
        os.path.join(RESULTS_FOLDER, f"automl{args.sufixo}.csv"), index=False)
    print(f"\n{len(resultados)} execucoes em {(time.time()-inicio)/60:.1f} min")
