"""
PRATICA 4 -- os metodos 1, 2, 3 e 5: a solucao por meta-aprendizagem.

Os quatro vem das atividades anteriores e sao reexecutados na particao
comum, como o enunciado exige no quadro "Mesmo protocolo de fato".

    1. Melhor algoritmo medio   -> o de melhor rank medio nos DEMAIS
                                   datasets, configuracao padrao.
                                   A referencia SEM meta-caracteristicas.
    2. Recomendacao por MtL     -> o algoritmo indicado pelo meta-modelo
                                   em LODO, configuracao padrao.
    3. MtL com warm-start       -> a busca da Atividade 3, agora REAL.
    5. Oraculo                  -> o melhor dos m = 5 na particao.
                                   Nao e metodo; e o teto da selecao.

(O metodo 4, o AutoML, esta em p4_automl.py.)

O QUE MUDA EM RELACAO A ATIVIDADE 3: A BUSCA DEIXA DE SER SIMULADA
------------------------------------------------------------------
Na Atividade 3, avaliar uma configuracao era consultar uma celula da
matriz de perdas -- o que permitia repetir cada busca com dez sementes a
custo nulo. Aqui isso nao vale mais, por duas razoes.

A primeira e de protocolo: aquelas celulas vieram de validacao cruzada
sobre o dataset inteiro, e o enunciado exige medicao na particao de
teste que o AutoML tambem usa.

A segunda e o proprio objeto da Analise 2. O enunciado pede "o tempo de
cada metodo" e, para a meta-aprendizagem, que se separe o custo ONLINE
do OFFLINE. Uma busca simulada nao tem custo online -- consultar uma
tabela e gratuito. Medir a meta-aprendizagem contra o AutoML exigiria
inventar um custo, e inventado ele nao se compara com o relogio do
FLAML. Entao a busca passa a treinar de verdade, com relogio correndo.

A matriz de perdas continua indispensavel, mas no outro papel: e dela,
das colunas dos OUTROS 29 datasets, que saem os pontos iniciais.

SEPARACAO ONLINE / OFFLINE
--------------------------
custo OFFLINE (pago uma vez, amortizado sobre datasets futuros):
    construir a matriz P        -- Atividade 1
    extrair as meta-features    -- Atividade 2, nos 29 de treino
    construir a matriz de perdas-- Atividade 3
custo ONLINE (pago a cada dataset novo):
    extrair as meta-features DO ALVO
    consultar o meta-modelo
    treinar o recomendado, ou executar a busca
Esta separacao e o que a Analise 2 pede e o que permite responder
"quantos datasets novos seriam necessarios para que o custo offline se
pagasse".
"""

import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from harris import HarrisForest, normalize_performance_matrix
from hpo_space import N_INIT, T, como_kwargs, gerar_configuracoes
from p4_protocolo import (RANDOM_STATE, RESULTS_FOLDER, algoritmos_padrao,
                          avaliar_predicao, carregar, listar_datasets,
                          particionar, sera_de_busca, treinar_avaliar)
from warmstart import (configuracoes_para_matriz, iniciais_por_similaridade,
                       normalizar_meta_features)

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_RAIZ, "results")


# ======================================================================
# METODO 1 -- melhor algoritmo medio (sem meta-caracteristicas)
# ======================================================================

def melhor_algoritmo_medio(P, alvo):
    """
    O algoritmo de melhor rank medio sobre os DEMAIS datasets.

    O LODO vale aqui tambem: o alvo sai antes de calcular o rank medio.
    Sem isso, o metodo 1 veria a performance que se quer prever, e o
    "baseline sem meta-caracteristicas" deixaria de ser honesto.

    E a Eq. 2.2 do Capitulo 2 (Average Ranking) aplicada a algoritmos.
    """
    treino = P.drop(index=alvo)
    return treino.rank(axis=1).mean().idxmin()


# ======================================================================
# METODO 2 -- recomendacao por meta-aprendizagem
# ======================================================================

def recomendar_por_metamodelo(X_meta, P, alvo, lambda_param=0.5,
                              random_state=RANDOM_STATE):
    """
    O algoritmo top-1 indicado pelo meta-modelo, em LODO.

    O meta-modelo e a floresta hibrida de Harris usada na Atividade 1:
    treina sobre (meta-caracteristicas -> performances normalizadas) dos
    29 datasets de treino e prediz o vetor de performances do alvo; o
    recomendado e o de menor perda predita.

    A normalizacao de P por dataset (normalize_performance_matrix) e o
    que torna linhas de escalas muito diferentes comparaveis no nivel
    meta -- o mesmo problema que a Aula 8 descreve no SCoT: sem
    normalizar, o modelo aprenderia "qual dataset e facil" em vez de
    "qual algoritmo e bom".
    """
    treino = [d for d in X_meta.index if d != alvo]

    P_norm = normalize_performance_matrix(P.loc[treino])
    modelo = HarrisForest(lambda_param=lambda_param, random_state=random_state)
    modelo.fit(X_meta.loc[treino].to_numpy(float), np.asarray(P_norm))

    predito = modelo.predict(X_meta.loc[[alvo]].to_numpy(float))[0]
    return P.columns[int(np.argmin(predito))]


# ======================================================================
# METODO 3 -- meta-aprendizagem com warm-start, busca REAL
# ======================================================================

def _melhoria_esperada(mu, sigma, melhor):
    """EI na forma fechada da Eq. 6.7 do livro (identica a Atividade 3)."""
    sigma = np.maximum(sigma, 1e-12)
    u = (melhor - mu) / sigma
    return sigma * (u * norm.cdf(u) + norm.pdf(u))


def busca_com_warm_start(X_treino, y_treino, configuracoes, indices_iniciais,
                         orcamento=T, random_state=RANDOM_STATE,
                         n_arvores=50, frac_validacao=0.25):
    """
    SMBO real sobre as K configuracoes, partindo dos pontos quentes.

    A diferenca para executar_smbo da Atividade 3 e que aqui cada
    avaliacao TREINA um XGBoost: o historico vem de uma particao interna
    do treino (treino-interno / validacao), que o alvo nunca ve. O teste
    permanece intocado, como o enunciado exige do AutoML e, por simetria,
    de todos os metodos.

    Returns
    -------
    (indice_vencedor, segundos, historico) -- a configuracao incumbente
    ao fim do orcamento, o custo real da busca e os indices avaliados.
    """
    rng = np.random.default_rng(random_state)
    K = len(configuracoes)
    matriz_cfg = configuracoes_para_matriz(configuracoes)

    X_int, X_val, y_int, y_val = train_test_split(
        X_treino, y_treino, test_size=frac_validacao,
        random_state=random_state, shuffle=True)

    t0 = time.perf_counter()
    historico, perdas = [], []

    def avaliar(indice):
        kwargs = como_kwargs(configuracoes.iloc[indice],
                             random_state=random_state, n_jobs=-1)
        _, _ = None, None
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor
        escala = StandardScaler()
        Xi = escala.fit_transform(X_int)
        Xv = escala.transform(X_val)
        modelo = TransformedTargetRegressor(
            regressor=XGBRegressor(**kwargs), transformer=StandardScaler())
        modelo.fit(Xi, y_int)
        return sera_de_busca(y_val, modelo.predict(Xv), y_int)

    for indice in list(dict.fromkeys(indices_iniciais))[:orcamento]:
        historico.append(int(indice))
        perdas.append(avaliar(int(indice)))

    while len(historico) < orcamento:
        restantes = np.setdiff1d(np.arange(K), historico)
        if restantes.size == 0:
            break
        if len(historico) < 2:
            escolhido = int(rng.choice(restantes))
        else:
            substituto = RandomForestRegressor(
                n_estimators=n_arvores, random_state=random_state, n_jobs=-1)
            substituto.fit(matriz_cfg[historico], np.asarray(perdas))
            por_arvore = np.stack([
                arvore.predict(matriz_cfg[restantes])
                for arvore in substituto.estimators_])
            mu, sigma = por_arvore.mean(axis=0), por_arvore.std(axis=0)
            ei = _melhoria_esperada(mu, sigma, min(perdas))
            escolhido = int(restantes[int(np.argmax(ei))])
        historico.append(escolhido)
        perdas.append(avaliar(escolhido))

    segundos = time.perf_counter() - t0
    return int(historico[int(np.argmin(perdas))]), segundos, historico


# ======================================================================
# Execucao
# ======================================================================

def rodar(datasets=None, verbose=True):
    """Executa os metodos 1, 2, 3 e 5 em todos os datasets."""
    P = pd.read_csv(os.path.join(RESULTS, "P_matrix.csv"), index_col=0)
    X_base = pd.read_csv(os.path.join(RESULTS, "X_matrix.csv"), index_col=0)
    X_ext = pd.read_csv(os.path.join(RESULTS, "X_estendida.csv"), index_col=0)
    L = pd.read_csv(os.path.join(RESULTS, "hpo", "loss_matrix.csv"), index_col=0)
    configuracoes = pd.read_csv(
        os.path.join(RESULTS, "hpo", "configuracoes.csv"), index_col=0)

    datasets = datasets or listar_datasets()
    algos = algoritmos_padrao()
    linhas = []

    for nome in datasets:
        X, y = carregar(nome)
        X_tr, X_te, y_tr, y_te = particionar(X, y)

        # --- os 5 algoritmos padrao: servem aos metodos 1, 2 e 5 de uma vez
        sera_algo, tempo_algo = {}, {}
        for sigla, estimador in algos.items():
            s, dt = treinar_avaliar(estimador, X_tr, y_tr, X_te, y_te, y)
            sera_algo[sigla], tempo_algo[sigla] = s, dt

        # METODO 1
        alg1 = melhor_algoritmo_medio(P, nome)
        # METODO 2, nas duas matrizes de meta-caracteristicas
        alg2_base = recomendar_por_metamodelo(X_base, P, nome)
        alg2_ext = recomendar_por_metamodelo(X_ext, P, nome)

        # METODO 5 -- oraculo
        alg5 = min(sera_algo, key=sera_algo.get)

        # METODO 3 -- warm-start real
        indice_alvo = list(L.columns).index(nome)
        perdas_treino = np.delete(L.to_numpy(float), indice_alvo, axis=1)
        X_meta_np = X_base.loc[L.columns].to_numpy(float)
        X_tr_meta, x_alvo = normalizar_meta_features(X_meta_np, indice_alvo)
        iniciais = iniciais_por_similaridade(
            perdas_treino, X_tr_meta, x_alvo, N_INIT, 3)
        vencedor, t_busca, _ = busca_com_warm_start(
            X_tr, y_tr, configuracoes, iniciais)

        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor
        kw = como_kwargs(configuracoes.iloc[vencedor],
                         random_state=RANDOM_STATE, n_jobs=-1)
        s3, t3_final = treinar_avaliar(
            XGBRegressor(**kw), X_tr, y_tr, X_te, y_te, y)

        linhas.append({
            "dataset": nome,
            "sera_melhor_medio": sera_algo[alg1], "alg_melhor_medio": alg1,
            "tempo_melhor_medio": tempo_algo[alg1],
            "sera_mtl_base": sera_algo[alg2_base], "alg_mtl_base": alg2_base,
            "tempo_mtl_base": tempo_algo[alg2_base],
            "sera_mtl_ext": sera_algo[alg2_ext], "alg_mtl_ext": alg2_ext,
            "tempo_mtl_ext": tempo_algo[alg2_ext],
            "sera_warmstart": s3, "cfg_warmstart": configuracoes.index[vencedor],
            "tempo_warmstart": t_busca + t3_final,
            "sera_oraculo": sera_algo[alg5], "alg_oraculo": alg5,
            **{f"sera_{k}": v for k, v in sera_algo.items()},
            **{f"tempo_{k}": v for k, v in tempo_algo.items()},
        })

        if verbose:
            r = linhas[-1]
            print(f"{nome:26} medio={alg1:<4}{r['sera_melhor_medio']:>10.1f}  "
                  f"mtl={alg2_base:<4}{r['sera_mtl_base']:>10.1f}  "
                  f"ws={r['sera_warmstart']:>10.1f} ({t_busca:>5.1f}s)  "
                  f"oraculo={alg5:<4}{r['sera_oraculo']:>10.1f}", flush=True)

    return pd.DataFrame(linhas).set_index("dataset")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--sufixo", default="")
    args = parser.parse_args()

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    inicio = time.time()
    tabela = rodar(datasets=args.datasets)
    tabela.to_csv(os.path.join(RESULTS_FOLDER, f"baselines{args.sufixo}.csv"))

    print(f"\n{len(tabela)} datasets em {(time.time()-inicio)/60:.1f} min")

    # Qual matriz de meta-caracteristicas recomenda melhor: o enunciado
    # pede o metodo 2 "com X_base ou X_estendido, a que tiver se saido
    # melhor", entao a escolha e medida, nao arbitrada.
    print("\nMetodo 2 -- qual matriz de meta-caracteristicas recomenda melhor:")
    for col, rot in (("sera_mtl_base", "X_base"), ("sera_mtl_ext", "X_estendida")):
        rel = (tabela[col] - tabela["sera_oraculo"]) / tabela["sera_oraculo"]
        print(f"  {rot:12} perda relativa ao oraculo: media {rel.mean():.4f} | "
              f"mediana {rel.median():.4f} | acertou o oraculo em "
              f"{(tabela[col] == tabela['sera_oraculo']).sum()} de {len(tabela)}")

    print("\nCusto ONLINE por metodo (segundos, mediana):")
    for col, rot in (("tempo_melhor_medio", "1. melhor medio"),
                     ("tempo_mtl_base", "2. recomendacao MtL"),
                     ("tempo_warmstart", "3. MtL + warm-start")):
        print(f"  {rot:22} {tabela[col].median():8.2f}s  "
              f"(total {tabela[col].sum():.0f}s)")
    print("\n  -> o orcamento CURTO do AutoML sai da mediana do metodo 3.")
