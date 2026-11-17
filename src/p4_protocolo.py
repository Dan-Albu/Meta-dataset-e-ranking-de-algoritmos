"""
PRATICA 4 -- o protocolo comum a todos os metodos.

POR QUE ESTE MODULO EXISTE
--------------------------
O enunciado e explicito: "Para cada dataset, avalie todos os metodos na
mesma particao de teste e com a mesma medida de performance. O sistema
de AutoML so pode usar os dados de treino -- a validacao interna que ele
faz para escolher modelos e dele, e nao deve tocar no teste."

Isso MUDA o protocolo das atividades anteriores, e a mudanca e a
armadilha desta pratica. As Atividades 1 e 3 mediam tudo por validacao
cruzada sobre o dataset INTEIRO: a matriz P veio de 10 folds, a matriz
de perdas de 5. Um sistema de AutoML nao cabe nesse molde -- ele precisa
de um conjunto de treino que ele particiona internamente como quiser, e
de um teste que ninguem toca.

A consequencia e que as matrizes P e de perdas deixam de ser a fonte dos
NUMEROS comparados e passam a ser apenas a fonte do META-CONHECIMENTO:

    as outras n-1 colunas de P       -> decidem QUAL algoritmo recomendar
    as outras n-1 colunas de L       -> decidem POR ONDE a busca comeca
    a particula de treino do alvo    -> onde tudo e de fato treinado
    a particao de teste do alvo      -> onde tudo e de fato medido

O enunciado preve exatamente isso no quadro "Mesmo protocolo de fato":
"reexecute os algoritmos dos itens 1, 2 e 5 na mesma particao usada pelo
AutoML -- caso contrario, a comparacao mistura protocolos diferentes".

DUAS DECISOES SOBRE A RELEVANCIA phi
------------------------------------
A SERA precisa de phi, e phi depende de uma distribuicao de referencia.
Ha duas situacoes, e elas pedem respostas diferentes:

1. AVALIACAO FINAL (a comparacao entre metodos). phi vem do alvo do
   dataset INTEIRO, como nas Atividades 1 e 3. A regiao rara e
   propriedade do problema, nao da particao, e usar a mesma referencia
   para todos os metodos e o que torna os numeros comparaveis entre si
   e com as atividades anteriores.

2. BUSCA INTERNA (o que o AutoML e o warm-start otimizam). phi vem
   apenas do y de TREINO. Aqui phi do dataset inteiro seria vazamento:
   a regiao rara do teste entraria na definicao do que a busca persegue.

A diferenca entre as duas foi medida na auditoria da Pratica 3: a maior
diferenca absoluta em phi entre as duas referencias fica em 0,03 a 0,06
por fold, com pior caso de 0,17.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from metrics import relevance, sera

_SRC = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_SRC)
DATA_FOLDER = os.path.join(_RAIZ, "data")
RESULTS_FOLDER = os.path.join(_RAIZ, "results", "p4")

RANDOM_STATE = 42
TEST_SIZE = 0.30


def algoritmos_padrao(random_state=RANDOM_STATE):
    """
    Os m = 5 algoritmos da matriz P, com a configuracao PADRAO.

    Identicos aos da Atividade 1 (meta_dataset_builder.py), inclusive na
    semente. Reproduzi-los aqui em vez de importar o construtor da
    Atividade 1 e deliberado: aquele modulo carrega o pipeline de
    validacao cruzada inteiro, e aqui so se quer o dicionario.
    """
    return {
        "RF": RandomForestRegressor(random_state=random_state),
        "SVM": SVR(),
        "XGB": XGBRegressor(random_state=random_state, verbosity=0),
        "KNN": KNeighborsRegressor(),
        "DT": DecisionTreeRegressor(random_state=random_state),
    }


def carregar(nome, pasta=None):
    """Carrega um dataset. Alvo na PRIMEIRA coluna, convencao do projeto."""
    pasta = pasta or DATA_FOLDER
    df = pd.read_csv(os.path.join(pasta, f"{nome}.csv"))
    y = df.iloc[:, 0].to_numpy(dtype=float)
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    return X, y


def listar_datasets(pasta=None):
    pasta = pasta or DATA_FOLDER
    return sorted(f[:-4] for f in os.listdir(pasta) if f.endswith(".csv"))


def particionar(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    A particao unica do dataset, compartilhada por TODOS os metodos.

    Uma so chamada, uma so semente: qualquer metodo que chame esta
    funcao com o mesmo dataset recebe exatamente os mesmos indices. E o
    que garante a exigencia de "mesma particao de teste" sem precisar
    passar indices de um modulo a outro.

    Nao ha estratificacao. Em regressao nao existe classe para
    estratificar, e estratificar pela regiao rara (phi > 0,8) mudaria a
    proporcao de raros entre treino e teste em relacao ao dataset
    original -- o que alteraria o problema que se pretende medir.
    """
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, shuffle=True)


def avaliar_predicao(y_teste, y_predito, y_completo):
    """
    SERA no teste, com phi do dataset INTEIRO (ver decisao 1 no topo).

    Esta e a unica funcao que produz o numero comparado entre metodos.
    Todos os metodos passam por aqui, inclusive o AutoML.
    """
    phi = relevance(y_teste, y_reference=y_completo)
    return float(sera(y_teste, y_predito, phi))


def sera_de_busca(y_val, y_predito, y_treino):
    """
    SERA para a busca INTERNA, com phi do treino (ver decisao 2 no topo).

    Usada pelo warm-start e entregue ao FLAML como metrica customizada.
    """
    phi = relevance(y_val, y_reference=y_treino)
    return float(sera(y_val, y_predito, phi))


def treinar_avaliar(estimador, X_treino, y_treino, X_teste, y_teste,
                    y_completo):
    """
    Treina um estimador na particao de treino e mede na de teste.

    Reproduz o pipeline da Atividade 1 em tudo: StandardScaler em X
    ajustado SO no treino (sem isso SVM e KNN, sensiveis a escala,
    ficam injustamente prejudicados frente as arvores) e
    TransformedTargetRegressor no alvo (sem isso o SVR sofre em datasets
    de y com escala grande, porque epsilon e C pressupoem y numa faixa
    normal).

    Returns
    -------
    (sera_teste, segundos) -- o numero comparado e o custo de obte-lo.
    O tempo cobre ajuste e predicao, que e o que um usuario pagaria.
    """
    escala = StandardScaler()
    X_tr = escala.fit_transform(X_treino)
    X_te = escala.transform(X_teste)

    modelo = TransformedTargetRegressor(
        regressor=clone(estimador), transformer=StandardScaler())

    t0 = time.perf_counter()
    modelo.fit(X_tr, y_treino)
    y_predito = modelo.predict(X_te)
    segundos = time.perf_counter() - t0

    return avaliar_predicao(y_teste, y_predito, y_completo), segundos


def perda_relativa(tabela_sera):
    """
    Converte SERA bruta em perda relativa ao melhor resultado observado
    em cada dataset, que e o que a Analise 1 do enunciado pede.

    perda = (sera_do_metodo - melhor_do_dataset) / melhor_do_dataset

    Por que relativa e nao normalizada entre melhor e pior, como na
    Pratica 3: ali o enunciado pedia explicitamente a normalizacao entre
    L* e a pior da matriz; aqui ele pede "a perda em relacao ao melhor
    resultado observado em cada dataset". A razao tambem e mais robusta
    neste caso -- nao ha "pior" bem definido, ja que o conjunto de
    metodos comparados muda conforme o orcamento do AutoML entra ou sai.

    Parameters
    ----------
    tabela_sera : pd.DataFrame
        Linhas = datasets, colunas = metodos, valores = SERA no teste.
    """
    melhor = tabela_sera.min(axis=1)
    return tabela_sera.sub(melhor, axis=0).div(melhor.clip(lower=1e-12), axis=0)


if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    datasets = listar_datasets()
    print(f"{len(datasets)} datasets | particao {int((1-TEST_SIZE)*100)}/"
          f"{int(TEST_SIZE*100)}, semente {RANDOM_STATE}\n")

    linhas = []
    for nome in datasets:
        X, y = carregar(nome)
        X_tr, X_te, y_tr, y_te = particionar(X, y)
        phi_total = relevance(y, y_reference=y)
        phi_te = relevance(y_te, y_reference=y)
        linhas.append({
            "dataset": nome, "n": len(y), "d": X.shape[1],
            "n_treino": len(y_tr), "n_teste": len(y_te),
            "raros_total": int((phi_total > 0.8).sum()),
            "raros_teste": int((phi_te > 0.8).sum()),
            "pct_raros_total": round(100 * (phi_total > 0.8).mean(), 1),
            "pct_raros_teste": round(100 * (phi_te > 0.8).mean(), 1),
        })

    particoes = pd.DataFrame(linhas).set_index("dataset")
    particoes.to_csv(os.path.join(RESULTS_FOLDER, "particoes.csv"))
    print(particoes.to_string())

    # Diagnostico que decide se a particao e utilizavel: um teste sem
    # instancias raras tornaria a SERA degenerada naquele dataset.
    sem_raros = particoes[particoes["raros_teste"] == 0]
    print(f"\ndatasets cujo TESTE nao tem instancia rara: {len(sem_raros)}")
    if len(sem_raros):
        print(sem_raros[["n", "raros_total", "raros_teste"]].to_string())
        print("  -> nesses, a SERA de teste nao mede a regiao rara; declarar.")

    desvio = (particoes["pct_raros_teste"] - particoes["pct_raros_total"]).abs()
    print(f"\ndesvio na proporcao de raros (teste vs total): "
          f"mediana {desvio.median():.1f} pp | maximo {desvio.max():.1f} pp "
          f"({desvio.idxmax()})")
