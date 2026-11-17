"""
ETAPA 2 -- warm-starting da otimizacao de hiperparametros.

Implementa os cinco metodos do enunciado e o protocolo
leave-one-dataset-out. Como a matriz de perdas ja esta construida, cada
busca e SIMULADA: avaliar uma configuracao e consultar uma celula.

OS CINCO METODOS
----------------
1. padrao              configuracao padrao da biblioteca, sem busca
2. aleatoria           random search sobre as K, sem reposicao
3. smbo_frio           ninit pontos aleatorios, depois SMBO
4. smbo_similar        ninit = melhores configuracoes dos k datasets
                       mais proximos no espaco de meta-caracteristicas
                       (Feurer et al., 2014)
5. smbo_independente   ninit = configuracoes de melhor rank medio sobre
                       os datasets de treino, sem olhar para o alvo

O SMBO dos metodos 3 a 5 e o mesmo: mesmo modelo substituto e mesma
funcao de aquisicao. So os pontos iniciais mudam.

O metodo 5 e o controle que da sentido ao metodo 4. Se empatarem, o
ganho nao vem das meta-caracteristicas -- vem apenas de existirem
configuracoes boas em quase todo dataset. Separar essas duas explicacoes
e o ponto central da analise, e por isso o metodo 5 recebe exatamente o
mesmo tratamento que o 4.

O SMBO IMPLEMENTADO
-------------------
Segue o Algoritmo 6.2 do Capitulo 6 (Hutter et al., 2011): historico de
observacoes, modelo substituto, funcao de aquisicao, iteracao.

Modelo substituto: floresta aleatoria. O livro (Secao 6.3.1, "Random
forests as surrogate models of loss") registra que varios autores a
preferem por lidar bem com dados discretos e de alta dimensao, por
serem rapidas de treinar e por fornecerem estimativas de incerteza
utilizaveis; e a escolha do SMAC e do Auto-WEKA. Aqui ela cai bem
tambem porque o espaco de busca e discreto -- as K configuracoes --, o
que dispensaria um processo gaussiano.

Funcao de aquisicao: melhoria esperada (EI), na forma fechada que o
livro apresenta na Equacao 6.7 para substituto com estimativas
frequentistas de media e variancia:

    EI(theta) = sigma * [u * Phi(u) + phi(u)],  u = (Lmin - mu) / sigma

onde mu e sigma sao a media e o desvio-padrao das predicoes das arvores,
Phi a distribuicao acumulada e phi a densidade da normal padrao. Boas
candidatas sao as de baixa perda prevista (mu pequeno) OU de alta
incerteza (sigma grande) -- e a EI arbitra esse compromisso.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer

from hpo_space import ESPACO, N_INIT, T

METODOS = ["padrao", "aleatoria", "smbo_frio",
           "smbo_similar", "smbo_independente"]


# ---------------------------------------------------------------------
# Representacao das configuracoes para o modelo substituto
# ---------------------------------------------------------------------

def configuracoes_para_matriz(configuracoes):
    """
    Leva as configuracoes ao hipercubo unitario, aplicando log onde o
    espaco foi definido em log.

    Sem isso, n_estimators (20 a 500) dominaria learning_rate (0,005 a
    0,5) em qualquer modelo sensivel a escala, e as particoes da
    floresta ficariam concentradas na dimensao errada.
    """
    colunas = []
    for nome, spec in ESPACO.items():
        valores = configuracoes[nome].to_numpy(dtype=float)
        if spec["escala"] == "log":
            unitario = ((np.log(valores) - np.log(spec["min"]))
                        / (np.log(spec["max"]) - np.log(spec["min"])))
        else:
            unitario = (valores - spec["min"]) / (spec["max"] - spec["min"])
        colunas.append(unitario)
    return np.column_stack(colunas)


# ---------------------------------------------------------------------
# O SMBO
# ---------------------------------------------------------------------

def _melhoria_esperada(mu, sigma, melhor_perda):
    """
    EI na forma fechada da Equacao 6.7 do livro.

    Sigma zero significa que todas as arvores concordam: nao ha
    incerteza a explorar, e a EI e nula.
    """
    sigma = np.maximum(sigma, 1e-12)
    u = (melhor_perda - mu) / sigma
    return sigma * (u * norm.cdf(u) + norm.pdf(u))


def _limite_inferior(mu, sigma, melhor_perda, kappa=2.0):
    """
    A funcao de aquisicao usada na Aula 8: mu - 2*sigma, menor e melhor.

    O slide "Exemplo: escolhendo o proximo ponto" apresenta-a como "uma
    funcao de aquisicao simples" e ilustra a troca entre exploracao e
    explotacao com tres candidatos: theta_a (mu=0,14, sigma=0,01) seria
    escolhido pela perda prevista, mas theta_b (mu=0,20, sigma=0,06),
    "pouco testada", ganha quando a incerteza entra na conta.

    Retorna o negativo para que a convencao seja a mesma da EI --
    maximizar o valor devolvido --, e os dois criterios possam ser
    trocados sem mexer no laco da busca.
    """
    return -(mu - kappa * np.maximum(sigma, 0.0))


AQUISICOES = {"ei": _melhoria_esperada, "lcb": _limite_inferior}


def executar_smbo(perdas_alvo, matriz_configuracoes, indices_iniciais,
                  orcamento=T, random_state=0, n_arvores=50,
                  aquisicao="ei"):
    """
    Uma busca SMBO sobre as K configuracoes.

    Parameters
    ----------
    perdas_alvo : np.ndarray (K,)
        Coluna da matriz de perdas correspondente ao dataset alvo. E o
        oraculo: avaliar a configuracao i e ler perdas_alvo[i].
    matriz_configuracoes : np.ndarray (K, d)
        Configuracoes no hipercubo unitario.
    indices_iniciais : lista de int
        Os ninit pontos iniciais. E so aqui que os cinco metodos diferem.
    orcamento : int
        T avaliacoes no total, contando os pontos iniciais.
    aquisicao : {"ei", "lcb"}
        "ei" e a Melhoria Esperada da Eq. 6.7 do livro; "lcb" e o
        mu - 2*sigma apresentado na Aula 8. O enunciado exige apenas que
        os metodos 3 a 5 usem a MESMA aquisicao, nao qual delas; o padrao
        e EI por ser a do livro-base, e a alternativa existe para que o
        relatorio possa mostrar que a conclusao nao depende da escolha.

    Returns
    -------
    (historico, perdas) -- indices avaliados na ordem e suas perdas.
    """
    rng = np.random.default_rng(random_state)
    K = len(perdas_alvo)
    criterio = AQUISICOES[aquisicao]

    historico = list(dict.fromkeys(int(i) for i in indices_iniciais))[:orcamento]
    perdas = [float(perdas_alvo[i]) for i in historico]

    while len(historico) < orcamento:
        restantes = np.setdiff1d(np.arange(K), historico)
        if len(restantes) == 0:
            break

        # Com uma unica observacao a floresta nao tem o que modelar;
        # sorteia-se ate haver historico suficiente.
        if len(historico) < 2:
            escolhido = int(rng.choice(restantes))
        else:
            substituto = RandomForestRegressor(
                n_estimators=n_arvores, random_state=random_state, n_jobs=-1)
            substituto.fit(matriz_configuracoes[historico], perdas)

            por_arvore = np.stack([
                arvore.predict(matriz_configuracoes[restantes])
                for arvore in substituto.estimators_
            ])
            mu = por_arvore.mean(axis=0)
            sigma = por_arvore.std(axis=0)

            valor = criterio(mu, sigma, min(perdas))
            escolhido = int(restantes[int(np.argmax(valor))])

        historico.append(escolhido)
        perdas.append(float(perdas_alvo[escolhido]))

    return historico, perdas


# ---------------------------------------------------------------------
# Os pontos iniciais de cada metodo
# ---------------------------------------------------------------------

def iniciais_aleatorios(K, n_init=N_INIT, random_state=0):
    """Metodo 3 -- partida fria."""
    rng = np.random.default_rng(random_state)
    return list(rng.choice(K, size=min(n_init, K), replace=False))


def iniciais_por_similaridade(perdas_treino, X_meta_treino, x_meta_alvo,
                              n_init=N_INIT, k_vizinhos=3):
    """
    Metodo 4 -- warm-start por datasets similares (Feurer et al., 2014).

    Os k datasets mais proximos do alvo no espaco de meta-caracteristicas
    doam suas melhores configuracoes. A premissa, nas palavras do livro
    (Secao 6.4.1): "configurations that have performed well on datasets
    similar to the current dataset will also perform well on the current
    dataset".

    Distancia: norma L1 sobre meta-caracteristicas normalizadas. Feurer
    et al. (2015b) propoem duas funcoes de distancia, e esta -- uma
    p-norma entre meta-caracteristicas -- e a primeira delas.

    Quando k nao coincide com ninit, percorre-se os vizinhos em ordem de
    proximidade tomando suas melhores configuracoes em rodizio, ate
    reunir ninit configuracoes distintas. Com k=5 e ninit=5 isso da a
    melhor de cada vizinho; com k=1, as cinco melhores do vizinho unico.

    O procedimento canonico ACOPLA k e ninit: uma configuracao por
    vizinho. O exemplo da Aula 8 e explicito -- "Os vizinhos mais
    proximos de dt sao d1 e d3. O SMBO comeca pelas melhores
    configuracoes deles na tabela de perdas -- theta1 e theta2 -- em vez
    de pontos aleatorios" -- dois vizinhos, dois pontos iniciais. E o
    Auto-sklearn 1.0, como descrito em MetaML (Maia et al., 2025, Secao
    3.1), faz o mesmo: "the meta-learner selects the k configurations to
    be used by optimization [...] the instances are selected considering
    the k closest datasets".

    O enunciado, porem, exige os mesmos T e ninit para os cinco metodos
    e pede para variar k, o que desacopla os dois. Os dois arranjos tem
    defeito simetrico -- com ninit fixo, variar k confunde numero de
    vizinhos com profundidade no ranking de cada um; com ninit=k,
    confunde numero de vizinhos com divisao do orcamento entre pontos
    iniciais e passos guiados pelo modelo. A analise roda os dois
    (ver analysis_hpo.variar_vizinhos).
    """
    distancias = np.abs(X_meta_treino - x_meta_alvo).sum(axis=1)
    ordem_vizinhos = np.argsort(distancias)[:k_vizinhos]

    # ranking das configuracoes dentro de cada vizinho
    rankings = [
        np.argsort(perdas_treino[:, vizinho]) for vizinho in ordem_vizinhos
    ]

    escolhidos, posicao = [], 0
    while len(escolhidos) < n_init and posicao < perdas_treino.shape[0]:
        for ranking in rankings:
            if posicao < len(ranking):
                candidato = int(ranking[posicao])
                if candidato not in escolhidos:
                    escolhidos.append(candidato)
                    if len(escolhidos) == n_init:
                        break
        posicao += 1

    return escolhidos


def iniciais_por_rank_medio(perdas_treino, n_init=N_INIT):
    """
    Metodo 5 -- warm-start independente dos dados.

    As configuracoes de melhor rank medio sobre todos os datasets de
    treino, sem olhar para o alvo. E o metodo do rank medio (AR) da
    Atividade 1 e do Capitulo 2 (Eq. 2.2), aplicado a configuracoes em
    vez de algoritmos: ranquear por dataset, mediar os ranks, reordenar.

    Ranquear antes de mediar e o que torna datasets de escalas
    diferentes comparaveis -- a SERA varia por ordens de magnitude entre
    datasets, e mediar perdas brutas deixaria um unico dataset decidir.
    """
    ranks = pd.DataFrame(perdas_treino).rank(axis=0, method="average")
    rank_medio = ranks.mean(axis=1)
    return list(rank_medio.nsmallest(n_init).index.astype(int))


def diagnosticar_iniciais(matriz_perdas, configuracoes, X_meta,
                          n_init=N_INIT, k_vizinhos=3, random_state=0):
    """
    Caracteriza os pontos iniciais que cada metodo escolhe, por dois eixos.

    REDUNDANCIA (dispersao interna). Distancia media entre os ninit
    pontos de uma mesma escolha, no hipercubo unitario. Baixa dispersao
    significa que o metodo gastou seus pontos iniciais em configuracoes
    quase iguais entre si, cobrindo pouco do espaco.

    O alerta vem do proprio material. O Capitulo 5, ao criticar o metodo
    do rank medio, observa que ele "just follows the ranking, and hence
    is unable to skip very similar algorithms", e completa: "This problem
    is quite common, as similar variants can arise for instance when
    considering algorithms that include parameters, which may be set to
    different values. Quite often, many of the settings have a limited
    impact on performance". MetaML (Maia et al., 2025, Secao 1) faz a
    analogia explicita com HPO: "small variations in certain
    hyperparameters often do not significantly affect performance on
    certain datasets, which can lead to redundant configurations".

    DEPENDENCIA DOS DADOS (cobertura). Numero de configuracoes distintas
    que o metodo usa ao longo dos n datasets alvo. O metodo 5 e
    data-independent por construcao: retirar uma de n-1 colunas quase nao
    altera o rank medio, de modo que ele tende a repetir o mesmo punhado
    de configuracoes em todos os alvos. O metodo 4 varia com o alvo. A
    diferenca de cobertura mede o quanto as meta-caracteristicas estao de
    fato personalizando a escolha -- a pergunta da Analise 2.

    Returns
    -------
    pd.DataFrame indexado por metodo, com dispersao media e cobertura.
    """
    matriz_cfg = configuracoes_para_matriz(configuracoes)
    n_datasets = matriz_perdas.shape[1]
    K = matriz_perdas.shape[0]

    def dispersao(indices):
        if len(indices) < 2:
            return np.nan
        pontos = matriz_cfg[indices]
        distancias = [
            np.linalg.norm(pontos[i] - pontos[j])
            for i in range(len(pontos)) for j in range(i + 1, len(pontos))
        ]
        return float(np.mean(distancias))

    registros = {m: {"dispersoes": [], "usadas": set()}
                 for m in ("aleatoria", "smbo_similar", "smbo_independente")}

    for indice in range(n_datasets):
        perdas_treino = np.delete(matriz_perdas.to_numpy(), indice, axis=1)
        X_treino, x_alvo = normalizar_meta_features(X_meta, indice)

        escolhas = {
            "aleatoria": iniciais_aleatorios(K, n_init, random_state + indice),
            "smbo_similar": iniciais_por_similaridade(
                perdas_treino, X_treino, x_alvo, n_init, k_vizinhos),
            "smbo_independente": iniciais_por_rank_medio(
                perdas_treino, n_init),
        }
        for metodo, indices in escolhas.items():
            registros[metodo]["dispersoes"].append(dispersao(indices))
            registros[metodo]["usadas"].update(indices)

    linhas = {
        metodo: {
            "dispersao_media": float(np.nanmean(dados["dispersoes"])),
            "configuracoes_distintas": len(dados["usadas"]),
            "razao_sobre_maximo": len(dados["usadas"]) / (n_init * n_datasets),
        }
        for metodo, dados in registros.items()
    }
    tabela = pd.DataFrame(linhas).T
    tabela.index.name = "metodo"
    return tabela


# ---------------------------------------------------------------------
# Protocolo leave-one-dataset-out
# ---------------------------------------------------------------------

def normalizar_meta_features(X_meta, indice_alvo, metodo="quantil"):
    """
    Normaliza as meta-caracteristicas EXCLUINDO o dataset alvo.

    O enunciado e explicito: o alvo sai da matriz de perdas E da
    normalizacao das meta-caracteristicas. Incluir o alvo na media e no
    desvio usados para padronizar seria vazamento -- a posicao relativa
    dos vizinhos passaria a depender, ainda que pouco, do proprio alvo.
    Os dois metodos abaixo ajustam so no treino e aplicam ao alvo, de
    modo que ambos respeitam o LODO.

    metodo="zscore" -- padronizacao classica. QUEBRA neste conjunto.
        Diagnostico medido: a distancia L1 de space_ga ao seu vizinho
        mais proximo e 513,4, e 98% dela vem de duas meta-features --
        mean_of_means, com z = -319,6, e mean_of_stds, com z = +187,7.
        Nao e divisao por desvio quase-nulo (o menor desvio no treino e
        2,7e-02); e que essas duas tem cauda pesada entre datasets, de
        modo que um dataset de escala muito diferente vai para centenas
        de desvios e passa a decidir sozinho quem e o vizinho. Para os
        casos extremos, a vizinhanca fica determinada pela escala dos
        preditores e nao pela estrutura do problema.

    metodo="quantil" -- transformacao por quantis, o padrao. Cada
        meta-feature e mapeada para a sua posicao relativa em [0,1],
        de modo que nenhuma pode contribuir mais de 1 para a distancia
        L1 e a cauda pesada deixa de dominar.

    A escolha segue MetaML (Maia et al., 2025, Secao 5.4), que trata o
    mesmo problema com a mesma ferramenta: "The values of the
    meta-features used in this study vary in different intervals and may
    present outliers, therefore the Quantile Transformer scaling
    algorithm was employed because of its robustness to outliers".

    O relatorio reporta a Analise 3 nas duas normalizacoes, porque a
    conclusao sobre a relacao entre similaridade e ganho depende de a
    distancia medir o que se pretende que ela meca.
    """
    treino = np.delete(X_meta, indice_alvo, axis=0)
    alvo = X_meta[indice_alvo]

    if metodo == "zscore":
        media = treino.mean(axis=0)
        desvio = treino.std(axis=0)
        desvio[desvio == 0] = 1.0
        return (treino - media) / desvio, (alvo - media) / desvio

    if metodo == "quantil":
        transformador = QuantileTransformer(
            n_quantiles=min(len(treino), 1000),
            output_distribution="uniform",
            subsample=10_000, random_state=42,
        )
        treino_t = transformador.fit_transform(treino)
        alvo_t = transformador.transform(alvo.reshape(1, -1))[0]
        return treino_t, alvo_t

    raise ValueError(f"metodo desconhecido: {metodo}")


def executar_todos(matriz_perdas, configuracoes, X_meta,
                   orcamento=T, n_init=N_INIT, k_vizinhos=3,
                   sementes=range(10), metodos=None, verbose=False,
                   aquisicao="ei", matriz_tempos=None):
    """
    Roda os cinco metodos em todos os datasets, sob LODO e varias
    sementes.

    Parameters
    ----------
    matriz_perdas : pd.DataFrame (K x n)
    configuracoes : pd.DataFrame (K x hiperparametros)
    X_meta : pd.DataFrame (n x meta-caracteristicas)
        Indexada pelos mesmos nomes de dataset que as colunas da matriz.
    sementes : iteravel
        A busca e simulada, entao repetir e praticamente gratuito. Varias
        sementes sao necessarias porque as diferencas entre metodos sao
        da ordem da variancia entre execucoes.
    matriz_tempos : pd.DataFrame (K x n), opcional
        Tempo medio de treino por celula, produzido por
        build_loss_matrix. Quando fornecida, cada linha ganha o tempo
        acumulado da busca ate aquela avaliacao, o que permite medir o
        orcamento em SEGUNDOS e nao em numero de avaliacoes.

    Returns
    -------
    DataFrame longo: uma linha por (dataset, metodo, semente, avaliacao).
    Colunas de perda do incumbente:
        perda_bruta -- SERA, na escala do dataset
        perda_norm  -- (incumbente - L*) / (pior - L*), o que o enunciado
                       pede: linear entre o otimo e a pior da matriz
        perda_rank  -- (rank do incumbente - 1) / (K - 1), de 0 a 1

    POR QUE AS DUAS NORMALIZACOES
    A linear e a do enunciado e preserva magnitude, mas colapsa quando a
    dispersao do dataset e extrema: com a pior configuracao 1145x pior
    que a otima (caso de airfoild), uma configuracao 2x pior que a otima
    normaliza para 0,0009 -- praticamente zero. Esse dataset deixa de
    contribuir para a media entre datasets, enquanto um de dispersao 1,2x
    contribui em escala cheia.

    A Aula 8 discute exatamente essa troca ao apresentar o SCoT. Sem
    normalizar, "o surrogate aprenderia 'qual dataset e facil', e nao
    'qual configuracao e boa'"; com ranking, "o ranking descarta a escala
    absoluta, mas preserva a informacao de quais configuracoes sao
    melhores ou piores" -- ao preco de que "o ranking so sabe que theta1
    > theta2 > theta3. Nao sabe que theta1 e theta2 sao quase iguais
    enquanto theta3 e muito pior". As duas colunas deixam o relatorio
    mostrar que a conclusao nao depende da escolha.
    """
    metodos = metodos or METODOS
    datasets = list(matriz_perdas.columns)
    X_meta = X_meta.loc[datasets]

    matriz_cfg = configuracoes_para_matriz(configuracoes)
    perdas_np = matriz_perdas.to_numpy(dtype=float)
    X_meta_np = X_meta.to_numpy(dtype=float)
    indice_padrao = list(matriz_perdas.index).index("cfg_00")

    if matriz_tempos is not None:
        tempos_alinhados = matriz_tempos.reindex(
            index=matriz_perdas.index, columns=datasets)
    else:
        tempos_alinhados = None

    linhas = []
    for i_alvo, alvo in enumerate(datasets):
        perdas_alvo = perdas_np[:, i_alvo]
        perdas_treino = np.delete(perdas_np, i_alvo, axis=1)
        X_treino, x_alvo = normalizar_meta_features(X_meta_np, i_alvo)

        # Normalizacao da curva: 0 = perda otima do dataset, 1 = pior
        # configuracao da matriz. Sem isso, datasets de SERA grande
        # dominariam a media sobre datasets.
        melhor, pior = perdas_alvo.min(), perdas_alvo.max()
        amplitude = max(pior - melhor, 1e-12)

        # Rank de cada configuracao no dataset alvo, de 1 (melhor) a K.
        ranks_alvo = pd.Series(perdas_alvo).rank(method="average").to_numpy()
        divisor_rank = max(len(perdas_alvo) - 1, 1)

        tempos_alvo = (tempos_alinhados[alvo].to_numpy(dtype=float)
                       if tempos_alinhados is not None
                       and alvo in tempos_alinhados.columns
                       and tempos_alinhados[alvo].notna().any() else None)

        for metodo in metodos:
            for semente in sementes:
                if metodo == "padrao":
                    # Sem busca: a mesma configuracao em toda avaliacao.
                    historico = [indice_padrao] * orcamento
                elif metodo == "aleatoria":
                    rng = np.random.default_rng(semente)
                    historico = list(rng.choice(
                        len(perdas_alvo), size=orcamento, replace=False))
                else:
                    if metodo == "smbo_frio":
                        iniciais = iniciais_aleatorios(
                            len(perdas_alvo), n_init, random_state=semente)
                    elif metodo == "smbo_similar":
                        iniciais = iniciais_por_similaridade(
                            perdas_treino, X_treino, x_alvo,
                            n_init=n_init, k_vizinhos=k_vizinhos)
                    else:
                        iniciais = iniciais_por_rank_medio(
                            perdas_treino, n_init=n_init)
                    historico, _ = executar_smbo(
                        perdas_alvo, matriz_cfg, iniciais,
                        orcamento=orcamento, random_state=semente,
                        aquisicao=aquisicao)

                indices = np.array(historico, dtype=int)

                # incumbente: melhor perda ate a i-esima avaliacao
                perdas_vistas = perdas_alvo[indices]
                incumbente = np.minimum.accumulate(perdas_vistas)
                normalizada = (incumbente - melhor) / amplitude

                # rank do incumbente, no mesmo esquema acumulado
                rank_incumbente = np.minimum.accumulate(ranks_alvo[indices])
                rank_norm = (rank_incumbente - 1.0) / divisor_rank

                if tempos_alvo is not None:
                    acumulado = np.cumsum(tempos_alvo[indices])
                else:
                    acumulado = np.full(len(indices), np.nan)

                for avaliacao in range(1, len(indices) + 1):
                    j = avaliacao - 1
                    linhas.append({
                        "dataset": alvo, "metodo": metodo,
                        "semente": semente, "avaliacao": avaliacao,
                        "perda_norm": float(normalizada[j]),
                        "perda_rank": float(rank_norm[j]),
                        "perda_bruta": float(incumbente[j]),
                        "configuracao": int(indices[j]),
                        "segundos_acumulados": float(acumulado[j]),
                    })

        if verbose:
            print(f"  {alvo}", flush=True)

    return pd.DataFrame(linhas)
