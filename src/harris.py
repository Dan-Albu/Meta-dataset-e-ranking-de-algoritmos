"""
HARRIS -- Hybrid rAnking and RegRessIon foreSts
Implementacao fiel a Fehring, Hanselle & Tornede (2022), NeurIPS MetaLearn
Workshop, combinada com os slides da Aula 4 (slide 25) e o enunciado da
atividade (Secao 2.2, item 6).

--------------------------------------------------------------------------
NOTA IMPORTANTE SOBRE O lambda (documente isso no relatorio):
O paper apresenta DUAS formulas diferentes e invertidas entre si:
  - Eq. 3 (Secao 3, citando trabalho anterior de Hanselle et al.):
        l_lambda = lambda * l_regressao + (1 - lambda) * l_ranking
  - Secao 4 (onde o HARRIS e definido de fato):
        L(D)     = lambda * L_ranking  + (1 - lambda) * L_regressao

Seguimos a SEGUNDA, porque:
  (a) e a formula da secao que define o HARRIS propriamente dito;
  (b) e a que os slides da Aula 4 (slide 25) usam, com a nota explicita
      "lambda = 0 -> floresta de regressao pura; lambda = 1 -> ranking puro";
  (c) e a que o enunciado da atividade pede.
--------------------------------------------------------------------------

Os 6 elementos que o paper especifica, e onde estao implementados aqui:
  1. Dois rotulos por no (regressao = media; ranking = consenso de Borda)
     -> _regression_label() e _ranking_label()
  2. Escolha do split pela soma ponderada pelos tamanhos (Eq. 4)
     -> _find_best_split()
  3. Perda de regressao = MSE sobre todos algoritmos e instancias
     -> _regression_loss()
  4. Perda de ranking = 1 - correlacao de Spearman
     -> _ranking_loss()
  5. Criterio de parada = profundidade da arvore
     -> _build_tree(), parametro max_depth
  6. Na predicao, usa APENAS o rotulo de regressao da folha
     -> HybridTree.predict()

Escalonamento das perdas: o paper alerta (fim da Secao 4) que "a diferenca
de escala entre as perdas de ranking e regressao pode fazer uma dominar a
outra, anulando o efeito do lambda". Resolvemos como eles: (a) a perda de
ranking e dividida pela perda maxima possivel (2, pois 1 - Spearman varia
em [0, 2]); (b) a matriz P e normalizada por linha para [0, 1] antes do
treino -- ver normalize_performance_matrix(). Essa normalizacao tambem e
recomendada de forma independente pelo Cap. 3 do livro (Secao 3.3,
"Rescaling into 0-1 interval") para comparar performances entre datasets.
"""

import numpy as np
from scipy.stats import spearmanr  # usado so em testes/inspecao; a perda
                                    # de ranking usa Pearson vetorizado


def normalize_performance_matrix(P):
    """
    Normaliza a matriz de performances por LINHA para o intervalo [0, 1].

    Por que e necessario: a SERA varia enormemente entre datasets (no nosso
    meta-dataset, de ~0.15 a ~1e12). Sem normalizar, a perda de regressao
    seria inteiramente dominada por 2-3 datasets de escala gigante, e o
    lambda perderia o efeito -- exatamente o problema que o paper descreve.

    Parameters
    ----------
    P : np.ndarray, shape (n_datasets, n_algoritmos)

    Returns
    -------
    np.ndarray : mesma forma, cada linha reescalada para [0, 1].
        A ORDEM dentro da linha e preservada (menor continua menor), entao
        o ranking derivado dela e identico ao da matriz original.
    """
    P = np.asarray(P, dtype=float)
    minimos = P.min(axis=1, keepdims=True)
    maximos = P.max(axis=1, keepdims=True)
    amplitude = maximos - minimos
    # Linha constante (todos algoritmos identicos): evita divisao por zero
    amplitude[amplitude == 0] = 1.0
    return (P - minimos) / amplitude


class HybridTree:
    """
    Uma arvore hibrida de ranking e regressao (o componente basico do HARRIS).

    Diferente de uma arvore de decisao comum, cada no guarda DOIS rotulos e
    o split e escolhido por uma combinacao convexa de duas perdas.

    Parameters
    ----------
    lambda_param : float em [0, 1]
        Peso da perda de RANKING. lambda=0 -> regressao pura;
        lambda=1 -> ranking puro (conforme slide 25).
    max_depth : int
        Profundidade maxima -- o criterio de parada usado pelo paper.
    min_samples_split : int
        Minimo de instancias num no para tentar dividi-lo.
    max_features : int ou None
        Quantos atributos sortear em cada split (como numa random forest
        padrao). None = usa todos.
    random_state : int ou None
        Semente para o sorteio de atributos.
    """

    def __init__(self, lambda_param=0.5, max_depth=4, min_samples_split=2,
                 max_features=None, random_state=None):
        self.lambda_param = lambda_param
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.tree_ = None

    # ------------------------------------------------------------------
    # Rotulos de um no (paper, Secao 4)
    # ------------------------------------------------------------------

    @staticmethod
    def _regression_label(Y):
        """
        Rotulo de REGRESSAO de um no: a media das performances das
        instancias que caem nele (paper: "obtained by averaging the
        labels in the associated dataset D").

        Returns
        -------
        np.ndarray, shape (k,) -- uma performance media por algoritmo.
        """
        return Y.mean(axis=0)

    @staticmethod
    def _ranking_label_from_ranks(Y_ranks):
        """
        Rotulo de RANKING de um no via metodo de Borda, a partir dos ranks
        JA pre-computados (ver nota de performance em fit()).

        Como funciona o metodo de Borda aqui: cada instancia (dataset)
        "vota" ranqueando os k algoritmos; mediamos as posicoes de cada
        algoritmo entre todos os votantes; o consenso e o ranking dessas
        medias.

        Curiosidade util para o relatorio: isso e exatamente a mesma logica
        da abordagem AR (rank medio) da Secao 2.1 -- so que aplicada dentro
        de cada no da arvore, em vez de no meta-dataset inteiro.
        """
        return _rankdata_ascending(Y_ranks.mean(axis=0))

    @staticmethod
    def _ranking_label(Y):
        """
        Versao "de conveniencia" do rotulo de Borda, calculando os ranks na
        hora. Usada fora do laco critico (ex.: inspecao de folhas, testes).
        """
        ranks = np.apply_along_axis(_rankdata_ascending, 1, Y)
        return _rankdata_ascending(ranks.mean(axis=0))

    # ------------------------------------------------------------------
    # As duas perdas (paper, Secao 4)
    # ------------------------------------------------------------------

    @staticmethod
    def _regression_loss(Y, rotulo_regressao):
        """
        Perda de REGRESSAO: erro quadratico medio entre as performances
        reais das instancias do no e o rotulo de regressao do no
        (paper: "the mean squared error over all algorithms and instances").

        Mede o quao HOMOGENEAS sao as performances dentro do no: se todas
        as instancias tem performances parecidas, a media as representa bem
        e o MSE e baixo.
        """
        return float(np.mean((Y - rotulo_regressao) ** 2))

    @staticmethod
    def _ranking_loss_from_ranks(Y_ranks, rotulo_ranking):
        """
        Perda de RANKING: 1 - correlacao de Spearman entre o ranking de
        cada instancia e o ranking de consenso do no (paper: "the Spearman
        correlation turned into a loss function by subtracting it from 1").

        Normalizada dividindo por 2, que e a perda maxima possivel
        (Spearman varia em [-1, 1], entao 1 - Spearman varia em [0, 2]) --
        e o escalonamento ao intervalo unitario que o paper exige para o
        lambda ter efeito real.

        Mede o quao homogeneas sao as ORDENS dentro do no, ignorando as
        magnitudes: duas instancias com performances muito diferentes mas
        que ordenam os algoritmos igual tem perda de ranking zero.

        NOTA DE IMPLEMENTACAO: a correlacao de Spearman entre dois vetores
        e, por definicao, a correlacao de Pearson entre seus RANKS. Como
        aqui as duas entradas ja sao ranks, calculamos Pearson diretamente
        e de forma vetorizada (todas as instancias de uma vez), em vez de
        chamar scipy.stats.spearmanr instancia por instancia. Isso e
        matematicamente identico e ordens de magnitude mais rapido -- o
        que importa porque esta funcao roda dentro da busca de splits,
        o laco mais quente do algoritmo.
        """
        n = len(Y_ranks)
        if n < 2:
            # Um unico elemento: o consenso e ele mesmo, perda zero.
            return 0.0

        c = np.asarray(rotulo_ranking, dtype=float)
        c_centrado = c - c.mean()
        denom_c = np.sqrt((c_centrado ** 2).sum())
        if denom_c == 0:
            # Consenso constante (todos empatados no Borda): Spearman
            # indefinido -> tratamos como correlacao nula.
            return 0.5  # = 1.0 / 2 (normalizacao)

        R_centrado = Y_ranks - Y_ranks.mean(axis=1, keepdims=True)
        denom_R = np.sqrt((R_centrado ** 2).sum(axis=1))

        numerador = (R_centrado * c_centrado).sum(axis=1)
        denominador = denom_R * denom_c

        # Instancias com ranks constantes (todos algoritmos empatados):
        # correlacao indefinida -> tratada como nula.
        correlacoes = np.zeros(n, dtype=float)
        validos = denominador > 0
        correlacoes[validos] = numerador[validos] / denominador[validos]

        return float(np.mean(1.0 - correlacoes) / 2.0)

    def _node_loss(self, Y, Y_ranks):
        """
        Perda hibrida de um no (paper, Secao 4):
            L(D) = lambda * L_ranking(D) + (1 - lambda) * L_regressao(D)

        Note que o lambda multiplica o RANKING (ver nota no topo do
        arquivo sobre a divergencia entre Eq. 3 e Secao 4 do paper).
        """
        if len(Y) == 0:
            return 0.0

        perda = 0.0
        # So calcula cada perda se ela tiver peso -- economiza tempo
        if self.lambda_param > 0:
            perda += self.lambda_param * self._ranking_loss_from_ranks(
                Y_ranks, self._ranking_label_from_ranks(Y_ranks)
            )
        if self.lambda_param < 1:
            perda += (1 - self.lambda_param) * self._regression_loss(
                Y, self._regression_label(Y)
            )
        return perda

    # ------------------------------------------------------------------
    # Construcao da arvore
    # ------------------------------------------------------------------

    def _find_best_split(self, X, Y, Y_ranks, rng):
        """
        Busca o melhor par (atributo, ponto de corte) minimizando a soma
        ponderada pelos tamanhos dos nos filhos -- Eq. 4 do paper:

            (|D+|/|D|) * L(D+) + (|D-|/|D|) * L(D-)

        O paper resolve isso por "simple enumeration of all possible
        features and splitting points imposed by the training data" --
        e exatamente o que fazemos: para cada atributo sorteado, testamos
        os pontos medios entre valores consecutivos distintos.

        Returns
        -------
        (indice_atributo, limiar, perda) ou (None, None, inf) se nao ha
        split valido.
        """
        n_amostras, n_atributos = X.shape

        # Sorteio de atributos, como numa random forest padrao
        if self.max_features is not None and self.max_features < n_atributos:
            atributos = rng.choice(n_atributos, self.max_features, replace=False)
        else:
            atributos = np.arange(n_atributos)

        melhor_atributo, melhor_limiar, melhor_perda = None, None, np.inf

        for atributo in atributos:
            valores = np.unique(X[:, atributo])
            if len(valores) < 2:
                continue  # atributo constante, nao ha como dividir

            # pontos de corte = medios entre valores consecutivos distintos
            limiares = (valores[:-1] + valores[1:]) / 2.0

            for limiar in limiares:
                mascara_esq = X[:, atributo] <= limiar
                n_esq = mascara_esq.sum()
                n_dir = n_amostras - n_esq
                if n_esq == 0 or n_dir == 0:
                    continue

                perda = (
                    (n_esq / n_amostras)
                    * self._node_loss(Y[mascara_esq], Y_ranks[mascara_esq])
                    + (n_dir / n_amostras)
                    * self._node_loss(Y[~mascara_esq], Y_ranks[~mascara_esq])
                )

                if perda < melhor_perda:
                    melhor_atributo, melhor_limiar, melhor_perda = (
                        atributo, limiar, perda
                    )

        return melhor_atributo, melhor_limiar, melhor_perda

    def _build_tree(self, X, Y, Y_ranks, profundidade, rng):
        """
        Constroi a arvore recursivamente. Criterio de parada: profundidade
        maxima (como no paper), poucas amostras, ou nenhum split valido.

        Cada no folha guarda o rotulo de regressao (usado na predicao) e
        o de ranking (guardado para inspecao/estudo, mas nao usado na
        predicao -- ver docstring de predict()).
        """
        no = {
            "rotulo_regressao": self._regression_label(Y),
            "rotulo_ranking": self._ranking_label_from_ranks(Y_ranks),
            "n_amostras": len(Y),
            "folha": True,
        }

        if profundidade >= self.max_depth or len(Y) < self.min_samples_split:
            return no

        atributo, limiar, _ = self._find_best_split(X, Y, Y_ranks, rng)
        if atributo is None:
            return no

        mascara_esq = X[:, atributo] <= limiar
        no.update({
            "folha": False,
            "atributo": atributo,
            "limiar": limiar,
            "esquerda": self._build_tree(
                X[mascara_esq], Y[mascara_esq], Y_ranks[mascara_esq],
                profundidade + 1, rng
            ),
            "direita": self._build_tree(
                X[~mascara_esq], Y[~mascara_esq], Y_ranks[~mascara_esq],
                profundidade + 1, rng
            ),
        })
        return no

    def fit(self, X, Y):
        """
        Treina a arvore.

        NOTA DE PERFORMANCE: os ranks de cada linha de Y sao calculados UMA
        VEZ aqui e propagados pela recursao, em vez de recalculados a cada
        avaliacao de split. Como a busca de splits testa
        (n_atributos x n_pontos_de_corte) combinacoes por no, recalcular os
        ranks toda vez seria o gargalo do algoritmo.

        Parameters
        ----------
        X : np.ndarray, shape (n_datasets, n_meta_features)
        Y : np.ndarray, shape (n_datasets, n_algoritmos)
            Performances JA NORMALIZADAS por linha (use
            normalize_performance_matrix()).
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        Y_ranks = np.apply_along_axis(_rankdata_ascending, 1, Y)

        rng = np.random.default_rng(self.random_state)
        self.tree_ = self._build_tree(X, Y, Y_ranks, 0, rng)
        return self

    def predict(self, X):
        """
        Prediz as performances de novas instancias.

        Conforme o paper: "we propagate the instance down the tree until a
        leaf node is reached. Based on label y^regression we finally return
        the algorithm performing best according to this label."

        Ou seja: na predicao usa-se APENAS o rotulo de regressao. O rotulo
        de ranking existe so durante o TREINO, para compor a perda que
        guia a escolha dos splits.

        Returns
        -------
        np.ndarray, shape (n_amostras, n_algoritmos)
        """
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_uma(linha) for linha in X])

    def _predict_uma(self, x):
        no = self.tree_
        while not no["folha"]:
            if x[no["atributo"]] <= no["limiar"]:
                no = no["esquerda"]
            else:
                no = no["direita"]
        return no["rotulo_regressao"]


class HarrisForest:
    """
    A floresta HARRIS: um conjunto de HybridTree, construida "analogamente
    a random forests padrao" (paper, Secao 4) -- ou seja, com bootstrap das
    instancias e sorteio de atributos em cada split.

    Parameters
    ----------
    n_trees : int
        Numero de arvores na floresta.
    lambda_param : float em [0, 1]
        Peso da perda de ranking (mesma semantica da HybridTree).
    max_depth, min_samples_split : ver HybridTree.
    max_features : int, "sqrt" ou None
        Atributos sorteados por split. "sqrt" = raiz do total (padrao
        classico de random forest).
    random_state : int
    """

    def __init__(self, n_trees=50, lambda_param=0.5, max_depth=4,
                 min_samples_split=2, max_features="sqrt", random_state=42):
        self.n_trees = n_trees
        self.lambda_param = lambda_param
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X, Y):
        """
        Treina a floresta. Cada arvore ve uma amostra bootstrap diferente
        das instancias (amostragem com reposicao), o que gera a diversidade
        que faz o ensemble funcionar.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n_amostras, n_atributos = X.shape

        if self.max_features == "sqrt":
            max_feat = max(1, int(np.sqrt(n_atributos)))
        else:
            max_feat = self.max_features

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for i in range(self.n_trees):
            indices = rng.choice(n_amostras, n_amostras, replace=True)
            arvore = HybridTree(
                lambda_param=self.lambda_param,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            arvore.fit(X[indices], Y[indices])
            self.trees_.append(arvore)
        return self

    def predict(self, X):
        """
        Predicao da floresta: media das predicoes das arvores (agregacao
        padrao de random forest para regressao).

        Returns
        -------
        np.ndarray, shape (n_amostras, n_algoritmos)
        """
        predicoes = np.array([arvore.predict(X) for arvore in self.trees_])
        return predicoes.mean(axis=0)


def _rankdata_ascending(valores):
    """
    Ranking com empates recebendo rank medio, em ordem CRESCENTE
    (menor valor = rank 1) -- mesma convencao usada em toda a atividade,
    ja que a SERA e uma metrica de erro.

    Implementado aqui em numpy puro para nao depender de scipy.stats.rankdata
    dentro do laco mais quente do codigo (a busca de splits).
    """
    valores = np.asarray(valores, dtype=float)
    ordem = valores.argsort()
    ranks = np.empty(len(valores), dtype=float)
    ranks[ordem] = np.arange(1, len(valores) + 1, dtype=float)

    # Corrige empates: todos os empatados recebem a media de suas posicoes
    unicos, inversos, contagens = np.unique(
        valores, return_inverse=True, return_counts=True
    )
    for i, contagem in enumerate(contagens):
        if contagem > 1:
            mascara = inversos == i
            ranks[mascara] = ranks[mascara].mean()

    return ranks


if __name__ == "__main__":
    # Teste rapido de sanidade -- rode `python harris.py` para conferir
    # que os componentes funcionam isoladamente.
    rng = np.random.default_rng(0)
    X_teste = rng.random((20, 4))
    Y_teste = rng.random((20, 5))
    Y_norm = normalize_performance_matrix(Y_teste)

    print("Testando os rotulos de um no:")
    print("  rotulo de regressao:", HybridTree._regression_label(Y_norm).round(3))
    print("  rotulo de ranking (Borda):", HybridTree._ranking_label(Y_norm))

    print("\nTestando a floresta com diferentes lambdas:")
    for lam in [0.0, 0.5, 1.0]:
        floresta = HarrisForest(n_trees=5, lambda_param=lam, max_depth=3)
        floresta.fit(X_teste, Y_norm)
        pred = floresta.predict(X_teste[:2])
        print(f"  lambda={lam}: predicao da 1a instancia = {pred[0].round(3)}")
