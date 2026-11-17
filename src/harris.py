import numpy as np
from scipy.stats import spearmanr  # usado so em testes/inspecao; a perda
                                    # de ranking usa Pearson vetorizado


def normalize_performance_matrix(P):

    P = np.asarray(P, dtype=float)
    minimos = P.min(axis=1, keepdims=True)
    maximos = P.max(axis=1, keepdims=True)
    amplitude = maximos - minimos
    # Linha constante (todos algoritmos identicos): evita divisao por zero
    amplitude[amplitude == 0] = 1.0
    return (P - minimos) / amplitude

class HybridTree:

    def __init__(self, lambda_param=0.5, max_depth=4, min_samples_split=2,
                 max_features=None, random_state=None):
        self.lambda_param = lambda_param
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.tree_ = None

    #------------------------------------------------------------------
    # Rotulos de regressão de um nó

    @staticmethod
    def _regression_label(Y):

        return Y.mean(axis=0)

    @staticmethod
    def _ranking_label_from_ranks(Y_ranks):

        return _rankdata_ascending(Y_ranks.mean(axis=0))

    @staticmethod
    def _ranking_label(Y):

        ranks = np.apply_along_axis(_rankdata_ascending, 1, Y)
        return _rankdata_ascending(ranks.mean(axis=0))

    # ------------------------------------------------------------------
    # As duas perdas 

    @staticmethod
    def _regression_loss(Y, rotulo_regressao):

        return float(np.mean((Y - rotulo_regressao) ** 2))

    @staticmethod
    def _ranking_loss_from_ranks(Y_ranks, rotulo_ranking):

        n = len(Y_ranks)
        if n < 2:
            # Um unico elemento: o consenso e ele mesmo, perda zero.
            return 0.0

        c = np.asarray(rotulo_ranking, dtype=float)
        c_centrado = c - c.mean()
        denom_c = np.sqrt((c_centrado ** 2).sum())
        if denom_c == 0:
            # empatados no Borda
            # indefinido (tratamos como correlacao nula).
            return 0.5  # = 1.0 / 2 (normalizacao)

        R_centrado = Y_ranks - Y_ranks.mean(axis=1, keepdims=True)
        denom_R = np.sqrt((R_centrado ** 2).sum(axis=1))

        numerador = (R_centrado * c_centrado).sum(axis=1)
        denominador = denom_R * denom_c

        # Instancias com ranks constantes
        # correlacao indefinida (tratada como nula).
        correlacoes = np.zeros(n, dtype=float)
        validos = denominador > 0
        correlacoes[validos] = numerador[validos] / denominador[validos]

        return float(np.mean(1.0 - correlacoes) / 2.0)

    def _node_loss(self, Y, Y_ranks):

        if len(Y) == 0:
            return 0.0

        perda = 0.0
        # So calcula cada perda se ela tiver peso
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
   
    def _find_best_split(self, X, Y, Y_ranks, rng):
 
        n_amostras, n_atributos = X.shape

        # Sorteio de atributos
        if self.max_features is not None and self.max_features < n_atributos:
            atributos = rng.choice(n_atributos, self.max_features, replace=False)
        else:
            atributos = np.arange(n_atributos)

        melhor_atributo, melhor_limiar, melhor_perda = None, None, np.inf

        for atributo in atributos:
            valores = np.unique(X[:, atributo])
            if len(valores) < 2:
                continue  

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

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        Y_ranks = np.apply_along_axis(_rankdata_ascending, 1, Y)

        rng = np.random.default_rng(self.random_state)
        self.tree_ = self._build_tree(X, Y, Y_ranks, 0, rng)
        return self

    def predict(self, X):
       
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

        predicoes = np.array([arvore.predict(X) for arvore in self.trees_])
        return predicoes.mean(axis=0)


def _rankdata_ascending(valores):

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
    # para conferir que os componentes funcionam isoladamente.
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
