import os
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from metrics import relevance, sera
from harris import HarrisForest, normalize_performance_matrix


_SRC_DIR = os.path.dirname(os.path.abspath(__file__))       # .../metalearning-ranking/src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                     # .../metalearning-ranking
DATA_FOLDER = os.path.join(_PROJECT_ROOT, "data")
RESULTS_FOLDER = os.path.join(_PROJECT_ROOT, "results")


class MetaDatasetBuilder:

    def __init__(self, data_folder, target_column, n_folds=10, random_state=42,
                 excluded_files=None):
        self.data_folder = data_folder
        self.target_column = target_column
        self.n_folds = n_folds
        self.random_state = random_state

        # Arquivos excluidos do meta-dataset, com justificativa (nao
        # apagamos do disco -- so pulamos no carregamento, mantendo
        # rastreabilidade da decisao para o relatorio). Use este
        # parametro se decidir excluir algo mais adiante.
        #
        # NOTA (decisao pendente, NAO aplicada ainda):
        # 'airfoild.csv' tem inconsistencia de escala no alvo, detectada
        # em EDA -- 25% das linhas com valores ~1000x maiores que o
        # minimo (min=103.38, Q1=118117.5), indicando erro na preparacao
        # dos dados de origem, anterior a este pipeline. Mantido no
        # meta-dataset por ora; decisao de excluir (ou nao) fica para o
        # fechamento do relatorio.
        self.excluded_files = excluded_files or []

        # Portfolio de algoritmos (>= 5 exigido pela atividade).
        self.algorithms = {
            "RF": RandomForestRegressor(random_state=random_state),
            "SVM": SVR(),
            "XGB": XGBRegressor(random_state=random_state, verbosity=0),
            "KNN": KNeighborsRegressor(),
            "DT": DecisionTreeRegressor(random_state=random_state),
        }

        self.per_fold_results = {}

        self.performance_matrix = None  
        self.harris_results = None      

    # ------------------------------------------------------------------
    # Metrica (metrics.py)


    def _score(self, y_true, y_pred, y_reference):
    
        phi_values = relevance(y_true, y_reference=y_reference)
        return sera(y_true, y_pred, phi_values)


    def load_datasets(self):
  
        datasets = []
        for filename in os.listdir(self.data_folder):
            if not filename.endswith(".csv"):
                continue
            if filename in self.excluded_files:
                continue

            path = os.path.join(self.data_folder, filename)
            df = pd.read_csv(path)

            y = df.iloc[:, 0]      # primeira coluna = alvo
            X = df.iloc[:, 1:]     # todas as demais colunas = features

            dataset_name = filename.replace(".csv", "")
            datasets.append((dataset_name, X, y))

        return datasets

    def evaluate_algorithm(self, algo, X, y):

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Normaliza as features -- ajustado SO com o treino, aplicado no teste. Sem isso, SVM e KNN (sensiveis a escala) ficam injustamente prejudicados frente a RF/XGB/DT 
            # (que nao ligam para escala, por serem baseados em arvore).
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Normaliza tambem o ALVO (y). Sem isso, SVR fica prejudicado em datasets com y de escala grande (ex.: precos, contagens de sistema): os hiperparametros default (epsilon, C) do SVR
            # pressupoem y numa faixa "normal". TransformedTargetRegressor treina no y normalizado e ja reverte a predicao de volta pra escala original automaticamente.
            model = TransformedTargetRegressor(regressor=algo, transformer=StandardScaler())
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            score = self._score(y_test, y_pred, y_reference=y)
            fold_scores.append(score)

        return fold_scores

    def build_performance_matrix(self, max_datasets=None):

        resultados = {}
        datasets = self.load_datasets()
        if max_datasets is not None:
            datasets = datasets[:max_datasets]

        for dataset_name, X, y in datasets:
            print(f"Processando {dataset_name} ({X.shape[0]} linhas)...")
            resultados[dataset_name] = {}
            self.per_fold_results[dataset_name] = {}
            for algo_name, algo in self.algorithms.items():
                print(f"  -> {algo_name}...")
                folds = self.evaluate_algorithm(clone(algo), X, y)
                self.per_fold_results[dataset_name][algo_name] = folds
                resultados[dataset_name][algo_name] = np.mean(folds)

        self.performance_matrix = pd.DataFrame(resultados).T
        return self.performance_matrix

    def save_per_fold_results(self, path):

        with open(path, "w") as f:
            json.dump(self.per_fold_results, f, indent=2)
        print(f"per_fold_results salvo em {path}")

    def load_per_fold_results(self, path):

        with open(path) as f:
            self.per_fold_results = json.load(f)
        print(f"per_fold_results carregado de {path}")

    def build_meta_features(self):

        linhas = {}
        for dataset_name, X, y in self.load_datasets():
            print(f"Extraindo meta-features de {dataset_name}...")
            features = {}
            features.update(self._general_features(X, y))
            features.update(self._statistical_features(X))
            features.update(self._domain_features(y))
            linhas[dataset_name] = features

        self.meta_features = pd.DataFrame(linhas).T
        return self.meta_features

    def _general_features(self, X, y):

        n_inst, n_attr = X.shape
        return {
            "nr_inst": n_inst,
            "nr_attr": n_attr,
            "inst_to_attr": n_inst / n_attr,
        }

    def _statistical_features(self, X):

        means = X.mean(numeric_only=True)
        stds = X.std(numeric_only=True)
        skews = X.skew(numeric_only=True)
        kurts = X.kurtosis(numeric_only=True)

        corr = X.corr(numeric_only=True).abs()
        # Mantem so o triangulo superior, sem a diagonal (correlacao de
        # cada atributo com ele mesmo = 1, nao deve entrar na media)
        upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pares_corr = corr.where(upper_mask).stack()

        return {
            "mean_of_means": means.mean(),
            "mean_of_stds": stds.mean(),
            "mean_skewness": skews.mean(),
            "mean_kurtosis": kurts.mean(),
            "mean_abs_correlation": pares_corr.mean() if len(pares_corr) else 0.0,
            "prop_high_correlation": (pares_corr > 0.7).mean() if len(pares_corr) else 0.0,
        }

    def _domain_features(self, y):

        phi_values = relevance(y, y_reference=y)
        mean_y = y.mean()
        return {
            "target_skewness": y.skew(),
            "target_cv": y.std() / mean_y if mean_y != 0 else np.nan,
            "mean_relevance": phi_values.mean(),
            "prop_rare_instances": (phi_values > 0.5).mean(),
        }

    # ------------------------------------------------------------------
    # ETAPA 3 (Secao 1.3): matriz de ranks (R)
    # ------------------------------------------------------------------

    def build_rank_matrix(self):

        if self.performance_matrix is None:
            raise RuntimeError(
                "Chame build_performance_matrix() antes de build_rank_matrix()."
            )

        self.rank_matrix = self.performance_matrix.rank(
            axis=1, ascending=True, method="average"
        )

        # Verificacao de sanidade
        # matematica do rank medio, vale com ou sem empates. Se alguma linha nao bater, e sinal de problema na matriz P (ex.: NaN).
        n_algos = len(self.algorithms)
        soma_esperada = n_algos * (n_algos + 1) / 2
        somas_reais = self.rank_matrix.sum(axis=1)
        fora_do_esperado = somas_reais[~np.isclose(somas_reais, soma_esperada)]
        if len(fora_do_esperado) > 0:
            print(f"AVISO: soma de rank inesperada (esperado {soma_esperada}) em:")
            print(fora_do_esperado)
        else:
            print(f"OK: soma de ranks bate em todas as linhas (esperado {soma_esperada}).")

        return self.rank_matrix


    def aggregate_average_rank(self):

        if self.rank_matrix is None:
            raise RuntimeError("Chame/carregue build_rank_matrix() antes.")

        scores = self.rank_matrix.mean(axis=0)
        ranking_final = scores.rank(method="average")
        return scores, ranking_final

    def aggregate_median_rank(self):

        if self.rank_matrix is None:
            raise RuntimeError("Chame/carregue build_rank_matrix() antes.")

        scores = self.rank_matrix.median(axis=0)
        ranking_final = scores.rank(method="average")
        return scores, ranking_final

    def aggregate_significant_wins(self, alpha=0.05):

        if not self.per_fold_results:
            raise RuntimeError(
                "self.per_fold_results esta vazio. Rode "
                "build_performance_matrix() (nao so build_rank_matrix() "
                "a partir de um CSV) antes de usar este metodo."
            )

        algo_names = list(self.algorithms.keys())
        vitorias = {a: 0 for a in algo_names}

        for dataset_name, folds_por_algo in self.per_fold_results.items():
            for a, b in itertools.combinations(algo_names, 2):
                folds_a = np.asarray(folds_por_algo[a])
                folds_b = np.asarray(folds_por_algo[b])

                diffs = folds_a - folds_b
                if np.allclose(diffs, 0):
                    continue  # empate perfeito -- Wilcoxon nao aceita
                              # diferencas todas zero, e nao ha vitoria

                _, p_valor = wilcoxon(folds_a, folds_b)
                if p_valor < alpha:
                    # SERA e erro: MENOR media = melhor
                    if folds_a.mean() < folds_b.mean():
                        vitorias[a] += 1
                    else:
                        vitorias[b] += 1

        vitorias_series = pd.Series(vitorias)
        # rank 1 vai para quem tem MAIS, entao ascending=False
        ranking_final = vitorias_series.rank(ascending=False, method="average")
        return vitorias_series, ranking_final

    def _predict_lodo(self, meta_target, meta_model=None):
 
        if self.meta_features is None:
            raise RuntimeError("Chame build_meta_features() antes.")

        if meta_model is None:
            meta_model = RandomForestRegressor(random_state=self.random_state)

        X_meta = self.meta_features.loc[meta_target.index]

        # normalizacao da Etapa 1, ajustada so no treino de cada rodada.
        predicoes = []
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(X_meta):
            X_train = X_meta.iloc[train_idx]
            X_test = X_meta.iloc[test_idx]
            y_train = meta_target.iloc[train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            modelo = clone(meta_model)
            modelo.fit(X_train_scaled, y_train)
            predicoes.append(modelo.predict(X_test_scaled)[0])

        return pd.DataFrame(
            predicoes, index=meta_target.index, columns=meta_target.columns
        )

    def approach_1_predict_performance(self, meta_model=None):

        if self.performance_matrix is None:
            raise RuntimeError("Chame/carregue build_performance_matrix() antes.")

        print("Abordagem 1: LODO sobre a matriz P...")
        P_predita = self._predict_lodo(self.performance_matrix, meta_model)
        R_predita = P_predita.rank(axis=1, ascending=True, method="average")
        return P_predita, R_predita

    def approach_2_predict_ranking(self, meta_model=None):

        if self.rank_matrix is None:
            raise RuntimeError("Chame/carregue build_rank_matrix() antes.")

        print("Abordagem 2: LODO sobre a matriz R...")
        R_bruta = self._predict_lodo(self.rank_matrix, meta_model)
        # menor valor predito = melhor posicao = rank 1
        R_predita = R_bruta.rank(axis=1, ascending=True, method="average")
        return R_bruta, R_predita

    def approach_3_harris(self, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
                          n_trees=50, max_depth=4):

        if self.performance_matrix is None:
            raise RuntimeError("Chame/carregue build_performance_matrix() antes.")
        if self.meta_features is None:
            raise RuntimeError("Chame build_meta_features() antes.")

        X_meta = self.meta_features.loc[self.performance_matrix.index]
        P_norm = normalize_performance_matrix(self.performance_matrix.values)

        resultados = {}
        for lam in lambdas:
            print(f"HARRIS: LODO com lambda={lam}...")
            predicoes = []
            for train_idx, test_idx in LeaveOneOut().split(X_meta):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_meta.iloc[train_idx])
                X_test = scaler.transform(X_meta.iloc[test_idx])

                floresta = HarrisForest(
                    n_trees=n_trees,
                    lambda_param=lam,
                    max_depth=max_depth,
                    random_state=self.random_state,
                )
                floresta.fit(X_train, P_norm[train_idx])
                predicoes.append(floresta.predict(X_test)[0])

            P_predita = pd.DataFrame(
                predicoes,
                index=self.performance_matrix.index,
                columns=self.performance_matrix.columns,
            )
            resultados[lam] = P_predita.rank(
                axis=1, ascending=True, method="average"
            )

        self.harris_results = resultados
        return resultados


if __name__ == "__main__":
    builder = MetaDatasetBuilder(
        data_folder=DATA_FOLDER,
        target_column="target",
        excluded_files=[],  
    )

    # Testando load_datasets
    datasets = builder.load_datasets()
    print(f"Total de datasets carregados: {len(datasets)}")
    nome, X, y = datasets[0]
    print(f"Primeiro dataset: {nome}")
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")

    # Testando evaluate_algorithm() com UM algoritmo em UM dataset
    algo_teste = clone(builder.algorithms["RF"])
    fold_scores = builder.evaluate_algorithm(algo_teste, X, y)
    print(f"\nSERA por fold (RF em '{nome}'): {[round(s, 4) for s in fold_scores]}")
    print(f"Media dos {len(fold_scores)} folds: {np.mean(fold_scores):.4f}")

    # --- RODADA COMPLETA: os 30 datasets x 5 algoritmos ---

    print("\n--- Rodada completa: build_performance_matrix() ---")
    P = builder.build_performance_matrix()
    P.to_csv(os.path.join(RESULTS_FOLDER, "P_matrix.csv"))
    builder.save_per_fold_results(os.path.join(RESULTS_FOLDER, "per_fold_results.json"))
    print(P)

    # matriz de meta-caracteristicas (X) ---
    print("\n--- Construindo matriz X (meta-features) ---")
    X_meta = builder.build_meta_features()
    X_meta.to_csv(os.path.join(RESULTS_FOLDER, "X_matrix.csv"))
    print(X_meta)

    # --- ETAPA 3: matriz de ranks (R) ---
    print("\n--- Construindo matriz R (ranks) ---")
    R = builder.build_rank_matrix()
    R.to_csv(os.path.join(RESULTS_FOLDER, "R_matrix.csv"))
    print(R)

    # Previa de teste
    print("\nMedia de rank por algoritmo (previa informal, nao e o AR formal):")
    print(R.mean(axis=0).sort_values())

    # abordagens de agregacao
    print("\n--- AR (rank medio) ---")
    scores_ar, rank_ar = builder.aggregate_average_rank()
    print("Scores brutos (media dos ranks):")
    print(scores_ar.sort_values())
    print("Ranking final (AR):")
    print(rank_ar.sort_values())

    print("\n--- MR (rank mediano) ---")
    scores_mr, rank_mr = builder.aggregate_median_rank()
    print("Scores brutos (mediana dos ranks):")
    print(scores_mr.sort_values())
    print("Ranking final (MR):")
    print(rank_mr.sort_values())

    print("\n--- SW (vitorias significativas, alpha=0.05) ---")
    vitorias_sw, rank_sw = builder.aggregate_significant_wins(alpha=0.05)
    print("Total de vitorias significativas por algoritmo:")
    print(vitorias_sw.sort_values(ascending=False))
    print("Ranking final (SW):")
    print(rank_sw.sort_values())

    agregados = pd.DataFrame({
        "AR_score": scores_ar,
        "AR_rank": rank_ar,
        "MR_score": scores_mr,
        "MR_rank": rank_mr,
        "SW_vitorias": vitorias_sw,
        "SW_rank": rank_sw,
    })
    agregados.to_csv(os.path.join(RESULTS_FOLDER, "rankings_agregados.csv"))
    print("\nRankings agregados salvos em results/rankings_agregados.csv")
    print(agregados)

    # meta-modelos treinados (Abordagens 1 e 2)
    print("\n--- Abordagem 1: regressor sobre P, converte depois ---")
    P_pred_a1, R_pred_a1 = builder.approach_1_predict_performance()
    R_pred_a1.to_csv(os.path.join(RESULTS_FOLDER, "R_predita_abordagem1.csv"))
    print(R_pred_a1.head(10))

    print("\n--- Abordagem 2: regressor sobre R, prediz posicoes direto ---")
    R_bruta_a2, R_pred_a2 = builder.approach_2_predict_ranking()
    R_pred_a2.to_csv(os.path.join(RESULTS_FOLDER, "R_predita_abordagem2.csv"))
    print(R_pred_a2.head(10))

    # Previa de qualidade: 
    # Spearman medio entre ranking predito e real.
    from scipy.stats import spearmanr
    print("\nPrevia -- Spearman medio (predito vs. real), por abordagem:")
    for nome_ab, R_pred in [("Abordagem 1", R_pred_a1), ("Abordagem 2", R_pred_a2)]:
        correlacoes = [
            spearmanr(R.loc[ds], R_pred.loc[ds]).statistic
            for ds in R.index
        ]
        print(f"  {nome_ab}: {np.mean(correlacoes):.4f}")

    # Baseline de comparacao: 
    # as agregacoes devolvem SEMPRE o mesmo ranking, entao o Spearman delas e medido contra cada dataset real.
    for nome_ab, ranking_fixo in [("AR", rank_ar), ("MR", rank_mr), ("SW", rank_sw)]:
        correlacoes = [
            spearmanr(R.loc[ds], ranking_fixo[R.columns]).statistic
            for ds in R.index
        ]
        print(f"  {nome_ab}: {np.mean(correlacoes):.4f}")

    # HARRIS
    print("\n--- HARRIS: floresta hibrida de ranking e regressao ---")
    harris_rankings = builder.approach_3_harris()

    for lam, R_pred in harris_rankings.items():
        R_pred.to_csv(
            os.path.join(RESULTS_FOLDER, f"R_predita_harris_lambda{lam}.csv")
        )

    print("\nEfeito do lambda no HARRIS (Spearman medio):")
    for lam, R_pred in harris_rankings.items():
        correlacoes = [
            spearmanr(R.loc[ds], R_pred.loc[ds]).statistic for ds in R.index
        ]
        print(f"  lambda={lam}: {np.mean(correlacoes):.4f}")

    # --------------------------------------------------------------------------
    #avaliacao experimental das seis abordagens

    from evaluation import (
        spearman_por_abordagem, curva_de_perda, area_sob_curva,
        diagrama_diferenca_critica, plotar_curva_de_perda,
    )

    melhor_lambda = max(
        harris_rankings,
        key=lambda lam: np.mean([
            spearmanr(R.loc[ds], harris_rankings[lam].loc[ds]).statistic
            for ds in R.index
        ]),
    )
    print(f"\nMelhor lambda do HARRIS: {melhor_lambda}")

    abordagens = {
        "AR": rank_ar,
        "MR": rank_mr,
        "SW": rank_sw,
        "A1": R_pred_a1,
        "A2": R_pred_a2,
        f"HARRIS(L={melhor_lambda})": harris_rankings[melhor_lambda],
    }

    #Metrica 1: correlacao de Spearman
    print("\n=== METRICA 1: correlacao de Spearman ===")
    spearman_df = spearman_por_abordagem(R, abordagens)
    spearman_df.to_csv(os.path.join(RESULTS_FOLDER, "spearman_por_dataset.csv"))
    print(spearman_df.mean().sort_values(ascending=False).round(4))

    #Metrica 2: curva de perda 
    print("\n=== METRICA 2: curva de perda ===")
    P_norm = pd.DataFrame(
        normalize_performance_matrix(P.values), index=P.index, columns=P.columns
    )
    curva = curva_de_perda(P_norm, abordagens, menor_e_melhor=True)
    curva.to_csv(os.path.join(RESULTS_FOLDER, "curva_de_perda.csv"))
    print(curva.round(4))
    print("\nArea sob a curva (menor = melhor):")
    print(area_sob_curva(curva).sort_values().round(4))
    plotar_curva_de_perda(curva, os.path.join(RESULTS_FOLDER, "curva_de_perda.png"))

    # --- Metrica 3: diagrama de diferenca critica ---
    print("\n=== METRICA 3: diagrama de diferenca critica ===")
    resultado_cd = diagrama_diferenca_critica(
        spearman_df, os.path.join(RESULTS_FOLDER, "diagrama_diferenca_critica.png")
    )
    print("\nRanks medios das abordagens (1 = melhor):")
    print(resultado_cd["ranks_medios"].sort_values().round(3))
