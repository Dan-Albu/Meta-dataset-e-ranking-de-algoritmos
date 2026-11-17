"""
ETAPA 1 da Atividade Pratica 3 -- o espaco de configuracoes Theta.

ALGORITMO BASE ESCOLHIDO: XGBoost
---------------------------------
O enunciado pede um algoritmo "que tenha se saido bem na Atividade 1".
RF (15 vitorias em 30) e XGBoost (12 em 30) qualificam; ambos nunca
ficaram em ultimo. O criterio de desempate foi medido, nao suposto:
quanto os hiperparametros de cada um afetam a SERA.

Amostrando 8 configuracoes de cada algoritmo em 5 datasets:

    algoritmo   razao pior/melhor   CV entre configuracoes
    RF                 2,01                 0,215
    XGBoost            3,39                 0,383

XGBoost e quase duas vezes mais sensivel. Isso importa diretamente para
esta atividade: com uma matriz de perdas plana, qualquer metodo de busca
encontra rapidamente uma configuracao proxima da otima e todos empatam.
Seria o mesmo problema de teto diagnosticado na Atividade 2, onde a
margem disponivel para o meta-conhecimento era de apenas 0,107 em
correlacao de Spearman porque dois algoritmos venciam em 27 dos 30
datasets. Escolher o algoritmo mais sensivel e o que torna a comparacao
entre warm-start e busca fria capaz de revelar diferenca.

O ESPACO Theta (5 hiperparametros)
----------------------------------
    n_estimators      20 a 500     log      numero de arvores
    learning_rate     0.005 a 0.5  log      o mais influente em boosting
    max_depth         2 a 12       inteiro  complexidade de cada arvore
    subsample         0.5 a 1.0    linear   regularizacao por linhas
    colsample_bytree  0.4 a 1.0    linear   regularizacao por colunas

Escala logaritmica em n_estimators e learning_rate porque o efeito
desses dois e multiplicativo, nao aditivo: a diferenca entre 0,005 e
0,01 importa tanto quanto a diferenca entre 0,25 e 0,5. Amostrar
linearmente concentraria quase todas as configuracoes na faixa alta.
Os dois tambem interagem -- um learning_rate baixo exige mais arvores --,
e cobrir bem o plano formado por eles e o que faz o espaco ser
informativo.

COMO AS K CONFIGURACOES FORAM OBTIDAS
-------------------------------------
Sequencia de Sobol (quase-aleatoria, de baixa discrepancia), mais a
configuracao padrao da biblioteca, exigida pelo enunciado.

A justificativa vem do proprio livro-base. O Capitulo 6, Secao 6.2.2,
resume os argumentos de Bergstra e Bengio (2012) contra a grade: ela nao
escala com o numero de hiperparametros, pois cada novo hiperparametro
multiplica o numero de pontos; a discretizacao pode remover o otimo do
espaco de busca; e quando algum hiperparametro e irrelevante, a grade
desperdicia avaliacoes repetindo os mesmos valores dos demais, enquanto
a busca aleatoria explora mais valores distintos de cada um.

Sobol melhora a aleatoria pura pelo mesmo raciocinio levado adiante: a
sequencia e estratificada em cada dimensao, de modo que K pontos cobrem
cada faixa de forma mais uniforme, sem os aglomerados e vazios que a
amostragem independente produz. Com K pequeno -- e 60 e pequeno para
cinco dimensoes --, essa diferenca de cobertura e justamente onde ela
mais pesa.

DIMENSIONAMENTO
---------------
    K     = 60 configuracoes
    T     = 20 avaliacoes por busca
    ninit =  5 pontos iniciais

O enunciado exige K >= 3T; com K = 60 e T = 20 a restricao e atendida na
igualdade. Uma busca aleatoria percorre, portanto, um terco da tabela no
seu orcamento -- larga o bastante para que os metodos nao convirjam por
esgotamento do espaco.

O custo medido foi de aproximadamente 144 segundos por configuracao
sobre os 30 datasets com 5 folds, no pior caso (n_estimators alto com
max_depth alto); a configuracao media custa bem menos. Para K = 60 isso
situa a construcao da matriz na casa de uma a duas horas.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ---------------------------------------------------------------------
# Definicao de Theta
# ---------------------------------------------------------------------

ESPACO = {
    "n_estimators":     {"min": 20,    "max": 500, "escala": "log",    "tipo": int},
    "learning_rate":    {"min": 0.005, "max": 0.5, "escala": "log",    "tipo": float},
    "max_depth":        {"min": 2,     "max": 12,  "escala": "linear", "tipo": int},
    "subsample":        {"min": 0.5,   "max": 1.0, "escala": "linear", "tipo": float},
    "colsample_bytree": {"min": 0.4,   "max": 1.0, "escala": "linear", "tipo": float},
}

# Configuracao padrao do XGBRegressor, exigida pelo enunciado.
# E a referencia do metodo 1 (sem busca) e serve de piso: qualquer
# metodo de busca que nao a supere nao esta agregando nada.
PADRAO = {
    "n_estimators": 100,
    "learning_rate": 0.3,
    "max_depth": 6,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}

K = 60
T = 20
N_INIT = 5


def _destransformar(unitario, spec):
    """Leva um valor em [0,1] para a faixa do hiperparametro."""
    if spec["escala"] == "log":
        valor = np.exp(
            np.log(spec["min"])
            + unitario * (np.log(spec["max"]) - np.log(spec["min"]))
        )
    else:
        valor = spec["min"] + unitario * (spec["max"] - spec["min"])

    if spec["tipo"] is int:
        return int(round(valor))
    return float(valor)


def gerar_configuracoes(k=K, random_state=42, incluir_padrao=True):
    """
    Gera as K configuracoes: a padrao mais (k-1) pontos de Sobol.

    A sequencia de Sobol tem garantias de baixa discrepancia para
    potencias de 2, de modo que geramos a potencia de 2 imediatamente
    acima e descartamos o excedente -- gerar exatamente k-1 pontos
    degradaria a propriedade que justifica usar Sobol.

    Returns
    -------
    pd.DataFrame com uma linha por configuracao, indexada por
    "cfg_00", "cfg_01", ... A configuracao padrao e sempre "cfg_00".
    """
    nomes = list(ESPACO.keys())
    n_sobol = k - 1 if incluir_padrao else k

    potencia = int(np.ceil(np.log2(max(n_sobol, 1))))
    amostrador = qmc.Sobol(d=len(nomes), scramble=True, seed=random_state)
    pontos = amostrador.random_base2(m=potencia)[:n_sobol]

    linhas = []
    if incluir_padrao:
        linhas.append(dict(PADRAO))

    for ponto in pontos:
        linhas.append({
            nome: _destransformar(ponto[i], ESPACO[nome])
            for i, nome in enumerate(nomes)
        })

    configuracoes = pd.DataFrame(linhas)
    configuracoes.index = [f"cfg_{i:02d}" for i in range(len(configuracoes))]
    return configuracoes


def como_kwargs(linha, random_state=42, n_jobs=-1):
    """Converte uma linha do DataFrame em kwargs do XGBRegressor."""
    kwargs = {}
    for nome, spec in ESPACO.items():
        valor = linha[nome]
        kwargs[nome] = int(valor) if spec["tipo"] is int else float(valor)
    kwargs["random_state"] = random_state
    kwargs["verbosity"] = 0
    kwargs["n_jobs"] = n_jobs
    return kwargs


if __name__ == "__main__":
    configuracoes = gerar_configuracoes()
    print(f"K = {len(configuracoes)} configuracoes "
          f"({len(ESPACO)} hiperparametros)\n")
    print("Padrao da biblioteca (cfg_00) e as 5 primeiras de Sobol:")
    print(configuracoes.head(6).to_string())

    print("\nCobertura do espaco:")
    for nome, spec in ESPACO.items():
        coluna = configuracoes[nome]
        print(f"  {nome:18} [{spec['min']:>7}, {spec['max']:>5}] "
              f"-> amostrado em [{coluna.min():.4g}, {coluna.max():.4g}], "
              f"{coluna.nunique()} valores distintos")

    print(f"\nK/T = {K}/{T} = {K / T:.1f} "
          f"(exigencia do enunciado: >= 3)")
