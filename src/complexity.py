"""
Medidas de complexidade para problemas de regressao.

ETAPA 2 da Atividade Pratica 2 -- construcao da matriz X estendida.

GRUPO A: as 12 medidas de Lorena, A. C., Maciel, A. I., de Miranda, P. B. C.,
Costa, I. G., Prudencio, R. B. C. (2018). "Data complexity meta-features for
regression problems". Machine Learning, 107(1):209-246.

    Correlacao com o alvo ...... C1, C2, C3, C4
    Linearidade ................ L1, L2, L3
    Suavidade .................. S1, S2, S3, S4
    Geometria/densidade ........ T2

GRUPO C: decomposicao por regiao de relevancia -- adaptacao propria, que
transporta para regressao a ideia de decomposicao por classe de Barella,
V. H. et al. (2021), "Assessing the data complexity of imbalanced datasets",
Information Sciences.

    A justificativa vem de Batista et al., citado por Barella et al.: o
    desbalanceamento nao e um problema por si so -- ele AMPLIFICA o efeito
    adverso de outras caracteristicas intrinsecas dos dados (sobreposicao,
    separabilidade). Medir apenas a proporcao de casos raros (p.rare) diz
    pouco; e preciso medir o que esta sendo amplificado, na regiao onde a
    amplificacao acontece.

    NOTA: essa tese foi estabelecida para classificacao. Transporta-la para
    regressao e uma hipotese deste trabalho, nao um resultado demonstrado.

DIRECAO DAS MEDIDAS (importante para interpretar as razoes do Grupo C):
    Maior = mais SIMPLES .... C1, C2
    Menor = mais SIMPLES .... C3, C4, L1, L2, L3, S1, S2, S3, S4
    T2 ...................... maior = menos esparso

Implementacao propria em Python. O codigo de referencia dos autores esta em
R (github.com/aclorena/ComplexityRegression); o pymfe implementa medidas de
complexidade apenas para classificacao.
"""

import warnings
import numpy as np
from scipy.stats import spearmanr, ConstantInputWarning

# Atributos constantes aparecem em varios datasets (colunas one-hot com
# uma unica categoria presente, por exemplo). Spearman e indefinido para
# eles e ja tratamos isso devolvendo 0; o aviso do scipy so polui a saida.
warnings.filterwarnings("ignore", category=ConstantInputWarning)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors


def _rankdata(valores):
    """Ranking com empates recebendo rank medio, em ordem crescente."""
    valores = np.asarray(valores, dtype=float)
    ordem = valores.argsort()
    ranks = np.empty(len(valores), dtype=float)
    ranks[ordem] = np.arange(1, len(valores) + 1, dtype=float)
    unicos, inversos, contagens = np.unique(
        valores, return_inverse=True, return_counts=True
    )
    for i, contagem in enumerate(contagens):
        if contagem > 1:
            mascara = inversos == i
            ranks[mascara] = ranks[mascara].mean()
    return ranks


# ----------------------------------------------------------------------
# Familia 1: correlacao dos atributos com o alvo
# ----------------------------------------------------------------------

def _abs_spearman(X, y):
    """Correlacao de Spearman absoluta de cada atributo com o alvo."""
    correlacoes = []
    for j in range(X.shape[1]):
        coluna = X[:, j]
        if np.all(coluna == coluna[0]):
            correlacoes.append(0.0)  # atributo constante
            continue
        rho = spearmanr(coluna, y).statistic
        correlacoes.append(0.0 if np.isnan(rho) else abs(rho))
    return np.array(correlacoes)


def C1(X, y):
    """
    Maxima correlacao de um atributo com o alvo (Eq. 2 do artigo).

    Existe ao menos um atributo fortemente ligado ao alvo? Em caso
    afirmativo, funcoes mais simples conseguem ajustar os dados.

    Spearman foi escolhida pelos autores por ser nao-parametrica: nao
    assume distribuicao particular entre o alvo e os atributos.

    Maior = mais simples. Custo O(d * n * log n).
    """
    return float(_abs_spearman(X, y).max())


def C2(X, y):
    """
    Correlacao media dos atributos com o alvo (Eq. 3).

    Diferente de C1, e sensivel a QUANTIDADE de atributos com associacao
    fraca -- um dataset com um atributo otimo e vinte inuteis tem C1 alto
    e C2 baixo.

    Maior = mais simples. Custo O(d * n * log n).
    """
    return float(_abs_spearman(X, y).mean())


def C3(X, y, limiar=0.9, max_remocao=0.5):
    """
    Eficiencia individual dos atributos (Eq. 4).

    Para cada atributo, quantos exemplos precisam ser removidos ate que
    sua correlacao com o alvo ultrapasse 0,9? Devolve o minimo entre os
    atributos, em proporcao do total.

    Intuicao dos autores: mede o poder explicativo de cada atributo,
    verificando a quantos dos exemplos originais ele consegue relacionar
    ao alvo com alta confianca.

    Menor = mais simples (zero quando ja existe atributo com correlacao
    acima do limiar). Custo O(d * n^2).

    NOTA DE IMPLEMENTACAO: o artigo propoe um algoritmo eficiente que
    atualiza as diferencas de ranking em vez de recalcula-las. Aqui
    usa-se a heuristica descrita no texto -- remover iterativamente o
    exemplo de maior discrepancia entre os rankings --, recalculando a
    correlacao a cada passo.

    TETO DE REMOCAO (max_remocao): em datasets onde nenhum atributo
    chega perto do limiar, o laco removeria exemplos ate sobrarem tres,
    a um custo de O(n) chamadas a spearmanr por atributo. Em cpu_small
    (correlacao maxima de 0,74) isso levava mais de dez minutos. O teto
    interrompe a busca apos remover max_remocao do conjunto e devolve
    esse valor, sinalizando "mais da metade dos exemplos teria de ser
    descartada" -- informacao suficiente para a meta-caracteristica.
    Declarar essa aproximacao no relatorio.
    """
    n, d = X.shape
    proporcoes = []

    for j in range(d):
        coluna = X[:, j]
        if np.all(coluna == coluna[0]):
            proporcoes.append(1.0)
            continue

        idx = np.arange(n)
        removidos = 0
        teto = int(max_remocao * n)
        while len(idx) > 3 and removidos < teto:
            rho = spearmanr(coluna[idx], y[idx]).statistic
            if np.isnan(rho):
                break
            if abs(rho) > limiar:
                break
            # remove o exemplo de maior discrepancia entre os dois rankings
            rank_x = _rankdata(coluna[idx])
            rank_y = _rankdata(y[idx])
            if rho < 0:
                rank_y = len(idx) + 1 - rank_y
            pior = np.argmax(np.abs(rank_x - rank_y))
            idx = np.delete(idx, pior)
            removidos += 1

        proporcoes.append(removidos / n)

    return float(min(proporcoes))


def C4(X, y, limiar_residuo=0.1):
    """
    Eficiencia coletiva dos atributos (Eq. 5).

    Parte do atributo mais correlacionado, ajusta uma reta entre ele e o
    alvo, e remove os exemplos de residuo pequeno (|e| <= 0,1). Repete com
    o proximo atributo mais correlacionado sobre os exemplos restantes,
    ate esgotar os atributos ou os exemplos. Devolve a proporcao que
    sobrou.

    Diferente de C3, os exemplos removidos em uma rodada nao voltam nas
    seguintes -- por isso "coletiva": mede se os atributos se COMPLEMENTAM
    para explicar o alvo.

    Menor = mais simples. Custo O(d * (d + n * log n)).
    """
    n, d = X.shape
    idx = np.arange(n)
    atributos_restantes = list(range(d))

    while len(idx) > 0 and atributos_restantes:
        correlacoes = {
            j: abs(spearmanr(X[idx, j], y[idx]).statistic or 0.0)
            for j in atributos_restantes
        }
        correlacoes = {j: (0.0 if np.isnan(c) else c) for j, c in correlacoes.items()}
        melhor = max(correlacoes, key=correlacoes.get)
        atributos_restantes.remove(melhor)

        modelo = LinearRegression().fit(X[idx, melhor].reshape(-1, 1), y[idx])
        residuos = np.abs(y[idx] - modelo.predict(X[idx, melhor].reshape(-1, 1)))
        idx = idx[residuos > limiar_residuo]

    return float(len(idx) / n)


# ----------------------------------------------------------------------
# Familia 2: linearidade
# ----------------------------------------------------------------------

def _residuos_lineares(X, y):
    """Residuos de uma regressao linear multipla ajustada nos proprios dados."""
    modelo = LinearRegression().fit(X, y)
    return y - modelo.predict(X)


def L1(X, y):
    """
    Erro absoluto medio de uma regressao linear multipla (Eq. 6).

    Menor = mais simples (o problema e linearmente ajustavel).
    Custo O(n * d^2).
    """
    return float(np.mean(np.abs(_residuos_lineares(X, y))))


def L2(X, y):
    """
    Variancia dos residuos da mesma regressao linear (Eq. 7).

    Igual a L1, mas penalizando mais os desvios grandes.
    Menor = mais simples. Custo O(n * d^2).
    """
    return float(np.mean(_residuos_lineares(X, y) ** 2))


def L3(X, y, random_state=42):
    """
    Nao-linearidade de um regressor linear (Eq. 11).

    Ordena os exemplos pelo alvo, interpola aleatoriamente cada par de
    vizinhos (entradas E saida), treina um regressor linear nos dados
    ORIGINAIS e mede o MSE nos pontos interpolados.

    Se os pontos originais estao distribuidos suavemente, suas variantes
    interpoladas ficam proximas dos dados originais e o MSE e baixo.

    Menor = mais simples. Custo O(n * (d^2 + log n)).
    """
    rng = np.random.default_rng(random_state)
    ordem = np.argsort(y)
    X_ord, y_ord = X[ordem], y[ordem]

    pesos = rng.random(len(y) - 1).reshape(-1, 1)
    X_novo = X_ord[:-1] + pesos * (X_ord[1:] - X_ord[:-1])
    y_novo = y_ord[:-1] + pesos.ravel() * (y_ord[1:] - y_ord[:-1])

    modelo = LinearRegression().fit(X, y)
    return float(np.mean((modelo.predict(X_novo) - y_novo) ** 2))


# ----------------------------------------------------------------------
# Familia 3: suavidade
# ----------------------------------------------------------------------

def S1(X, y):
    """
    Distribuicao do alvo (Eq. 8).

    Constroi a arvore geradora minima (MST) no espaco de entrada e mede a
    diferenca media entre os alvos de vertices adjacentes. Inspirada na
    medida N1 de Ho e Basu (2002).

    Vizinhos no espaco de entrada tem alvos parecidos? Em caso afirmativo,
    a funcao a ajustar e suave.

    Menor = mais simples. Custo O(n^2 * d).

    NOTA DE MEMORIA: a matriz densa de distancias ocupa O(n^2) -- com
    n = 2000 sao 32 MB, mas o pico durante squareform() e maior e pode
    derrubar o processo. Aqui a MST e construida a partir da matriz
    esparsa triangular, evitando materializar a densa completa.
    """
    n = len(y)
    condensada = pdist(X)
    # Monta apenas o triangulo superior em formato esparso
    linhas, colunas = np.triu_indices(n, k=1)
    esparsa = coo_matrix((condensada, (linhas, colunas)), shape=(n, n))
    mst = minimum_spanning_tree(esparsa.tocsr()).tocoo()
    return float(np.sum(np.abs(y[mst.row] - y[mst.col])) / n)


def S2(X, y):
    """
    Distribuicao das entradas (Eq. 9).

    Ordena os exemplos pelo alvo e mede a distancia euclidiana media entre
    pares vizinhos nessa ordem. Complementa S1: pontos com alvos parecidos
    vem de entradas parecidas?

    Menor = mais simples. Custo O(n * (d + log n)).
    """
    X_ord = X[np.argsort(y)]
    return float(np.sum(np.linalg.norm(np.diff(X_ord, axis=0), axis=1)) / len(y))


def S3(X, y):
    """
    Erro do regressor 1-NN por leave-one-out (Eq. 10).

    Valores altos indicam muitas lacunas no espaco de entrada -- o vizinho
    mais proximo de um ponto tem alvo muito diferente do dele.

    Menor = mais simples. Custo O(d * n^2).
    """
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    _, indices = nn.kneighbors(X)
    vizinho = indices[:, 1]  # coluna 0 e o proprio ponto
    return float(np.mean((y[vizinho] - y) ** 2))


def S4(X, y, random_state=42):
    """
    Nao-linearidade do regressor 1-NN.

    Mesma interpolacao de L3, mas avaliando um regressor 1-NN em vez de um
    linear. Mede rugosidade local sem supor linearidade.

    Menor = mais simples. Custo O(n * d * log n).
    """
    rng = np.random.default_rng(random_state)
    ordem = np.argsort(y)
    X_ord, y_ord = X[ordem], y[ordem]

    pesos = rng.random(len(y) - 1).reshape(-1, 1)
    X_novo = X_ord[:-1] + pesos * (X_ord[1:] - X_ord[:-1])
    y_novo = y_ord[:-1] + pesos.ravel() * (y_ord[1:] - y_ord[:-1])

    nn = NearestNeighbors(n_neighbors=1).fit(X)
    _, indices = nn.kneighbors(X_novo)
    return float(np.mean((y[indices.ravel()] - y_novo) ** 2))


# ----------------------------------------------------------------------
# Familia 4: geometria, topologia e densidade
# ----------------------------------------------------------------------

def T2(X, y=None):
    """
    Media de exemplos por dimensao (Eq. 13): n / d.

    Captura a esparsidade dos dados. Custo O(n + d).
    """
    n, d = X.shape
    return float(n / d)


# ----------------------------------------------------------------------
# Registro das medidas
# ----------------------------------------------------------------------

MEDIDAS = {
    "C1": C1, "C2": C2, "C3": C3, "C4": C4,
    "L1": L1, "L2": L2, "L3": L3,
    "S1": S1, "S2": S2, "S3": S3, "S4": S4,
    "T2": T2,
}

# Medidas em que MAIOR significa mais simples -- usado para interpretar
# a direcao das razoes do Grupo C.
MAIOR_E_SIMPLES = {"C1", "C2"}

# Medidas caras (O(n^2)), candidatas a subamostragem em datasets grandes.
CARAS = {"C3", "S1", "S3"}


def normalizar_0_1(A):
    """
    Reescala cada coluna para [0, 1].

    Exigencia explicita dos autores (secao 3.5 do artigo): "os dados devem
    ser normalizados entre [0,1] antes da aplicacao destas medidas (exceto
    T2)... a normalizacao e necessaria para todas as medidas que computam
    alguma distancia entre exemplos, como S1 a S4". Sem isso, as medidas
    nao sao comparaveis entre datasets de escalas diferentes -- o mesmo
    problema que ja enfrentamos com a SERA na Atividade 1.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        minimo, maximo = A.min(), A.max()
        amplitude = maximo - minimo
        return (A - minimo) / amplitude if amplitude > 0 else np.zeros_like(A)
    minimos = A.min(axis=0)
    amplitudes = A.max(axis=0) - minimos
    amplitudes[amplitudes == 0] = 1.0
    return (A - minimos) / amplitudes


def extrair_complexidade(X, y, medidas=None, max_amostras=2000,
                         random_state=42, normalizar=True):
    """
    GRUPO A -- calcula as medidas de complexidade sobre o dataset inteiro.

    Parameters
    ----------
    X : np.ndarray (n, d) -- atributos, preferencialmente ja normalizados.
    y : np.ndarray (n,)   -- alvo.
    medidas : lista de siglas, ou None para todas.
    max_amostras : int
        Subamostra aleatoria antes de calcular as medidas O(n^2) (S1, S3,
        C3). Com 20.640 instancias em california, S1 exigiria uma matriz
        de distancias de ~3,4 bilhoes de entradas. Documente esse limite
        no relatorio -- e uma aproximacao, nao o valor exato.

    Returns
    -------
    dict {sigla: valor}
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    medidas = medidas or list(MEDIDAS.keys())

    # T2 e razao de contagens, nao depende de escala -- calculado antes
    # da normalizacao para deixar claro que nao e afetado por ela.
    if normalizar:
        X = normalizar_0_1(X)
        y = normalizar_0_1(y)

    rng = np.random.default_rng(random_state)
    if len(y) > max_amostras:
        sub = rng.choice(len(y), max_amostras, replace=False)
        X_sub, y_sub = X[sub], y[sub]
    else:
        X_sub, y_sub = X, y

    resultado = {}
    for sigla in medidas:
        dados = (X_sub, y_sub) if sigla in CARAS else (X, y)
        try:
            resultado[sigla] = MEDIDAS[sigla](*dados)
        except Exception as erro:
            print(f"    aviso: {sigla} falhou ({erro}); usando NaN")
            resultado[sigla] = np.nan

    return resultado


def extrair_complexidade_por_regiao(X, y, phi, limiar_phi=0.8,
                                    medidas_regiao=("C1", "C2", "L1", "L2", "S1", "S3"),
                                    max_amostras=2000, random_state=42,
                                    complexidade_global=None):
    """
    GRUPO C -- decomposicao por regiao de relevancia (proposta deste
    trabalho).

    Calcula as medidas restritas as instancias RARAS (phi > limiar) e as
    razoes entre regiao rara e dataset completo.

    Parameters
    ----------
    phi : np.ndarray (n,) -- relevancia de cada instancia (metrics.relevance).
    limiar_phi : float
        Limiar que define a regiao rara. 0,8 segue o Meta-IR; a matriz X
        base deste projeto usa 0,5. E um hiperparametro da propria
        meta-caracteristica, sem consenso na literatura.
    complexidade_global : dict, opcional
        Saida de extrair_complexidade() para o mesmo dataset. Se fornecida,
        evita recalcular as medidas globais ao montar as razoes.

    Returns
    -------
    dict com chaves "<SIGLA>_rara" e "razao_<SIGLA>".

    INTERPRETACAO DAS RAZOES (as direcoes sao OPOSTAS entre medidas):
      - C1, C2 sao correlacoes, maior = mais simples.
        razao < 1  ->  regiao rara mais dificil
      - L1, L2, S1, S3 sao erros/distancias, menor = mais simples.
        razao > 1  ->  regiao rara mais dificil
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    phi = np.asarray(phi, dtype=float)

    raras = phi > limiar_phi
    resultado = {}

    # Sem instancias raras suficientes, as medidas nao sao calculaveis.
    # Vale registrar quantos datasets caem nesse caso -- e um achado sobre
    # o meta-dataset, nao uma falha do codigo.
    if raras.sum() < 5:
        for sigla in medidas_regiao:
            resultado[f"{sigla}_rara"] = np.nan
            resultado[f"razao_{sigla}"] = np.nan
        resultado["n_raras"] = int(raras.sum())
        return resultado

    global_ = complexidade_global or extrair_complexidade(
        X, y, medidas=list(medidas_regiao),
        max_amostras=max_amostras, random_state=random_state,
    )
    rara = extrair_complexidade(
        X[raras], y[raras], medidas=list(medidas_regiao),
        max_amostras=max_amostras, random_state=random_state,
    )

    for sigla in medidas_regiao:
        valor_rara = rara.get(sigla, np.nan)
        valor_global = global_.get(sigla, np.nan)
        resultado[f"{sigla}_rara"] = valor_rara

        if np.isnan(valor_rara) or np.isnan(valor_global) or valor_global == 0:
            resultado[f"razao_{sigla}"] = np.nan
        else:
            resultado[f"razao_{sigla}"] = valor_rara / valor_global

    resultado["n_raras"] = int(raras.sum())
    return resultado


if __name__ == "__main__":
    # Teste de sanidade replicando os datasets ilustrativos do artigo
    # (Fig. 1): Dataset1 tem um atributo linearmente ligado ao alvo;
    # Dataset2 tem apenas atributos aleatorios. Dataset1 deve ser
    # consistentemente mais simples.
    rng = np.random.default_rng(0)
    n = 300

    x1 = rng.random(n)
    X1 = np.column_stack([x1, rng.random(n)])
    y1 = x1.copy()  # alvo = atributo 1 (relacao linear perfeita)

    X2 = rng.random((n, 2))
    y2 = rng.random(n)  # alvo sem relacao com os atributos

    print(f"{'medida':>8} {'Dataset1':>12} {'Dataset2':>12}  esperado")
    esperado = {
        "C1": "D1 maior", "C2": "D1 maior", "C3": "D1 menor", "C4": "D1 menor",
        "L1": "D1 menor", "L2": "D1 menor", "L3": "D1 menor",
        "S1": "D1 menor", "S2": "D1 menor", "S3": "D1 menor", "S4": "D1 menor",
        "T2": "iguais",
    }
    r1 = extrair_complexidade(X1, y1)
    r2 = extrair_complexidade(X2, y2)
    for sigla in MEDIDAS:
        print(f"{sigla:>8} {r1[sigla]:>12.4f} {r2[sigla]:>12.4f}  {esperado[sigla]}")
