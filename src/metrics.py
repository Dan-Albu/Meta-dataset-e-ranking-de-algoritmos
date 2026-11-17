"""
Metricas para regressao desbalanceada -- implementacao em Python puro,
sem dependencia de R/rpy2 (decisao tomada para evitar a complexidade de
uma dependencia entre linguagens nesta atividade).

IMPORTANTE (cite no relatorio, secao 1.1): a phi() e a sera() abaixo sao
uma reimplementacao aproximada, validada empiricamente contra os valores
de %Rare publicados pela autora para os mesmos 30 datasets (ver docstring
de relevance() para a tabela comparativa). Baseiam-se em:
    Torgo & Ribeiro (2007), "Utility-based regression"
    Ribeiro & Moniz (2020), "Imbalanced regression and extreme value prediction"

O metodo original dos autores usa "adjusted boxplot" com interpolacao
Hermite para distribuicoes assimetricas, e a SERA oficial vem do pacote R
IRon. Aqui, phi() usa a regra classica de boxplot (Q1/Q3 + 1.5*IQR) --
mais simples, mas conceitualmente fiel (raro/extremo = relevante).
"""

import numpy as np


def relevance(y_values, y_reference=None, coef=1.5):
    """
    Calcula phi(y) para cada valor: 0 = valor "comum" (na mediana da
    distribuicao), 1 = valor extremo/raro (na cerca do boxplot ou alem),
    com rampa linear entre os dois.

    FORMULACAO: a rampa sobe da MEDIANA (phi = 0) ate a cerca do boxplot
    (phi = 1). Nao do Q3, como numa versao anterior deste arquivo.

    Por que a mediana e nao o Q3 -- validacao empirica contra referencia
    externa: o repositorio github.com/JusciAvelino/imbalancedRegression
    publica o %Rare dos MESMOS 30 datasets usados aqui, calculado com a
    implementacao original em R (pacotes UBL/IRon, via rpy2). Comparando
    as formulacoes contra esses valores de referencia:

        formulacao                        erro medio   correlacao
        rampa do Q3      (phi > 0.5)        3.66 pp       0.182
        rampa da mediana (phi > 0.8)        2.67 pp       0.687   <-- esta
        adjusted boxplot + pchip            5.49 pp       0.541
        outliers do boxplot padrao          5.36 pp       0.714

    A rampa do Q3 subestimava drasticamente a regiao rara em
    distribuicoes muito assimetricas -- erro acima de 14 pontos
    percentuais em meta (assimetria 14,7), forestFires (12,9) e
    analcatdata_apnea3 (5,0). A rampa da mediana reduz esses tres casos
    a menos de 3,3 pontos.

    LIMITACAO A DECLARAR NO RELATORIO: esta continua sendo uma
    aproximacao. O metodo original (Ribeiro, 2011) usa adjusted boxplot
    com interpolacao pchip entre pontos de controle, e nenhuma das
    reconstrucoes testadas reproduziu exatamente os valores publicados.
    O erro medio residual e de 2,67 pontos percentuais, concentrado em
    dois datasets de alvo DISCRETO com poucos valores distintos --
    wine_quality (nota de 3 a 9; divergencia de 19,9 pontos) e sensory
    (8,9 pontos) --, onde a nocao de cerca de boxplot se comporta de
    forma irregular. Nos demais 28 datasets a divergencia fica abaixo
    de 6 pontos percentuais.

    Parameters
    ----------
    y_values : array-like
        Valores para os quais calcular a relevancia (ex.: y de um fold).
    y_reference : array-like, opcional
        Distribuicao de referencia usada para calcular mediana e quartis
        (normalmente o alvo do dataset INTEIRO, nao so do fold -- a
        relevancia e propriedade do dataset, nao da particao de CV).
    coef : float
        Multiplicador do IQR que define a cerca. 1,5 e o valor classico
        de Tukey, adotado por Ribeiro (2011).

    Returns
    -------
    np.ndarray : valores de phi entre 0 e 1, mesmo formato de y_values.
    """
    y_values = np.asarray(y_values, dtype=float)
    ref = np.asarray(y_reference if y_reference is not None else y_values,
                     dtype=float)

    q1, mediana, q3 = np.percentile(ref, [25, 50, 75])
    iqr = q3 - q1

    # Protecao contra IQR EXATAMENTE zero (mais de 50% dos valores
    # identicos, o que zeraria o denominador da rampa).
    #
    # CUIDADO -- versao anterior deste arquivo usava a regra "IQR menor
    # que 1% da amplitude", que se mostrou ERRADA: ela confunde CAUDA
    # LONGA com valores repetidos. Em forestFires (maioria dos incendios
    # com area proxima de zero, poucos enormes), o IQR e legitimamente
    # pequeno frente a amplitude -- essa e a assinatura do
    # desbalanceamento, nao um defeito. A regra antiga inflava o IQR
    # para 25% da amplitude, empurrava a cerca para longe e zerava a
    # regiao rara exatamente nos datasets mais desbalanceados
    # (meta, forestFires, analcatdata_apnea3).
    if iqr <= 0:
        amplitude = ref.max() - ref.min()
        iqr = max(0.25 * amplitude, 1e-6)

    cerca_inferior = q1 - coef * iqr
    cerca_superior = q3 + coef * iqr

    phi = np.zeros_like(y_values)

    # cauda superior: rampa de 0 (na mediana) ate 1 (na cerca superior)
    acima = y_values >= mediana
    denom = (cerca_superior - mediana) or 1e-9
    phi[acima] = np.clip((y_values[acima] - mediana) / denom, 0, 1)

    # cauda inferior: rampa de 0 (na mediana) ate 1 (na cerca inferior)
    abaixo = y_values < mediana
    denom = (mediana - cerca_inferior) or 1e-9
    phi[abaixo] = np.clip((mediana - y_values[abaixo]) / denom, 0, 1)

    return phi


# Limiar que define a regiao rara. 0,8 segue Avelino et al. (2025) e e o
# valor com que a formulacao acima foi validada contra a referencia.
LIMIAR_RARO = 0.8


def phi_weighted_rmse(y_true, y_pred, phi_values, epsilon=1e-6):
    """
    RMSE ponderado pela relevancia: erros em valores raros pesam mais
    do que erros em valores comuns. Mesma unidade do RMSE tradicional
    (facil de explicar no relatorio) -- porem sensivel ao desbalanceamento.

    Parameters
    ----------
    y_true, y_pred : array-like
        Valores reais e preditos (de um fold de teste).
    phi_values : array-like
        Relevancia de cada y_true, calculada com relevance(y_true, y_reference=y_dataset_inteiro).
    epsilon : float
        Peso minimo para nao zerar completamente pontos com phi=0
        (evita ignorar totalmente a regiao "comum" do alvo).

    Returns
    -------
    float : menor = melhor (e uma metrica de erro).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    phi_values = np.asarray(phi_values, dtype=float)

    weights = phi_values + epsilon
    return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights)))


def sera(y_true, y_pred, phi_values, step=0.01):
    """
    SERA (Squared Error Relevance Area): integra o erro quadratico das
    observacoes cuja relevancia e >= t, para t variando de 0 a 1.
    Nao depende de escolher um unico limiar de "o que e raro" -- por isso
    e a metrica preferida na literatura de regressao desbalanceada.

    Parameters
    ----------
    y_true, y_pred : array-like
    phi_values : array-like
        Relevancia de cada y_true (ver relevance()).
    step : float
        Resolucao da grade de limiares t (0.01 = 100 pontos entre 0 e 1).

    Returns
    -------
    float : menor = melhor (e uma metrica de erro, mesma direcao do RMSE).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    phi_values = np.asarray(phi_values, dtype=float)

    sq_errors = (y_true - y_pred) ** 2
    thresholds = np.arange(0.0, 1.0 + step, step)

    area = 0.0
    for t in thresholds:
        mask = phi_values >= t
        if mask.any():
            area += sq_errors[mask].sum() * step

    return float(area)


if __name__ == "__main__":
    # Teste rapido -- rode `python metrics.py` para conferir que tudo importa
    # e funciona antes de usar dentro do pipeline principal.
    y_dataset = np.array([1, 2, 2, 3, 3, 3, 4, 4, 100])  # 100 e um valor extremo
    y_test = np.array([3, 4, 100])
    y_pred = np.array([3.2, 3.5, 60])

    phi_vals = relevance(y_test, y_reference=y_dataset)
    print("phi(y_test):", phi_vals)
    print("phi_weighted_rmse:", phi_weighted_rmse(y_test, y_pred, phi_vals))
    print("sera:", sera(y_test, y_pred, phi_vals))
