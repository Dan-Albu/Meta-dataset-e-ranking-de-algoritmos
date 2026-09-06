import numpy as np


def relevance(y_values, y_reference=None):

    y_values = np.asarray(y_values, dtype=float)
    ref = np.asarray(y_reference if y_reference is not None else y_values, dtype=float)

    q1, q3 = np.percentile(ref, [25, 75])
    iqr = q3 - q1

    # Protecao contra IQR degenerado (muitos valores repetidos/proximos):
    # sem isso, a divisao abaixo quase zera e qualquer ponto um pouco fora
    # da mediana vira phi=1 artificialmente, inflando a SERA sem sentido.
    # Se o IQR for menor que 1% do range total dos dados, usamos uma
    # fracao do range como IQR "efetivo".
    data_range = ref.max() - ref.min()
    min_iqr = 0.01 * data_range if data_range > 0 else 1e-6
    if iqr < min_iqr:
        iqr = max(0.25 * data_range, 1e-6)

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    phi = np.zeros_like(y_values)

    # cauda inferior: rampa de 0 (em q1) ate 1 (na cerca inferior ou alem)
    low_mask = y_values <= q1
    denom_low = (q1 - lower_fence) if (q1 - lower_fence) != 0 else 1e-9
    phi[low_mask] = np.clip((q1 - y_values[low_mask]) / denom_low, 0, 1)

    # cauda superior: rampa de 0 (em q3) ate 1 (na cerca superior ou alem)
    high_mask = y_values >= q3
    denom_high = (upper_fence - q3) if (upper_fence - q3) != 0 else 1e-9
    phi[high_mask] = np.clip((y_values[high_mask] - q3) / denom_high, 0, 1)

    return phi


def phi_weighted_rmse(y_true, y_pred, phi_values, epsilon=1e-6):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    phi_values = np.asarray(phi_values, dtype=float)

    weights = phi_values + epsilon
    return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights)))


def sera(y_true, y_pred, phi_values, step=0.01):

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
