import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from examples.notebooks.spectral_function_fermion import spectral_function_fermion
from examples.notebooks.chevbyshev_approximation_fermion_spectral_function import scale_up, scale_down


def generate_paras(L, samples, n, train_file, **kwargs):
    part = L // 2
    names_anderson = ['U', 'ef'] + [f'eis_{i}' for i in range(part)] + [f'hoppings_{i}' for i in range(part)]
    # generate anderson config
    print('generate anderson config')
    anderson = []
    for i in range(samples):
        if i % 1000 == 0:
            print(f"=== [{i+1}/{samples}] anderson parameters. ===")
        t_i = np.random.uniform(0.0, 1.5, part)
        epsilon_i = np.random.uniform(-5.0, 5.0, part)
        U = np.random.uniform(0, 10, 1)
        epsilon_f = np.random.uniform(-2.5, 2.5, 1)
        data = np.concatenate((U, epsilon_f, epsilon_i, t_i))
        anderson.append(data)
        if i < 3:
            print(f"=== {i}-th anderson config is {data} ===")
            plt.figure()
            plt.plot(data, 'ro')
            plt.show()

    # get chebyshev polynomials
    m = n + 1  # the number of sample in chebyshev
    x_min = -25
    x_max = 25
    r_k = np.polynomial.chebyshev.chebpts1(m)
    names_alpha = ['poly_' + str(i) for i in range(n + 1)]
    # calculate the Chebyshev coefficients
    x_k = scale_up(r_k, x_min, x_max)
    total_alpha = []
    for i, paras in enumerate(anderson):
        if i % 20 == 0:
            print(f"=== [{i+1}/{samples}] chebyshev polynomials. ===")
        paras = np.array(paras)
        y_k = spectral_function_fermion(x_k, paras)
        # builds the Vandermonde matrix of Chebyshev polynomial expansion at the r_k nodes
        # using the recurrence relation
        T = np.polynomial.chebyshev.chebvander(r_k, n)
        alpha = np.linalg.inv(T.T @ T) @ T.T @ y_k
        total_alpha.append(alpha)
        if i < 3:
            plt.figure()
            plt.plot(np.arange(len(alpha)), alpha, 'ro', markersize=2)
            plt.show()

    # save to csv file
    parameters = np.concatenate((anderson, total_alpha), axis=1)
    names = names_anderson + names_alpha
    df = pd.DataFrame(parameters, columns=names)
    print(df.head(2).to_string())
    df.to_csv(train_file)
    return names


# get
if __name__ == '__main__':
    L = 7
    samples = 2000
    order_n = 256
    train_file = f'train_{samples}_{order_n}.csv'
    print(f"generate {train_file}...")
    columns_name = generate_paras(L, samples, order_n, train_file)

    # read dataframe
    df = pd.read_csv(train_file)
    df.loc[1]
    df.loc[1, "U"]
    df.loc[0, ["poly_0", "poly_1"]]
    df.iloc[0, 1]
    df.loc[:, ["U", "ef"]]
