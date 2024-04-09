import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Carregando o arquivo de teste
print('Carregando Arquivo de teste')
arquivo = np.load('teste1.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

# Definição das configurações de arquitetura a serem testadas
architectures = [
    (3,),            # 1 camada oculta com 3 neurônios
    (6,),            # 1 camada oculta com 6 neurônios
    (2, 3),          # 2 camadas ocultas com 3 neurônios cada
]

# Número de execuções para cada configuração
Execucao = 10

# Resultados
results = []

for architecture in architectures:
    print(f"Arquitetura: {architecture}")
    errors = []
    for _ in range(Execucao):
        # Criando e treinando o modelo
        regr = MLPRegressor(hidden_layer_sizes=architecture,
                            max_iter=100,
                            activation='relu',  # Você pode alterar a função de ativação conforme necessário
                            solver='adam',
                            learning_rate='adaptive',
                            n_iter_no_change=50)
        regr.fit(x, y)

        # Fazendo previsões
        y_est = regr.predict(x)

        # Calculando o erro
        error = mean_squared_error(y, y_est)
        errors.append(error)

    # Calculando média e desvio padrão dos erros
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    print(f"Média do erro: {mean_error}")
    print(f"Desvio padrão do erro: {std_error}")
    results.append((architecture, mean_error, std_error))

# Exibindo os resultados
for architecture, mean_error, std_error in results:
    print(f"\nArquitetura: {architecture}\n Média do erro: {mean_error}\n Desvio padrão do erro: {std_error}")


plt.figure(figsize=[14,7])

#plot curso original
plt.subplot(1,3,1)
plt.plot(x,y)

#plot aprendizagem
plt.subplot(1,3,2)
plt.plot(regr.loss_curve_)

#plot regressor
plt.subplot(1,3,3)
plt.plot(x,y,linewidth=1,color='yellow')
plt.plot(x,y_est,linewidth=2)



plt.show()