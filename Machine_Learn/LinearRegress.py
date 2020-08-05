import numpy as np
import os

#Para plotar imagens mais bonitinhas

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Para conseguir reproduzir os mesmos resultados 
np.random.seed(42)

#Caminho definido para salvar as imagens
#Mude-o de acordo com o diretório que você deseja salvar suas imagens
DIR = "C:/Aula04_regressao_pt2/"

#Se a pasta ainda não foi criada, então vamos cria-la
if not os.path.isdir(DIR):
      os.makedirs(DIR)
      print("Criando pasta")

#Função para salvar as imagens na pasta
def salvar_figura(fig_id, tight_layout=True):
    path = os.path.join(DIR, fig_id + ".png")
    print("Salvando figura", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def monta_dados_X():
    X = 2 * np.random.rand(100,1) 
    return X

def monta_dados_y():
    y = 4 + 3 * X + np.random.randn(100,1)
    return y

# Inicializa dados de entrada
X = monta_dados_X()
y = monta_dados_y()


plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
salvar_figura("generated_data_plot")
plt.show()
