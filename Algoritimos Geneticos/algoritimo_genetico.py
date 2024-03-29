from random import random

class Produto():
    def __init__(self, nome, metcubica, valor):
        self.nome = nome;
        self.metcubica = metcubica
        self.valor = valor
        
class Individuo():
    def __init__(self, metcubica, valores, limite_metcubica, geracao=0):
        self.metcubica = metcubica
        self.valores = valores
        self.limite_metcubica = limite_metcubica
        self.nota_avaliacao = 0
        self.metcubica_usada = 0
        self.geracao = geracao
        self.cromossomo = []
        
        for i in range(len(metcubica)):
            if random() < 0.5:
                self.cromossomo.append("0")
            else:
                self.cromossomo.append("1")
                
    def avaliacao(self):
        nota = 0
        soma_metcubica = 0
        
        for i in range(len(self.cromossomo)):
           if self.cromossomo[i] == '1':
               nota += self.valores[i]
               soma_metcubica += self.metcubica[i]
               
        if soma_metcubica > self.limite_metcubica:
            nota = 1
            
        self.nota_avaliacao = nota
        self.metcubica_usada = soma_metcubica
        
    def crossover(self, outro_individuo):
        corte = round(random()  * len(self.cromossomo))
        
        filho1 = outro_individuo.cromossomo[0:corte] + self.cromossomo[corte::]
        filho2 = self.cromossomo[0:corte] + outro_individuo.cromossomo[corte::]
        
        filhos = [Individuo(self.metcubica, self.valores, self.limite_metcubica, self.geracao + 1),
                  Individuo(self.metcubica, self.valores, self.limite_metcubica, self.geracao + 1)]
        filhos[0].cromossomo = filho1
        filhos[1].cromossomo = filho2
        return filhos
    
    def mutacao(self, taxa_mutacao):
        #print("Antes %s " % self.cromossomo)
        for i in range(len(self.cromossomo)):
            if random() < taxa_mutacao:
                if self.cromossomo[i] == '1':
                    self.cromossomo[i] = '0'
                else:
                    self.cromossomo[i] = '1'
        #print("Depois %s " % self.cromossomo)
        return self
        
class AlgoritmoGenetico():
    def __init__(self, tamanho_populacao):
        self.tamanho_populacao = tamanho_populacao
        self.populacao = []
        self.geracao = 0
        self.melhor_solucao = 0
        
    def inicializa_populacao(self, metcubica, valores, limite_espacos):
        for i in range(self.tamanho_populacao):
            self.populacao.append(Individuo(metcubica, valores, limite_espacos))
        self.melhor_solucao = self.populacao[0]
        
    def ordena_populacao(self):
        self.populacao = sorted(self.populacao,
                                key = lambda populacao: populacao.nota_avaliacao,
                                reverse = True)
        
    def melhor_individuo(self, individuo):
        if individuo.nota_avaliacao > self.melhor_solucao.nota_avaliacao:
            self.melhor_solucao = individuo
            
    def soma_avaliacoes(self):
        soma = 0
        for individuo in self.populacao:
           soma += individuo.nota_avaliacao
        return soma
    
    def seleciona_pai(self, soma_avaliacao):
        pai = -1
        valor_sorteado = random() * soma_avaliacao
        soma = 0
        i = 0
        while i < len(self.populacao) and soma < valor_sorteado:
            soma += self.populacao[i].nota_avaliacao
            pai += 1
            i += 1
        return pai
     
if __name__ == '__main__':
    
    lista_produtos = []
    lista_produtos.append(Produto("Geladeira Dako", 0.751, 999.90))
    lista_produtos.append(Produto("Iphone 6", 0.0000899, 2911.12))
    lista_produtos.append(Produto("TV 55' ", 0.400, 4346.99))
    lista_produtos.append(Produto("TV 50' ", 0.290, 3999.90))
    lista_produtos.append(Produto("TV 42' ", 0.200, 2999.00))
    lista_produtos.append(Produto("Notebook Dell", 0.00350, 2499.90))
    lista_produtos.append(Produto("Ventilador Panasonic", 0.496, 199.90))
    lista_produtos.append(Produto("Microondas Electrolux", 0.0424, 308.66))
    lista_produtos.append(Produto("Microondas LG", 0.0544, 429.90))
    lista_produtos.append(Produto("Microondas Panasonic", 0.0319, 299.29))
    lista_produtos.append(Produto("Geladeira Brastemp", 0.635, 849.00))
    lista_produtos.append(Produto("Geladeira Consul", 0.870, 1199.89))
    lista_produtos.append(Produto("Notebook Lenovo", 0.498, 1999.90))
    lista_produtos.append(Produto("Notebook Asus", 0.527, 3999.00))
    
    #for produto in lista_produtos:
    #    print(produto.nome)

    cubagem = []
    valores = []
    nomes = []
    
    for produto in lista_produtos:
        cubagem.append(produto.metcubica)
        valores.append(produto.valor)
        nomes.append(produto.nome)
    limite = 3
    
    tamanho_populacao = 20
    ag = AlgoritmoGenetico(tamanho_populacao)
    ag.inicializa_populacao(cubagem, valores, limite)
    
    # Gera a nota de avaliação da população
    for individuo in ag.populacao:
        individuo.avaliacao()
    ag.ordena_populacao()
    ag.melhor_individuo(ag.populacao[0])
    soma = ag.soma_avaliacoes()
    nova_populacao = []
    probabilidade_mutacao = 0.01
    
    for individuos_gerados in range(0, ag.tamanho_populacao, 2):
        pai1 = ag.seleciona_pai(soma)
        pai2 = ag.seleciona_pai(soma)
        
        filhos = ag.populacao[pai1].crossover(ag.populacao[pai2])
        nova_populacao.append(filhos[0].mutacao(probabilidade_mutacao))
        nova_populacao.append(filhos[1].mutacao(probabilidade_mutacao))
        
    ag.populacao = list(nova_populacao)
    for individuo in ag.populacao:
        individuo.avaliacao()
    ag.ordena_populacao()
    ag.melhor_individuo(ag.populacao[0])        
    soma = ag.soma_avaliacoes()
    
    print("Melhor: %s" % ag.melhor_solucao.cromossomo, "\nValor: %s" % ag.melhor_solucao.nota_avaliacao)
