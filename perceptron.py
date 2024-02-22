import random
import numpy as np

# Conjunto de Treinamento

vetor_x = {
    'x1': {'vetor': [1, 1, 1, 1, 1, 0, 0, 1, 0, 0], 'yd1': 1, 'yd2': 0, 'yd3': 0},
    'x2': {'vetor': [1, 0, 0, 1, 1, 1, 1, 1, 0, 0], 'yd1': 0, 'yd2': 1, 'yd3': 0},
    'x3': {'vetor': [1, 0, 0, 1, 0, 1, 0, 1, 1, 1], 'yd1': 0, 'yd2': 0, 'yd3': 1}
}

treinamento = {1: vetor_x['x1'], 2: vetor_x['x2'], 3: vetor_x['x3']}

# 1º Passo: Inicializar as Sinapses Aleatoriamente

vetor_w = {
    'w1': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    'w2': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'w3': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
}

'''
def sinapses_aleatorias():
    vetor_ww = {
        'w1': [random.choice([0, 1]) for _ in range(10)],
        'w2': [random.choice([0, 1]) for _ in range(10)],
        'w3': [random.choice([0, 1]) for _ in range(10)]
    }
    print(vetor_ww)

sinapses_aleatorias()
'''

# 2º Passo: Treinar o próximo Vetor de entrada Xn

def treinar_vetorxn(vetor_x, vetor_w):
    resultados = {}

    for chave_x, info_x in vetor_x.items():
        vetor = info_x['vetor']
        resultados[chave_x] = {}

        for chave_w, sinapses in vetor_w.items():
            resultado = sum(x * w for x, w in zip(vetor, sinapses))
            resultados[chave_x][chave_w] = resultado

    return resultados


# 3º Passo: Calcular os erros dos neurônios

def calcular_erros(vetor_x, y,n):
    vetor_atual = f"x{n}"
    e1 = vetor_x[vetor_atual]['yd1'] - y[1]
    e2 = vetor_x[vetor_atual]['yd2'] - y[2]
    e3 = vetor_x[vetor_atual]['yd3'] - y[3]
    e = {1:e1, 2:e2, 3:e3}
    return e


# 4º Passo: Atualizar as sinapses

def delta_w(e, vetor_x, n, vetor_w):
    vetor_atual = f"x{n}"
    delta_w1 = 0.5 * e[1] * np.array(vetor_x[vetor_atual]['vetor'])
    delta_w2 = 0.5 * e[2] * np.array(vetor_x[vetor_atual]['vetor'])
    delta_w3 = 0.5 * e[3] * np.array(vetor_x[vetor_atual]['vetor'])

    delta_w = {1: delta_w1.tolist(), 2: delta_w2.tolist(), 3: delta_w3.tolist()}

    for i in range(1, 4):
        if all(value == 0 for value in delta_w[i]):
            delta_w[i] = vetor_w[f'w{i}']

    return delta_w


continua = True
n = 1

# Começar o treinamento

while continua:
    print("...")
    print(f"Início Treinamento número {n}")
    print("...")

    treinamento_atual = treinamento[n]
    print(f"Vetor X{n} = {treinamento_atual['vetor']}")

    # 1º passo: inicializar as sinapses aleatoriamente:
    print("\nInicializando as Sinapses Aleatoriamente: ")
    print(f"Vetor W1 = {vetor_w['w1']} ")
    print(f"Vetor W2 = {vetor_w['w2']} ")
    print(f"Vetor W3 = {vetor_w['w3']} ")

    # 2º passo: treinar o próximo vetor de entrada Xn
    resultados_vetor_entradaxn = treinar_vetorxn(vetor_x, vetor_w)
    print("\nTreinando o próximo vetor de entrada Xn:")
    x_atual = f"x{n}"

    resultados = resultados_vetor_entradaxn[x_atual]
    v1 = resultados['w1']
    v2 = resultados['w2']
    v3 = resultados['w3']
    print(f"Resultados para {x_atual} * Wi:")
    print(f"v1 = {x_atual} * w1 = {v1}")
    print(f"v2 = {x_atual} * w2 = {v2}")
    print(f"v3 = {x_atual} * w3 = {v3}")

    def check_y(num):
        if num > 0:
            return 1
        elif num <= 0:
            return 0

    v = {1: v1, 2: v2, 3: v3}

    # Condição y

    y = {}

    print("\nVerificando y = Se v > 0, então 1. Se v <= 0, então 0")
    for chave, valor in v.items():
        y[chave] = check_y(valor)
        print(f"Como v{chave} = {valor}. y{chave} = {y[chave]}")


    # 3º passo: calcular os erros dos neurônios
    print("\nPasso 3: Calcular erros dos neurônios")
    erros = calcular_erros(vetor_x, y, n)

    print(f"e1 = {erros[1]}")
    print(f"e2 = {erros[2]}")
    print(f"e3 = {erros[3]}")

    # 4º passo: atualizar as sinapses
    print("\nAtualizando as sinapses")
    deltaw = delta_w(erros, vetor_x, n, vetor_w)
    print(f"W1 = {deltaw[1]}")
    print(f"W2 = {deltaw[2]}")
    print(f"W3 = {deltaw[3]}")

    # Atualizar os valores das sinapses, ou seja, o w1, w2 e w3
    vetor_w['w1'] = deltaw[1]
    vetor_w['w2'] = deltaw[2]
    vetor_w['w3'] = deltaw[3]

    # 5º passo: cálculo do erro médio quadrático para o vetor Xn
    en = (pow(erros[1],2) + pow(erros[2],2) + pow(erros[3],2))/3
    print(f"O valor de E{n} = {round(en,3)}")
    en_valores = {}
    en_valores[n] = round(en,3)

    # 6º passo: Voltar ao passo 2 e treinar o próximo Xn
    # Verificar a confição para parada do loop

    n+= 1

    if(n > 3):
        continua = False
    else:
        continua = True

    '''
   
    def parar_loop(en_valores):
        soma_epoca = 0
        for chave in en_valores:
            soma_epoca += en_valores[chave]
        epoca = soma_epoca/3

        if epoca <= 0.0001:
            return False
        else:
            return True

    if(parar_loop(en_valores)) == False:
        continua = False
    elif(parar_loop()):
        continua = True
 '''
