"""
Implementa uma Rede Neural de duas camadas no PyTorch.
AVISO: você NÃO DEVE usar ".to()" ou ".cuda()" em cada bloco de implementação.
"""
import torch
import random
import statistics
from classificador_linear import amostrar_lote


def ola_rede_duas_camadas():
  """
  Esta é uma função de exemplo que tentaremos importar e executar para garantir 
  que nosso ambiente está configurado corretamente no Google Colab.
  """
  print('Olá do rede_duas_camadas.py!')


# Módulos de classe de modelo que usaremos mais tarde: Não edite / modifique esta classe
class RedeDuasCamadas(object):
  def __init__(self, tamanho_entrada, tamanho_oculta, tamanho_saida,
               dtype=torch.float32, device='cuda', std=1e-4):
    """
    Inicializa o modelo. Pesos são inicializados com pequenos valores aleatórios 
    e vieses são inicializados com zero. Pesos e vieses são armazenados na variável 
    self.params, que é um dicionário com as seguintes chaves:               

    W1: Pesos da primeira camada; tem shape (D, H)
    b1: Vieses da primeira camada; tem shape (H,)
    W2: Pesos da segunda camada; tem shape (H, C)
    b2: Vieses da segunda camada; tem shape (C,)

    Entrada:
    - tamanho_entrada: A dimensão D dos dados de entrada.
    - tamanho_oculta: O número de neurônios H na camada oculta.
    - tamanho_saida: O número de classes C.
    - dtype: Opcional, tipo de dados de cada parâmetro de peso.
    - device: Opcional, se os parâmetros de peso estão na GPU ou CPU.
    - std: Opcional, escala inicial dos parâmetros de peso.
    """
    # redefine a semente antes de começar
    random.seed(0)
    torch.manual_seed(0)

    self.params = {}
    self.params['W1'] = std * torch.randn(tamanho_entrada, tamanho_oculta, dtype=dtype, device=device)
    self.params['b1'] = torch.zeros(tamanho_oculta, dtype=dtype, device=device)
    self.params['W2'] = std * torch.randn(tamanho_oculta, tamanho_saida, dtype=dtype, device=device)
    self.params['b2'] = torch.zeros(tamanho_saida, dtype=dtype, device=device)

  def perda(self, X, y=None, reg=0.0):
    return rn_frente_tras(self.params, X, y, reg)

  def treinar(self, X, y, X_val, y_val,
              taxa_aprendizagem=1e-3, decaimento_taxa_aprendizagem=0.95,
              reg=5e-6, num_iters=100,
              tamanho_lote=200, verbose=False):
    return rn_treinar(
            self.params,
            rn_frente_tras,
            rn_prever,
            X, y, X_val, y_val,
            taxa_aprendizagem, decaimento_taxa_aprendizagem,
            reg, num_iters, tamanho_lote, verbose)

  def prever(self, X):
    return rn_prever(self.params, rn_frente_tras, X)

  def salvar(self, path):
    torch.save(self.params, path)
    print("Salvo em {}".format(path))

  def carregar(self, path):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint
    print("Carregando arquivo de ponto de verificação: {}".format(path))



def rn_passo_para_frente(params, X):
    """
    O primeiro estágio de nossa implementação da rede neural: Executar o 
    passo para frente da rede para calcular as saídas da camada oculta 
    e pontuações de classificação. A arquitetura da rede deve ser:

    camada FC -> ReLU (rep_oculta) -> camada FC (pontuações)

    Como prática, NÃO permitiremos o uso das operações torch.relu e torch.nn 
    apenas neste momento (você pode usá-las na próxima tarefa).

    Entrada:
    - params: Um dicionário de tensores do PyTorch que armazena os pesos de 
      um modelo. Deve ter as seguintes chaves com shape:
          W1: Pesos da primeira camada; tem shape (D, H)
          b1: Vieses da primeira camada; tem shape (H,)
          W2: Pesos da segunda camada; tem shape (H, C)
          b2: Vieses da segunda camada; tem shape (C,)
    - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.

    Retorno: Uma tupla de:
    - pontuacoes: Tensor de shape (N, C) com as pontuações de classificação para X
    - rep_oculta: Tensor de shape (N, H) com a representação da camada oculta
      para cada valor de entrada (depois da ReLU).
    """
    # Descompacte as variáveis do dicionário de parâmetros
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    # Calcule o passo para frente
    rep_oculta = None
    pontuacoes = None
    ############################################################################
    # TODO: Execute o passo para frente, calculando as pontuações de classe    #
    # para a entrada. Armazene o resultado na variável pontuacoes, que deve    #
    # ser um tensor de shape (N, C).                                           #
    ############################################################################
    # Substitua a instrução "pass" pelo seu código
    rep_oculta = torch.maximum(torch.tensor(0), torch.mm(X, W1) + b1)
    pontuacoes = torch.mm(rep_oculta, W2) + b2
    ###########################################################################
    #                             FIM DO SEU CODIGO                           #
    ###########################################################################

    return pontuacoes, rep_oculta


def rn_frente_tras(params, X, y=None, reg=0.0):
    """
    Calcula a perda e os gradientes de uma rede neural totalmente conectada de duas 
    camadas. Ao implementar perda e gradiente, por favor, não se esqueça de dimensionar 
    as perdas/gradientes pelo tamanho do lote.

    Entrada: Os primeiros dois parâmetros (params, X) são iguais a rn_passo_para_frente
    - params: Um dicionário de tensores do PyTorch que armazena os pesos de 
      um modelo. Deve ter as seguintes chaves com shape:
          W1: Pesos da primeira camada; tem shape (D, H)
          b1: Vieses da primeira camada; tem shape (H,)
          W2: Pesos da segundxa camada; tem shape (H, C)
          b2: Vieses da segunda camada; tem shape (C,)
    - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.
    - y: Vetor de rótulos de treinamento. y[i] é o rótulo de X[i], e cada y[i] é
      um número inteiro no intervalo 0 <= y[i] < C. Este parâmetro é opcional; Se 
      ele não for informado, retornamos apenas pontuações e, se for informado,
      em vez disso, retornamos a perda e os gradientes.
    - reg: Força de regularização.

    Retorno:
    Se y for None, retorna um tensor de pontuações de shape (N, C) onde pontuacoes[i, c] 
    é a pontuação para a classe c na entrada X[i].

    Se y não for None, em vez disso, retorna uma tupla de:
    - perda: Perda (perda de dados e perda de regularização) para amostras deste lote 
      de treinamento.
    - grads: Dicionário mapeando nomes de parâmetros aos gradientes desses parâmetros
      com relação à função de perda; tem as mesmas chaves que self.params.    
    """
    # Descompacte as variáveis do dicionário de parâmetros
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    pontuacoes, h1 = rn_passo_para_frente(params, X)
    # Se os rótulos não forem fornecidos, então retorne, nada precisa ser feito
    if y is None:
      return pontuacoes

    # Calcule a perda
    perda = None
    ############################################################################
    # TODO: Calcule a perda com base nos resultados de rn_passo_para_frente.   #
    # Deve incluir a perda de dados e de regularização L2 para W1 e W2.        #
    # Armazene o resultado na variável perda, que deve ser um escalar. Use a   #
    # perda do classificador Softmax. Ao implementar a regularização sobre W,  #
    # por favor, NÃO multiplique o termo de regularização por 1/2 (sem         #
    # coeficiente). Se você não for cuidadoso aqui, é fácil encontrar          #
    # instabilidade numérica (verifique a estabilidade numérica em             #
    # http://cs231n.github.io/linear-classify/).                               #
    ############################################################################
    # Substitua a instrução "pass" pelo seu código
    num_treino = X.shape[0]

    softmax = torch.exp(pontuacoes) / torch.sum(torch.exp(pontuacoes), dim=1).reshape(-1, 1)

    perda = torch.sum(-torch.log(softmax[range(num_treino), y]))
    perda /= num_treino
    perda += reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))
    ###########################################################################
    #                             FIM DO SEU CODIGO                           #
    ###########################################################################

    # Passo para trás: calcular gradientes
    grads = {}
    ###########################################################################
    # TODO: Execute o passo para trás, calculando as derivadas dos pesos e    #
    # vieses. Armazene os resultados no dicionário grads.                     #
    # Por exemplo, grads['W1'] deve armazenar o gradiente em W1, e ser um     #
    # tensor do mesmo tamanho.                                                #
    ###########################################################################
    # Substitua a instrução "pass" pelo seu código 
    softmax[range(num_treino), y] -= 1
    softmax /= num_treino

    # Gradiente em relação a W2 e b2
    grads['W2'] = torch.mm(h1.T, softmax)
    grads['W2'] += 2 * reg * W2
    grads['b2'] = torch.sum(softmax, dim=0)

    # Gradiente em relação a h1 (camada oculta)
    dh1 = torch.mm(softmax, W2.T)

    # Aplica a derivada da função de ativação
    dh1[h1 <= 0] = 0

    # Gradiente em relação a W1 e b1
    grads['W1'] = torch.mm(X.T, dh1) 
    grads['W1'] += 2 * reg * W1
    grads['b1'] = torch.sum(dh1, dim=0)
    ###########################################################################
    #                             FIM DO SEU CODIGO                           #
    ###########################################################################

    return perda, grads


def rn_treinar(params, func_perda, func_prev, X, y, X_val, y_val,
               taxa_aprendizagem=1e-3, decaimento_taxa_aprendizagem=0.95,
               reg=5e-6, num_iters=100,
               tamanho_lote=200, verbose=False):
  """
  Treina essa rede neural usando a descida de gradiente estocástico.

  Entrada:
  - params: Um dicionário de tensores do PyTorch que armazena os pesos de 
    um modelo. Deve ter as seguintes chaves com shape:
        W1: Pesos da primeira camada; tem shape (D, H)
        b1: Vieses da primeira camada; tem shape (H,)
        W2: Pesos da segunda camada; tem shape (H, C)
        b2: Vieses da segunda camada; tem shape (C,)
  - func_perda: Uma função de perda que calcula a perda e os gradientes.
    Recebe como entrada:
    - params: O mesmo que é fornecido para rn_treinar
    - X_lote: Um mini-lote de entradas de shape (B, D)
    - y_lote: Rótulos verdadeiros para X_lote
    - reg: O mesmo que é fornecido para rn_treinar
    E ele retorna uma tupla de:
      - perda: Escalar contendo a perda no mini-lote
      - grads: Dicionário mapeando nomes de parâmetros aos gradientes da perda
        com relação ao parâmetro correspondente.
  - func_prev: Função de previsão
  - X: Um tensor do PyTorch de shape (N, D) contendo dados de treinamento.
  - y: Um tensor do PyTorch de shape (N,) contendo rótulos de treinamento; 
    y[i] = c significa que X[i] tem rótulo c, onde 0 <= c < C.
  - X_val: Um tensor do PyTorch de shape (N_val, D) contendo dados de validação.
  - y_val: Um tensor do PyTorch de shape (N_val,) contendo rótulos de validação.
  - taxa_aprendizagem: escalar indicando taxa de aprendizagem para otimização.
  - decaimento_taxa_aprendizagem: escalar indicando o fator usado para diminuir 
    a taxa de aprendizagem após cada época.
  - reg: Escalar indicando a força de regularização.
  - num_iters: Número de passos a serem executados durante a otimização.
  - tamanho_lote: Número de amostras de treinamento a serem usados ​​em cada passo.
  - verbose: Booleano; se for verdadeiro imprime o progresso durante a otimização.

  Retorno: Um dicionário com estatísticas sobre o processo de treinamento.
  """
  num_amostras = X.shape[0]
  iteracoes_por_epoca = max(num_amostras // tamanho_lote, 1)

  # Use SGD para otimizar os parâmetros em self.model
  historico_perda = []
  historico_acc_treinamento = []
  historico_acc_validacao = []

  for it in range(num_iters):
    X_lote, y_lote = amostrar_lote(X, y, num_amostras, tamanho_lote)

    # Calcule a perda e os gradientes usando o mini-lote atual
    perda, grads = func_perda(params, X_lote, y=y_lote, reg=reg)
    historico_perda.append(perda.item())

    #########################################################################
    # TODO: Use os gradientes no dicionário grads para atualizar os         #
    # parâmetros da rede (armazenados no dicionário self.params) usando a   #
    # descida de gradiente estocástico. Você precisará usar os gradientes   #
    # armazenados no dicionário grads definido acima.                       #
    #########################################################################
    # Substitua a instrução "pass" pelo seu código
    params['W1'] -= grads['W1'] * taxa_aprendizagem
    params['b1'] -= grads['b1'] * taxa_aprendizagem

    params['W2'] -= grads['W2'] * taxa_aprendizagem
    params['b2'] -= grads['b2'] * taxa_aprendizagem
    #########################################################################
    #                            FIM DO SEU CODIGO                          #
    #########################################################################

    if verbose and it % 100 == 0:
      print('iteração %d / %d: perda %f' % (it, num_iters, perda.item()))

    # A cada época, verifique as acurácias de treinamento e de validação 
    # e reduza a taxa de aprendizagem.
    if it % iteracoes_por_epoca == 0:
      #  Verifique a acurácia
      y_prev_treinamento = func_prev(params, func_perda, X_lote)
      acc_treinamento = (y_prev_treinamento == y_lote).float().mean().item()
      y_pred_validacao = func_prev(params, func_perda, X_val)
      acc_validacao = (y_pred_validacao == y_val).float().mean().item()
      historico_acc_treinamento.append(acc_treinamento)
      historico_acc_validacao.append(acc_validacao)

      # Reduza a taxa de aprendizagem
      taxa_aprendizagem *= decaimento_taxa_aprendizagem

  return {
    'historico_perda': historico_perda,
    'historico_acc_treinamento': historico_acc_treinamento,
    'historico_acc_validacao': historico_acc_validacao,
  }


def rn_prever(params, func_perda, X):
  """
  Usa os pesos treinados desta rede de duas camadas para prever rótulos para
  os dados. Para cada amostra de dados, prevemos pontuações para cada uma das C
  classes e atribuímos cada amostra de dados à classe com a maior pontuação.

  Entrada:
  - params: Um dicionário de tensores do PyTorch que armazena os pesos de 
    um modelo. Deve ter as seguintes chaves com shape:
        W1: Pesos da primeira camada; tem shape (D, H)
        b1: Vieses da primeira camada; tem shape (H,)
        W2: Pesos da segunda camada; tem shape (H, C)
        b2: Vieses da segunda camada; tem shape (C,)
  - func_perda: Uma função de perda que calcula a perda e os gradientes.
  - X: Um tensor do PyTorch de shape (N, D) contendo N amostras D-dimensional 
    de dados para classificar.

  Retorno:
  - y_prev: Um tensor do PyTorch de shape (N,) contendo rótulos previstos para 
    cada uma das amostras de X. Para todo i, y_pred[i] = c significa que X[i] é 
    previsto ser da classe c, onde 0 <= c < C.    
  """
  y_prev = None

  ###########################################################################
  # TODO: Implemente esta função; deve ser MUITO simples!                   #
  ###########################################################################
  # Substitua a instrução "pass" pelo seu código
  probabilidades = func_perda(params, X)
  y_prev = torch.argmax(probabilidades, dim=1)
  ###########################################################################
  #                             FIM DO SEU CODIGO                           #
  ###########################################################################

  return y_prev



def rn_retorna_params_busca():
  """
  Retorna os hiperparâmetros candidatos para um modelo RedeDuasCamadas. 
  Você deve fornecer pelo menos dois parâmetros para cada um e o total de 
  combinações de busca em grade deve ser inferior a 256. Caso contrário, 
  levará muito tempo para treinar em tais combinações de hiperparâmetros.

  Retorno:
  - taxas_aprendizagem: candidatos a taxa de aprendizagem, por exemplo, 
                        [1e-3, 1e-2, ...]
  - tamanhos_oculta: tamanhos para a camada oculta, por exemplo, [8, 16, ...]
  - forcas_regularizacao: candidatos a forças de regularização, por exemplo, 
                          [1e0, 1e1, ...]
  - decaimentos_taxa_aprendizagem: candidatos a decaimento da taxa de 
                                   aprendizagem, por exemplo, [1.0, 0.95, ...]    
  """
  taxas_aprendizagem = []
  tamanhos_oculta = []
  forcas_regularizacao = []
  decaimentos_taxa_aprendizagem = []
  ###########################################################################
  # TODO: Adicione suas próprias listas de hiperparâmetros. Deve ser        #
  # semelhante aos hiperparâmetros usados para o SVM, mas pode ser          #
  # necessário selecionar hiperparâmetros diferentes para obter um bom      #
  # desempenho com o classificador softmax.                                 #
  ###########################################################################
  # Substitua a instrução "pass" pelo seu código
  taxas_aprendizagem = [1e-4, 1e-3, 1e0, 1e2]
  tamanhos_oculta = [128, 256]
  forcas_regularizacao = [1e-4, 1e-3]
  decaimentos_taxa_aprendizagem = [1, 0.95]
  ###########################################################################
  #                           FIM DO SEU CODIGO                             #
  ###########################################################################

  return taxas_aprendizagem, tamanhos_oculta, forcas_regularizacao, decaimentos_taxa_aprendizagem


def encontrar_melhor_rede(dic_dados, fn_retorna_params):
  """
  Ajuste de hiperparâmetros usando o conjunto de validação.
  Armazene seu modelo RedeDuasCamadas mais bem treinado em melhor_rede, com o 
  valor de retorno da operação ".treinar()" em melhor_estat e a acurácia de 
  validação do melhor modelo treinado em melhor_acc_validacao. Seus 
  hiperparâmetros devem ser obtidos a partir de rn_retorna_params_busca.    

  Entrada:
  - dic_dados (dicionário): Um dicionário que inclui
                            ['X_treino', 'y_treino', 'X_val', 'y_val']
                            como as chaves para treinar um classificador
  - fn_retorna_params (função): Uma função que fornece os hiperparâmetros
                                (p.ex., rn_retorna_params_busca) e retorna
                                (taxas_aprendizagem, tamanhos_oculta,
                                forcas_regularizacao, decaimentos_taxa_aprendizagem)
                                Você deve obter os hiperparâmetros de
                                fn_retorna_params.

  Retorno:
  - melhor_rede (instância): uma instância de RedeDuasCamadas treinada com
                             (['X_treino', 'y_treino'], tamanho_lote, 
                             taxa_aprendizagem, decaimento_taxa_aprendizagem, 
                             reg) por num_iter vezes.
  - melhor_estat (dicionário): valor de retorno da operação "melhor_rede.treinar()"
  - melhor_acc_validacao (float): acurácia de validação da melhor_rede
  """

  melhor_rede = None
  melhor_estat = None
  melhor_acc_validacao = 0.0

  #############################################################################
  # TODO: Ajuste hiperparâmetros usando o conjunto de validação. Armazene seu #
  # modelo mais bem treinado em melhor_rede.                                  #
  #                                                                           #
  # Para ajudar a depurar sua rede, pode ser útil usar visualizações          #
  # semelhantes às que usamos acima; essas visualizações terão diferenças     #
  # qualitativas significativas das que vimos para uma rede mal ajustada.     #
  #                                                                           #
  # Ajustar hiperparâmetros manualmente pode ser divertido, mas você pode     #
  # achar útil escrever código para varrer as possíveis combinações de        #
  # hiperparâmetros automaticamente, como fizemos nos exercícios anteriores.  #
  #############################################################################
  # Substitua a instrução "pass" pelo seu código
  taxas_aprendizagem, tamanhos_oculta, forcas_regularizacao, decaimentos_taxa_aprendizagem = fn_retorna_params()
  
  for tx in taxas_aprendizagem:
    for tam in tamanhos_oculta:
      for forca in forcas_regularizacao:
        for dec in decaimentos_taxa_aprendizagem:
          rede = RedeDuasCamadas(3 * 32 * 32, tam, 10, device=dic_dados['X_treino'].device, dtype=dic_dados['X_treino'].dtype)

          estats = rede.treinar(dic_dados['X_treino'], dic_dados['y_treino'], dic_dados['X_val'], 
                               dic_dados['y_val'], taxa_aprendizagem=tx, 
                               decaimento_taxa_aprendizagem=dec, reg=forca, 
                               num_iters=3000, tamanho_lote = 1000)

          acc_validacao = max(estats['historico_acc_validacao'])

          if acc_validacao > melhor_acc_validacao:
            melhor_rede = rede
            melhor_estat = estats
            melhor_acc_validacao = acc_validacao
  #############################################################################
  #                              FIM DO SEU CODIGO                            #
  #############################################################################

  return melhor_rede, melhor_estat, melhor_acc_validacao
