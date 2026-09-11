# H-GCL: Documento de Especificação Técnica e Design (SDD) para o Codex

Este documento fornece as especificações de baixo nível, schemas de dados exatos, pseudocódigos e as formulações matemáticas detalhadas necessárias para a implementação autônoma do framework **Heterogeneous Graph Contrastive Learning (H-GCL)** no dataset **Elliptic++** pelo **Codex**.

---

## 1. Schemas de Entrada e Saída de Dados (Data I/O)

O pipeline de dados deve ler cinco arquivos CSV principais e estruturá-los para o treinamento em duas fases (SSL e Downstream).

### A. Estrutura dos Arquivos de Entrada

1.  **`wallets_features.csv`**
    *   *Colunas:* `address` (str/int, identificador único da carteira), `time_step` (int, 1 a 49), e 56 atributos numéricos contínuos indexados de `feat_0` a `feat_55` (representando taxas, volumes e conectividade local).
    *   *Dimensão Esperada:* $[N_{\text{addr}} \times 58]$.
2.  **`txs_features.csv`**
    *   *Colunas:* `txId` (int, identificador único da transação), `time_step` (int, 1 a 49), e 183 atributos numéricos (sendo 94 locais e 89 agregados da vizinhança).
    *   *Dimensão Esperada:* $[N_{\text{tx}} \times 185]$.
3.  **`AddrTx_edgelist.csv` (Relação de Entrada/Input)**
    *   *Colunas:* `input_address` (origem, tipo endereço), `txId` (destino, tipo transação). Mapeia as moedas consumidas como entradas de uma transação.
4.  **`TxAddr_edgelist.csv` (Relação de Saída/Output)**
    *   *Colunas:* `txId` (origem, tipo transação), `output_address` (destino, tipo endereço). Mapeia a distribuição do valor para os endereços de saída.
5.  **`wallets_classes.csv`**
    *   *Colunas:* `address` (str/int), `class` (int). 
    *   *Mapeamento de Classes:* `1` = Ilícito (fraude), `2` = Lícito, `3` = Desconhecido (não rotulado, representando 71% das carteiras).

### B. Divisão Temporal e Máscaras de Treinamento

O Codex deve aplicar uma divisão temporal indutiva rigorosa baseada na evolução da rede Bitcoin:
*   **Subgrafo de Treino/Validação ($G_{\text{train}}$):** Passos temporais $1 \le t \le 34$.
*   **Subgrafo de Teste Indutivo ($G_{\text{test}}$):** Passos temporais $35 \le t \le 49$.

Duas máscaras booleanas devem ser criadas pelo Codex para gerenciar o fluxo de dados:
*   `ssl_mask`: Atribuída como `True` para **todos os nós** de endereço e transação no subgrafo de treino ($1 \le t \le 34$). O pré-treinamento do H-GCL ignora as classes e utiliza inclusive a Classe 3 (Desconhecidos).
*   `downstream_mask`: Atribuída como `True` apenas para nós de endereço em $G_{\text{train}}$ que possuam classe conhecida (`1` ou `2`). Nós de Classe 3 são mascarados como `False`.

---

## 2. Construção e Visualização do Grafo Heterogêneo no PyTorch Geometric

O Codex deve utilizar a biblioteca `torch_geometric.data.HeteroData` para construir um grafo bipartido direcionado.

### A. Inicialização do Objeto `HeteroData`

```python
import torch
from torch_geometric.data import HeteroData

data = HeteroData()

# 1. Definição das características dos nós
# x_addr: Tensor [N_addr, 56] contendo os atributos de wallets_features
data['address'].x = torch.tensor(x_addr, dtype=torch.float)

# x_tx: Tensor [N_tx, 183] contendo os atributos de txs_features
data['transaction'].x = torch.tensor(x_tx, dtype=torch.float)

# 2. Definição das arestas (Edge Lists)
# edge_index_sent: Tensor [2, E_sent] mapeando input_address -> txId
data['address', 'sent_to', 'transaction'].edge_index = torch.tensor(edge_index_sent, dtype=torch.long)

# edge_index_received: Tensor [2, E_recv] mapeando txId -> output_address
data['transaction', 'received_by', 'address'].edge_index = torch.tensor(edge_index_received, dtype=torch.long)
```

### B. Módulo de Visualização de Subgrafos com NetworkX

Para permitir a inspeção forense da topologia local de carteiras suspeitas sem estourar a memória do sistema, o Codex deve implementar a extração de subgrafos de $k$-hops utilizando `networkx`:

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_local_forensic_subgraph(hetero_data, target_address, hops=2, save_path="subgraph.png"):
    """
    Extrai e plota um subgrafo heterogêneo local centrado em target_address.
    """
    # Converter PyG HeteroData temporariamente para um grafo direcionado NetworkX
    G = nx.DiGraph()
    
    # Adicionar arestas sent_to (Address -> Tx)
    edges_sent = hetero_data['address', 'sent_to', 'transaction'].edge_index.t().numpy()
    for src, dst in edges_sent:
        G.add_edge(f"addr_{src}", f"tx_{dst}", relation="sent_to")
        
    # Adicionar arestas received_by (Tx -> Address)
    edges_recv = hetero_data['transaction', 'received_by', 'address'].edge_index.t().numpy()
    for src, dst in edges_recv:
        G.add_edge(f"tx_{src}", f"addr_{dst}", relation="received_by")
        
    # Extrair nós dentro do raio (hops) especificado usando busca em largura (BFS)
    start_node = f"addr_{target_address}"
    sub_nodes = {start_node}
    queue = [(start_node, 0)]
    visited = {start_node}
    
    while queue:
        curr, dist = queue.pop(0)
        if dist < hops:
            for neighbor in G.neighbors(curr):
                if neighbor not in visited:
                    visited.add(neighbor)
                    sub_nodes.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    
    sub_G = G.subgraph(sub_nodes)
    
    # Plotar com distinção de cores para nós de Endereço e Transação
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(sub_G, seed=42)
    
    node_colors = []
    for node in sub_G.nodes():
        if node.startswith("addr_"):
            node_colors.append("skyblue") # Carteiras
        else:
            node_colors.append("salmon")  # Transações
            
    nx.draw(sub_G, pos, with_labels=True, node_color=node_colors, 
            node_size=600, font_size=8, edge_color="gray", arrows=True)
    plt.title(f"Subgrafo Forense de {hops}-Hops Centrado em Endereço {target_address}")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
```

---

## 3. Arquitetura do R-GIN (Relational Graph Isomorphism Network)

Para processar a heterogeneidade das relações direcionadas do Bitcoin sem misturar semanticamente os atributos locais de carteiras com os de transações, o Codex deve implementar o **R-GIN**. 

### A. Formulação Matemática das Atualizações de Camada

A propagação de mensagens da camada $k-1$ para a camada $k$ ocorre em duas etapas matemáticas sequenciais:

#### 1. Atualização dos Nós de Transação ($tx$)
Os nós de transação agregam as mensagens das carteiras remetentes que as alimentaram por meio da relação `sent_to`:
$$h_{tx}^{(k)} = \text{MLP}_{\text{tx}}^{(k)} \left( (1 + \epsilon^{(k)}) \cdot h_{tx}^{(k-1)} + \sum_{u \in \mathcal{N}_{\text{sent\_to}}(tx)} h_u^{(k-1)} \right)$$

Onde:
*   $h_{tx}^{(k)} \in \mathbb{R}^{D_k}$ é a representação latente do nó de transação $tx$ na camada $k$.
*   $\mathcal{N}_{\text{sent\_to}}(tx)$ representa o conjunto de nós de endereços de entrada (vizinhos remetentes) de $tx$.
*   $\epsilon^{(k)}$ é um parâmetro aprendível ou fixado (geralmente inicializado em $0$).
*   $\text{MLP}_{\text{tx}}^{(k)}$ é uma rede neural multicamadas dedicada exclusivamente ao tipo transação.

#### 2. Atualização dos Nós de Endereço ($addr$)
Os nós de endereço atualizam suas representações integrando as mensagens das transações das quais receberam moedas por meio da relação `received_by`:
$$h_{addr}^{(k)} = \text{MLP}_{\text{addr}}^{(k)} \left( (1 + \epsilon^{(k)}) \cdot h_{addr}^{(k-1)} + \sum_{tx \in \mathcal{N}_{\text{received\_by}}(addr)} h_{tx}^{(k-1)} \right)$$

Onde:
*   $\mathcal{N}_{\text{received\_by}}(addr)$ é o conjunto de transações vizinhas que enviaram saídas (outputs) para a carteira $addr$.
*   $\text{MLP}_{\text{addr}}^{(k)}$ é uma rede neural dedicada exclusivamente ao tipo de nó endereço.

### B. Código de Implementação do Encoder R-GIN em PyG

```python
import torch.nn as nn
from torch_geometric.nn import Sequential, GINConv
from torch_geometric.nn import to_hetero

class RGINEncoder(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels):
        super().__init__()
        
        # 1. Definir redes GIN homogêneas de base
        # Usamos uma MLP de duas camadas internas dentro de cada GINConv
        self.conv_addr = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
        )
        self.conv_tx = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
        )
        
        # Projeções lineares iniciais para igualar as dimensões de entrada
        self.proj_addr = nn.Linear(in_channels_dict['address'], hidden_channels)
        self.proj_tx = nn.Linear(in_channels_dict['transaction'], hidden_channels)
        
    def forward(self, x_dict, edge_index_dict):
        # Projeção inicial de dimensão: [N, Features_Nativas] -> [N, Hidden_Channels]
        h_dict = {
            'address': F.relu(self.proj_addr(x_dict['address'])),
            'transaction': F.relu(self.proj_tx(x_dict['transaction']))
        }
        
        # Extração de arestas heterogêneas
        sent_edges = edge_index_dict[('address', 'sent_to', 'transaction')]
        recv_edges = edge_index_dict[('transaction', 'received_by', 'address')]
        
        # 1. Atualização de Transações (recebe mensagens de Address via sent_to)
        # Passamos a origem (address) e o destino (transaction) correspondente
        h_tx_new = self.conv_tx((h_dict['address'], h_dict['transaction']), sent_edges)
        
        # 2. Atualização de Addresses (recebe mensagens de Transaction via received_by)
        h_addr_new = self.conv_addr((h_dict['transaction'], h_dict['address']), recv_edges)
        
        return {
            'address': h_addr_new,
            'transaction': h_tx_new
        }
```

---

## 4. Algoritmos de Aumentações de Domínio

O motor de perturbação do H-GCL gera duas visões contrastivas alterando independentemente os atributos das carteiras e a topologia das conexões.

### A. Semantic Block Masking (Atributos)

Em vez de ocultar atributos aleatórios de forma cega, o Codex deve implementar o mascaramento de blocos funcionais completos de carteiras de forma a preservar a integridade das equações matemáticas do protocolo Bitcoin (ex: balanços totais).

#### **Mapeamento dos Índices de Atributos das Carteiras (56 Características):**
*   **Bloco 1 (Índices `0` a `15`):** Fluxos Financeiros Brutos (volumes enviados/recebidos).
*   **Bloco 2 (Índices `16` a `31`):** Custos e Tarifas (taxas de transação acumuladas, proporção de fees).
*   **Bloco 3 (Índices `32` a `47`):** Métricas de Conectividade Local (frequência de interação com vizinhos directos).
*   **Bloco 4 (Índices `48` a `55`):** Atributos Temporais e de Idade (tempo ativo de conta, dispersão entre blocos).

#### **Algoritmo de Mascaramento em PyTorch:**
```python
def semantic_block_masking(x_addr, p_mask=0.25):
    """
    Seleciona aleatoriamente um dos 4 blocos funcionais das carteiras 
    e zera os seus valores com base em p_mask.
    """
    x_aug = x_addr.clone()
    batch_size = x_addr.size(0)
    
    # Definição das fatias de índices dos blocos funcionais
    blocks = [
        (0, 16),   # Bloco 1: Fluxo Financeiro
        (16, 32),  # Bloco 2: Custos/Tarifas
        (32, 48),  # Bloco 3: Conectividade Local
        (48, 56)   # Bloco 4: Métricas Temporais
    ]
    
    for i in range(batch_size):
        if torch.rand(1).item() < p_mask:
            # Seleciona aleatoriamente um dos quatro blocos para mascarar
            block_idx = torch.randint(0, len(blocks), (1,)).item()
            start, end = blocks[block_idx]
            x_aug[i, start:end] = 0.0 # Zera o bloco funcional correspondente
            
    return x_aug
```

### B. Structural Edge Dropout Adaptativo por Densidade

A taxa de remoção de conexões direcionadas deve se ajustar de forma inversa à densidade estrutural de cada passo de tempo do grafo, protegendo passos esparsos de isolamento de nós.

#### **Formulação Matemática da Probabilidade de Descarte ($p_{\text{drop}}(t)$):**
$$p_{\text{drop}}(t) = \max \left( p_{\text{min}}, \min \left( p_{\text{max}}, p_{\text{base}} \times \frac{\bar{D}}{D(t)} \right) \right)$$

Onde:
*   $D(t) = \frac{|\mathcal{E}(t)|}{|\mathcal{V}(t)|}$ é a densidade de conexões (razão arestas/nós) calculada no passo temporal $t$.
*   $\bar{D}$ é a densidade de conexões média de todos os 49 passos temporais históricos do Elliptic++.
*   $p_{\text{base}}$ é a taxa de descarte de referência (inicialmente $0.15$).
*   $p_{\text{min}} = 0.05$ e $p_{\text{max}} = 0.30$ são os limites matemáticos de corte de segurança.

#### **Algoritmo de Descarte Adaptativo de Arestas em PyTorch:**
```python
def adaptive_edge_dropout(edge_index, time_steps, p_base=0.15, p_min=0.05, p_max=0.30):
    """
    Aplica descarte adaptativo de arestas heterogêneas baseado na densidade temporal local.
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    
    # 1. Calcular densidade por passo de tempo
    unique_t, counts_t = torch.unique(time_steps, return_counts=True)
    # Exemplo simplificado de densidade média de referência
    d_avg = 1.54 
    
    # Criar vetor de keep_mask booleano para as arestas
    keep_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
    
    for t in unique_t:
        t_val = t.item()
        # Filtrar nós ativos no timestamp t
        nodes_at_t = (time_steps == t_val).sum().item()
        # Obter arestas conectadas a nós desse timestamp (aresta aproximada por tempo de transação)
        # Substitui-se por filtragem real no grafo para calcular D(t)
        d_t = max(0.1, num_edges / max(1, nodes_at_t))
        
        # Calcular probabilidade de descarte adaptativa do passo t
        p_drop_t = p_base * (d_avg / d_t)
        p_drop_t = max(p_min, min(p_max, p_drop_t))
        
        # Gerar máscara de descarte para as arestas associadas ao passo temporal
        # Exclui-se aleatoriamente arestas da partição t com probabilidade p_drop_t
        rand_vals = torch.rand(num_edges, device=device)
        dropout_mask = (rand_vals < p_drop_t)
        
        keep_mask = keep_mask & (~dropout_mask)
        
    return edge_index[:, keep_mask]
```

---

## 5. Implementação Computacional Vetorizada da Perda InfoNCE

Para otimizar o tempo de execução do Codex em CPU/GPU e evitar loops de iteração demorados, a perda InfoNCE deve ser calculada de forma puramente matricial.

### A. Formulação Matemática Vetorizada

Seja um mini-batch contendo $B$ nós de carteira. O encoder H-GCL gera duas matrizes de representação latente de saídas das duas visões aumentadas:
*   $Z_1 \in \mathbb{R}^{B \times D}$ (Matriz obtida da visão com *Semantic Block Masking*).
*   $Z_2 \in \mathbb{R}^{B \times D}$ (Matriz obtida da visão com *Structural Edge Dropout*).

A otimização estrutural da perda InfoNCE é calculada nas seguintes etapas vetorizadas:

1.  **Normalização Euclidiana ($\ell_2$):** Projeta todas as representações diretamente na superfície de uma hiperesfera de raio unitário:
    $$\hat{Z}_1 = \frac{Z_1}{\|Z_1\|_2}, \quad \hat{Z}_2 = \frac{Z_2}{\|Z_2\|_2} \quad \in \mathbb{R}^{B \times D}$$
    Onde cada vetor de linha passa a ter norma $\|\hat{z}_i\|_2 = 1$.

2.  **Matriz de Similaridade de Cosseno Geral ($S$):** O produto matricial das representações normalizadas fornece as similaridades angulares cruzadas, que são então escaladas pela constante de temperatura $\tau = 0.07$:
    $$S = \frac{\hat{Z}_1 \hat{Z}_2^T}{\tau} \quad \in \mathbb{R}^{B \times B}$$
    Cada entrada $S_{i,j}$ desta matriz de dimensões $[B \times B]$ armazena a similaridade de cosseno escalada entre o nó $i$ da Visão 1 e o nó $j$ da Visão 2.

3.  **Vetor de Targets da Diagonal ($y$):** Dado que o par positivo de cada nó $i$ sob a Visão 1 é a sua versão alterada $i$ sob a Visão 2, as associações corretas estão localizadas na diagonal principal da matriz $S$. O vetor de alvos $y$ é uma sequência contínua de índices de classe de $0$ a $B-1$:
    $$y = [0, 1, 2, \dots, B-1]^T \quad \in \mathbb{R}^B$$

4.  **Entropia Cruzada Simétrica Bidirecional:** A perda total InfoNCE final é calculada aplicando a função Cross-Entropy Softmax convencional sobre as linhas e sobre as colunas da matriz $S$, forçando simetria de atração:
    $$\mathcal{L}_{\text{InfoNCE}} = \frac{1}{2} \left[ -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{S_{i,i}}}{\sum_{j=1}^{B} e^{S_{i,j}}} + \left( -\frac{1}{B} \sum_{j=1}^{B} \log \frac{e^{S_{j,j}}}{\sum_{i=1}^{B} e^{S_{i,j}}} \right) \right]$$

### B. Módulo de Perda Vetorizada em PyTorch

```python
import torch
import torch.nn.functional as F

class SymmetricalInfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2):
        """
        Calcula a perda InfoNCE de forma matricial paralela.
        z1 e z2 são tensores de embeddings de formato [B, D].
        """
        # Passo 1: Normalização L2 ao longo do eixo das dimensões (dim=1)
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        
        # Passo 2: Multiplicação matricial para gerar todas as BxB similaridades de cosseno
        # Divisão imediata pelo escalar de temperatura para calibração de escala
        similarity_matrix = torch.matmul(z1_norm, z2_norm.T) / self.temperature
        
        # Passo 3: Criação de labels da diagonal do mini-batch [0, 1, ..., B-1]
        batch_size = z1.size(0)
        labels = torch.arange(batch_size, device=z1.device)
        
        # Passo 4: Cálculo simétrico da perda utilizando cross entropy nativa
        loss_row_contrast = F.cross_entropy(similarity_matrix, labels)
        loss_col_contrast = F.cross_entropy(similarity_matrix.T, labels)
        
        # Média aritmética para estabilização de gradiente bidirecional
        return (loss_row_contrast + loss_col_contrast) / 2.0
```

---

## 6. Pipeline de Amostragem Estratificada e Fusão Tardia

Este módulo define como os classificadores downstream são treinados e avaliados sob escassez artificial de rótulos periciais (1%, 5%, 10%), garantindo que as classes lícitas e ilícitas mantenham a proporção exata da blockchain.

### A. Algoritmo de Amostragem Estratificada de Nós Rotulados

```python
import pandas as pd
import numpy as np

def generate_stratified_scarcity_masks(wallets_classes_csv_path, fraction=0.01, seed=42):
    """
    Lê o mapeamento de classes das carteiras, filtra os nós desconhecidos (Classe 3),
    e gera máscaras de treino/validação mantendo a taxa de desbalanceamento original (2% de fraude).
    """
    np.random.seed(seed)
    
    df = pd.read_csv(wallets_classes_csv_path) # [address, class]
    
    # 1. Isolar nós periciais reais (excluir Classe 3)
    df_labeled = df[df['class'].isin([1, 2])].copy()
    
    df_illicit = df_labeled[df_labeled['class'] == 1] # Fraudes (2% do dataset)
    df_licit = df_labeled[df_labeled['class'] == 2]   # Lícitos (98% do dataset)
    
    # 2. Calcular volume exato de amostras por classe com base na fração desejada
    size_illicit_train = max(1, int(len(df_illicit) * fraction))
    size_licit_train = max(1, int(len(df_licit) * fraction))
    
    # 3. Amostragem aleatória estratificada sem reposição
    train_illicit_sampled = df_illicit.sample(n=size_illicit_train, replace=False, random_state=seed)
    train_licit_sampled = df_licit.sample(n=size_licit_train, replace=False, random_state=seed)
    
    # Combinar nós de treino amostrados
    df_train_sampled = pd.concat([train_illicit_sampled, train_licit_sampled])
    
    # 4. Conjunto de validação e teste indutivo contendo todos os nós rotulados restantes
    df_eval_remaining = df_labeled[~df_labeled['address'].isin(df_train_sampled['address'])]
    
    # Retorna listas de identificadores (hashes de carteira) para indexar no PyG
    return df_train_sampled['address'].values, df_eval_remaining['address'].values
```

### B. Arquitetura do Mecanismo de Fusão Tardia (Late Fusion Engine)

O Codex deve configurar os dois ramos especialistas independentes e unificar suas decisões na camada de logits probabilísticos ponderada.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import torch.nn as nn
import torch.optim as optim

class LateFusionForensicsEngine:
    def __init__(self, r_gin_encoder, hidden_channels=128):
        # Ramo 1: Especialista Tabular (Random Forest)
        # Configurado de forma a atingir F1-Score ótimo em recursos locais
        self.tabular_specialist = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Ramo 2: Especialista Estrutural (H-GCL Encoder Congelado + MLP de Downstream)
        self.structural_encoder = r_gin_encoder
        self.structural_classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2), # Saída Logits [Classe Lícita, Classe Ilícita]
            nn.Softmax(dim=1)
        )
        
    def train_pipeline(self, train_addresses, train_labels, x_tabular, hetero_data):
        """
        Treina o Ramo 1 e o Ramo 2 de forma isolada sobre a fração de escassez.
        """
        # 1. Treinamento do Ramo Tabular (Random Forest)
        self.tabular_specialist.fit(x_tabular[train_addresses], train_labels)
        
        # 2. Extração dos Embeddings H-GCL do Ramo 2 (Congelado, sem atualização de gradientes)
        self.structural_encoder.eval()
        with torch.no_grad():
            embeddings_all = self.structural_encoder(hetero_data.x_dict, hetero_data.edge_index_dict)
            # Isolar apenas os embeddings das carteiras de treino amostradas
            train_embeddings = embeddings_all['address'][train_addresses]
            
        # 3. Treinar MLP rasa de classificação sobre os embeddings
        optimizer = optim.Adam(self.structural_classifier.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        self.structural_classifier.train()
        for epoch in range(50): # Treinamento rápido para evitar overfitting na escassez
            optimizer.zero_grad()
            outputs = self.structural_classifier(train_embeddings)
            loss = criterion(outputs, torch.tensor(train_labels, dtype=torch.long))
            loss.backward()
            optimizer.step()
            
    def predict_late_fusion(self, eval_addresses, x_tabular, hetero_data, w1=0.6, w2=0.4):
        """
        Executa a fusão tardia de decisões baseada na média ponderada de probabilidades.
        w1: Peso atribuído ao Ramo Tabular (Random Forest)
        w2: Peso atribuído ao Ramo Estrutural (H-GCL + MLP)
        """
        # Probabilidade do Ramo Tabular (Classes: [Lícito, Ilícito])
        prob_tabular = self.tabular_specialist.predict_proba(x_tabular[eval_addresses])
        
        # Probabilidade do Ramo Estrutural
        self.structural_encoder.eval()
        self.structural_classifier.eval()
        with torch.no_grad():
            embeddings_all = self.structural_encoder(hetero_data.x_dict, hetero_data.edge_index_dict)
            eval_embeddings = embeddings_all['address'][eval_addresses]
            prob_structural = self.structural_classifier(eval_embeddings).numpy()
            
        # Fusão Tardia: Média Ponderada
        prob_final = (w1 * prob_tabular) + (w2 * prob_structural)
        
        # Classe final predita (argmax sobre o vetor de probabilidades fundido)
        predictions = np.argmax(prob_final, axis=1)
        return predictions

    def optimize_weights_via_mcc(self, val_addresses, val_labels, x_tabular, hetero_data):
        """
        Busca em grade (Grid Search) os pesos w1 e w2 ótimos no conjunto de validação,
        utilizando o Matthews Correlation Coefficient (MCC) como métrica de otimização.
        """
        best_mcc = -1.0
        best_w1, best_w2 = 0.5, 0.5
        
        # Varredura em grade fina de pesos
        for w1 in np.linspace(0.0, 1.0, 21):
            w2 = 1.0 - w1
            preds = self.predict_late_fusion(val_addresses, x_tabular, hetero_data, w1=w1, w2=w2)
            # Otimizar MCC para mitigar o desbalanceamento de classe de 2% de fraude
            score_mcc = matthews_corrcoef(val_labels, preds)
            
            if score_mcc > best_mcc:
                best_mcc = score_mcc
                best_w1, best_w2 = w1, w2
                
        return best_w1, best_w2, best_mcc
```
