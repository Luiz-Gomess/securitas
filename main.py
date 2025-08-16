import pandas as pd
from sklearn.ensemble import IsolationForest
from fastapi import FastAPI
import uvicorn
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Carregar os dados
try:
    df_logs = pd.read_csv('insercao_logs.csv', delimiter=';')
    df_pagamento = pd.read_csv('folha_pagamento_prefeitura.csv', delimiter=';')
except FileNotFoundError:
    print("Certifique-se de que os arquivos 'insercao_logs.csv' e 'folha_pagamento_prefeitura.csv' estão no mesmo diretório.")
    exit()

# --- Funções de Análise ---

def analise_padrao_insercao(df):
    features = ['dia_semana', 'hora', 'tentativas']
    X = df[features]

    model = IsolationForest(contamination=0.1, random_state=42)
    df['score_raw'] = model.fit_predict(X)
    df['anomalia'] = df['score_raw'] == -1
    scores = model.decision_function(X)

    resultados = []
    for index, row in df.iterrows():
        if row['anomalia']:
            explicacao = f"Padrão de inserção incomum detectado para o usuário {row['usuario']}. Detalhes: Dia da semana: {row['dia_semana']}, Hora: {row['hora']}, Tentativas: {row['tentativas']}."
            resultados.append({
                "id": row['id'],
                "usuario": row['usuario'],
                "etapa": "requisicao",
                "score": scores[index],
                "explicacao": explicacao
            })
    return sorted(resultados, key=lambda k: k['score'])

def analise_historico_consignante():
    # Criando dados simulados para demonstração
    data = {'id_contrato': range(100),
            'duracao_contrato_meses': np.random.randint(12, 60, 100),
            'mes_liquidacao': np.random.randint(1, 60, 100),
            'usuario': [f'user_{i}' for i in np.random.randint(1, 10, 100)]}
    df = pd.DataFrame(data)
    df['liquidacao_precoce'] = df['mes_liquidacao'] < (df['duracao_contrato_meses'] * 0.2)

    X = df[['duracao_contrato_meses', 'mes_liquidacao']]
    model = IsolationForest(contamination=0.1, random_state=42)
    df['score_raw'] = model.fit_predict(X)
    df['anomalia'] = df['score_raw'] == -1
    scores = model.decision_function(X)

    resultados = []
    for index, row in df.iterrows():
        if row['anomalia']:
            explicacao = f"Liquidação de contrato anômala para o usuário {row['usuario']}. Contrato de {row['duracao_contrato_meses']} meses liquidado no mês {row['mes_liquidacao']}."
            resultados.append({
                "id": row['id_contrato'],
                "usuario": row['usuario'],
                "etapa": "liquidacao_consignado",
                "score": scores[index],
                "explicacao": explicacao
            })
    return sorted(resultados, key=lambda k: k['score'])

def analise_margem_consignavel(df):
    df['margem_calculada'] = df['salario_liquido'] * 0.35 # Exemplo de margem de 35%
    df['diferenca_margem'] = df['margem_consignavel'] - df['margem_calculada']

    X = df[['diferenca_margem']]
    model = IsolationForest(contamination=0.1, random_state=42)
    df['score_raw'] = model.fit_predict(X)
    df['anomalia'] = df['score_raw'] == -1
    scores = model.decision_function(X)

    resultados = []
    for index, row in df.iterrows():
        if row['anomalia']:
            explicacao = f"Inconsistência na margem consignável para {row['nome']}. Margem liberada de R${row['margem_consignavel']:.2f} difere da margem calculada de R${row['margem_calculada']:.2f}."
            resultados.append({
                "id": row['id'],
                "usuario": row['nome'],
                "etapa": "liberacao_margem",
                "score": scores[index],
                "explicacao": explicacao
            })
    return sorted(resultados, key=lambda k: k['score'])

def analise_padroes_uso(df):
    le = LabelEncoder()
    df['usuario_encoded'] = le.fit_transform(df['usuario'])
    operacoes_por_usuario = df.groupby('usuario')['id'].count().reset_index().rename(columns={'id': 'n_operacoes'})

    X = operacoes_por_usuario[['n_operacoes']]
    model = IsolationForest(contamination=0.1, random_state=42)
    operacoes_por_usuario['score_raw'] = model.fit_predict(X)
    operacoes_por_usuario['anomalia'] = operacoes_por_usuario['score_raw'] == -1
    scores = model.decision_function(X)

    resultados = []
    for index, row in operacoes_por_usuario.iterrows():
      if row['anomalia']:
        explicacao = f"Volume de operações incomum para o usuário {row['usuario']} ({row['n_operacoes']} operações)."
        resultados.append({
            "id": index,
            "usuario": row['usuario'],
            "etapa": "uso_sistema",
            "score": scores[index],
            "explicacao": explicacao
        })
    return sorted(resultados, key=lambda k: k['score'])

# --- Configuração da API FastAPI ---

app = FastAPI(
    title="API de Detecção de Anomalias para Hackathon",
    description="Esta API utiliza o Isolation Forest para detectar anomalias em dados financeiros e de sistema.",
    version="1.0.0",
)

@app.get("/analise/padrao_insercao", tags=["Análises"])
async def endpoint_analise_insercao():
    """
    Detecta anomalias em padrões de inserção de dados, como horários e dias atípicos.
    """
    return {"resultados": analise_padrao_insercao(df_logs.copy())}

@app.get("/analise/historico_consignante", tags=["Análises"])
async def endpoint_analise_consignante():
    """
    Avalia se a liquidação de um consignado é compatível com o histórico (dados simulados).
    """
    return {"resultados": analise_historico_consignante()}

@app.get("/analise/margem_consignavel", tags=["Análises"])
async def endpoint_analise_margem():
    """
    Verifica inconsistências nas margens consignáveis liberadas.
    """
    return {"resultados": analise_margem_consignavel(df_pagamento.copy())}

@app.get("/analise/padroes_uso_sistema", tags=["Análises"])
async def endpoint_analise_uso():
    """
    Identifica usuários com volumes ou sequências de operações incomuns.
    """
    return {"resultados": analise_padroes_uso(df_logs.copy())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
