# app.py
# Versão construída para seguir a lógica do seu notebook:
# - agrupa por CNPJ_Loja, calcula KPIs por loja
# - normaliza, faz PCA, cria score e ranking
# - gera gráficos (barra top N, scatter PCA)
# - monta interface Gradio (upload único) e roda via FastAPI/uvicorn (Render-ready)

import os
import io
import tempfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from fastapi import FastAPI
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

# -----------------------
# Funções de processamento
# -----------------------

def criar_matriz_lojas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega por CNPJ_Loja calculando KPIs básicos.
    Retorna DataFrame onde cada linha é uma loja.
    """
    # garanta nomes padronizados
    df = df.copy()
    # converter colunas que provavelmente são numéricas
    numeric_cols = ['Sell_In_Quantidade','Sell_In_Valor','Sell_Out_Quantidade','Sell_Out_Valor',
                    'Estoque_Inicial','Estoque_Final','Lead_Time','Giro_Estoque']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    agg = {
        'Sell_In_Quantidade': ['mean','sum','std'],
        'Sell_In_Valor': ['mean','sum'],
        'Sell_Out_Quantidade': ['mean','sum','std'],
        'Sell_Out_Valor': ['mean','sum'],
        'Estoque_Inicial': ['mean'],
        'Estoque_Final': ['mean'],
        'Lead_Time': ['mean'],
        'Giro_Estoque': ['mean'],
        'Categoria_Cosmetico': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        'Loja_Size': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    }

    # restrinja agregações às colunas presentes
    agg2 = {k:v for k,v in agg.items() if k in df.columns}

    grouped = df.groupby('CNPJ_Loja').agg(agg2)
    # ajustar multiindex de colunas
    grouped.columns = ['_'.join(filter(None,map(str, col))).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index().rename(columns={'index':'CNPJ_Loja'}).set_index('CNPJ_Loja')
    # renomeações úteis (quando existem)
    # garantindo nomes fixos para features usadas a seguir
    # se colunas não existirem, serão ignoradas no pipeline
    return grouped

def normalizar_matriz(matriz: pd.DataFrame, features: list) -> (pd.DataFrame, StandardScaler):
    """
    Normaliza as colunas 'features' retornando o dataframe normalizado e o scaler.
    """
    scaler = StandardScaler()
    X = matriz[features].fillna(0).values
    Xs = scaler.fit_transform(X)
    df_norm = pd.DataFrame(Xs, index=matriz.index, columns=features)
    return df_norm, scaler

def criar_score_investimento(matriz_norm: pd.DataFrame, original: pd.DataFrame) -> pd.Series:
    """
    Cria um score simples baseado em uma combinação ponderada de KPIs.
    Mantemos lógica interpretável:
      - vendas_total (Sell_Out_Quantidade_sum) positiva
      - receita_total (Sell_Out_Valor_sum) positiva
      - giro_estoque_media positiva
      - lead_time negativa (lead_time menor é melhor -> subtraímos)
    Se colunas não existirem, usamos apenas as disponíveis.
    """
    cols = matriz_norm.columns.tolist()
    score = pd.Series(0.0, index=matriz_norm.index)

    # mapeamento de nomes esperados no agregado
    mapping = {
        'vendas_total': 'Sell_Out_Quantidade_sum',
        'receita_total': 'Sell_Out_Valor_sum',
        'giro': 'Giro_Estoque_mean',
        'lead_time': 'Lead_Time_mean'
    }

    weights = {
        'vendas_total': 1.0,
        'receita_total': 1.0,
        'giro': 0.8,
        'lead_time': -0.6  # nota: negativa porque lead time menor é desejável
    }

    for k, colname in mapping.items():
        if colname in matriz_norm.columns:
            score += weights[k] * matriz_norm[colname]

    # normalize final score into 0-100
    if score.std() == 0:
        score_norm = pd.Series(50.0, index=score.index)
    else:
        score_norm = 50 + 10 * (score - score.mean()) / (score.std())  # média 50, desvio 10
    return score_norm

def calcular_pca(matriz_norm: pd.DataFrame, n_components=2) -> pd.DataFrame:
    pca = PCA(n_components=n_components)
    X = matriz_norm.fillna(0).values
    if X.shape[0] < 2:
        # sem dados suficientes
        res = np.zeros((X.shape[0], n_components))
    else:
        res = pca.fit_transform(X)
    df_pca = pd.DataFrame(res, index=matriz_norm.index, columns=[f'PC{i+1}' for i in range(n_components)])
    return df_pca

# -----------------------
# Visualização (matplotlib -> PIL-friendly)
# -----------------------
from PIL import Image

def fig_to_image_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

def plot_top_scores(ranking_df, top_n=10):
    top = ranking_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, top_n*0.4)))
    ax.barh(top.index.astype(str), top['score'], align='center')
    ax.invert_yaxis()
    ax.set_xlabel('Score')
    ax.set_title(f'Top {top_n} Lojas por Score')
    plt.tight_layout()
    img = fig_to_image_bytes(fig)
    plt.close(fig)
    return img

def plot_pca_scatter(pca_df, ranking_df, highlight_n=5):
    df = pca_df.join(ranking_df[['score']])
    fig, ax = plt.subplots(figsize=(7,6))
    sc = ax.scatter(df['PC1'], df['PC2'], s=50, alpha=0.7)
    for i, idx in enumerate(df.index):
        if i < 0: pass
    # destacar top N
    top_idx = ranking_df.head(highlight_n).index
    for idx in top_idx:
        if idx in df.index:
            ax.scatter(df.loc[idx,'PC1'], df.loc[idx,'PC2'], s=120, edgecolor='red', facecolor='none', linewidth=1.5)
            ax.text(df.loc[idx,'PC1'], df.loc[idx,'PC2'], str(idx), fontsize=8)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA das Lojas (melhores destacadas)')
    plt.tight_layout()
    img = fig_to_image_bytes(fig)
    plt.close(fig)
    return img

# -----------------------
# Pipeline principal
# -----------------------

def processar_arquivo(file_obj, top_n=10):
    """
    Função principal chamada pelo Gradio.
    Recebe arquivo enviado, retorna:
      - resumo markdown
      - ranking (pandas.DataFrame) para exibição
      - imagem de top scores (PIL.Image)
      - imagem de PCA (PIL.Image)
      - caminho para CSV com ranking para download
    """
    try:
        df = pd.read_csv(file_obj.name, parse_dates=[col for col in ['Data'] if 'Data' in pd.read_csv(file_obj.name, nrows=0).columns])
    except Exception as e:
        return f"Erro ao ler CSV: {e}", None, None, None, None

    # verifica coluna chave
    if 'CNPJ_Loja' not in df.columns:
        return "Arquivo inválido: coluna 'CNPJ_Loja' não encontrada.", None, None, None, None

    matriz = criar_matriz_lojas(df)

    # escolha de features numéricas a usar para normalização (apenas as que existirem)
    possible_features = [
        'Sell_Out_Quantidade_sum','Sell_Out_Valor_sum',
        'Sell_In_Quantidade_sum','Sell_In_Valor_sum',
        'Giro_Estoque_mean','Lead_Time_mean',
        'Estoque_Final_mean','Estoque_Inicial_mean'
    ]
    features = [f for f in possible_features if f in matriz.columns]
    if len(features) == 0:
        return "Nenhuma feature numérica encontrada para análise.", None, None, None, None

    matriz_norm, scaler = normalizar_matriz(matriz, features)

    # score e ranking
    score = criar_score_investimento(matriz_norm, matriz)
    ranking = pd.DataFrame({
        'CNPJ_Loja': matriz.index.astype(str),
        'score': score
    }).set_index('CNPJ_Loja').sort_values('score', ascending=False)

    # PCA
    pca_df = calcular_pca(matriz_norm, n_components=2)

    # gráficos
    img_top = plot_top_scores(ranking, top_n=top_n)
    img_pca = plot_pca_scatter(pca_df, ranking, highlight_n=top_n if top_n>0 else 5)

    # salvar CSV temporário com ranking para download
    tmpdir = tempfile.gettempdir()
    csv_path = os.path.join(tmpdir, f"ranking_lojas_{int(pd.Timestamp.now().timestamp())}.csv")
    ranking.reset_index().to_csv(csv_path, index=False)

    # criar um resumo em Markdown
    resumo = []
    resumo.append(f"**Lojas analisadas:** {len(matriz)}")
    resumo.append(f"**Features usadas na análise:** {', '.join(features)}")
    resumo.append(f"**Top {top_n} lojas geradas:**")
    top_list = ranking.head(top_n).reset_index()
    for i,row in top_list.iterrows():
        resumo.append(f"- {row.name}: Score {row['score']:.2f}")

    resumo_md = "\n\n".join(resumo)

    return resumo_md, ranking.reset_index(), img_top, img_pca, csv_path

# -----------------------
# Interface Gradio
# -----------------------

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Análise automatizada de lojas — Ranking e Recomendações")
        gr.Markdown("Faça upload do CSV (formato esperado com coluna `CNPJ_Loja`, `Data`, `Sell_In_Quantidade`, `Sell_Out_Quantidade`, `Giro_Estoque`, `Lead_Time`, etc.). O sistema processa tudo automaticamente e gera ranking, gráficos e arquivo para download.")

        with gr.Row():
            arquivo = gr.File(label="Envie seu arquivo CSV", file_types=[".csv"])
            top_n = gr.Slider(value=10, minimum=1, maximum=30, step=1, label="Top N lojas a destacar")

        btn = gr.Button("Analisar arquivo")

        with gr.Tabs():
            with gr.TabItem("Resumo"):
                saida_md = gr.Markdown()
            with gr.TabItem("Ranking"):
                saida_table = gr.Dataframe(headers=None)
                download_btn = gr.File(label="Download CSV do ranking")
            with gr.TabItem("Visualizações"):
                img_top_out = gr.Image(type="pil")
                img_pca_out = gr.Image(type="pil")

        def run_and_return(file_obj, top_n_val):
            return processar_arquivo(file_obj, top_n=int(top_n_val))

        btn.click(fn=run_and_return, inputs=[arquivo, top_n], outputs=[saida_md, saida_table, img_top_out, img_pca_out, download_btn])

    return demo

# -----------------------
# FastAPI mount (Render-ready)
# -----------------------

app = FastAPI()
gradio_app = build_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    import uvicorn
    porta = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=porta)
