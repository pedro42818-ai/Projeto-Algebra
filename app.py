import os
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI

# ==========================
# Funções
# ==========================

def carregar_csv(arquivo):
    try:
        df = pd.read_csv(arquivo.name)
        return df
    except Exception as e:
        return f"Erro ao carregar CSV: {e}"

def regressao_linear(df, coluna_x, coluna_y):
    try:
        df = pd.DataFrame(df)  # Corrige entrada do Gradio
        X = df[[coluna_x]].values
        y = df[coluna_y].values

        modelo = LinearRegression()
        modelo.fit(X, y)

        a = modelo.coef_[0]
        b = modelo.intercept_

        return f"Equação: y = {a:.4f}x + {b:.4f}"
    except Exception as e:
        return f"Erro: {e}"

# ==========================
# Interface
# ==========================

def interface_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Sistema de Álgebra — Regressão Linear")

        arquivo = gr.File(label="Envie seu arquivo CSV", file_types=[".csv"])
        botao_carregar = gr.Button("Carregar Dados")
        saida_df = gr.Dataframe(label="Prévia dos Dados")

        coluna_x = gr.Textbox(label="Coluna X")
        coluna_y = gr.Textbox(label="Coluna Y")
        botao_reg = gr.Button("Rodar Regressão")
        saida_reg = gr.Textbox(label="Resultado")

        botao_carregar.click(carregar_csv, arquivo, saida_df)
        botao_reg.click(regressao_linear, [saida_df, coluna_x, coluna_y], saida_reg)

    return demo

# ==========================
# FastAPI + Gradio
# ==========================

app = FastAPI()

gradio_interface = interface_app()

# monta o gradio dentro do fastapi
app = gr.mount_gradio_app(app, gradio_interface, path="/")

# ==========================
# Execução local (não usada no Render)
# ==========================

if __name__ == "__main__":
    import uvicorn
    porta = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=porta)
