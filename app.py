import os
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ==========================
# Funções de processamento
# ==========================

def carregar_csv(arquivo):
    try:
        df = pd.read_csv(arquivo.name)
        return df
    except Exception as e:
        return f"Erro ao carregar CSV: {e}"


def regressao_linear(df, coluna_x, coluna_y):
    try:
        df = pd.DataFrame(df)  # <<< IMPORTANTE!

        X = df[[coluna_x]].values
        y = df[coluna_y].values

        modelo = LinearRegression()
        modelo.fit(X, y)

        a = modelo.coef_[0]
        b = modelo.intercept_

        return f"Equação: y = {a:.4f}x + {b:.4f}"
    except Exception as e:
        return f"Erro na regressão linear: {e}"


# ==========================
# Interface Gradio
# ==========================

def interface_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Sistema de Álgebra — Regressão Linear")

        arquivo = gr.File(label="Envie seu arquivo CSV", file_types=[".csv"])
        botao_carregar = gr.Button("Carregar Dados")
        saida_df = gr.Dataframe(label="Prévia dos Dados", interactive=True)

        coluna_x = gr.Textbox(label="Nome da coluna X")
        coluna_y = gr.Textbox(label="Nome da coluna Y")
        botao_reg = gr.Button("Rodar Regressão Linear")
        saida_reg = gr.Textbox(label="Resultado")

        botao_carregar.click(fn=carregar_csv, inputs=arquivo, outputs=saida_df)
        botao_reg.click(fn=regressao_linear, inputs=[saida_df, coluna_x, coluna_y], outputs=saida_reg)

    return demo


if __name__ == "__main__":
    porta = int(os.environ.get("PORT", 8080))
    interface = interface_app()
    interface.queue().launch(
        server_name="0.0.0.0",
        server_port=porta,
        inbrowser=False,
        show_error=True
    )
