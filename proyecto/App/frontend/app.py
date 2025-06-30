# frontend/app.py
import os
import requests
import pandas as pd
import gradio as gr

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# ------------------- helpers ------------------------------------------------
def get_top_clients(n):
    try:
        r = requests.get(f"{BACKEND_URL}/top-customers", params={"top_n": int(n)}, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        return gr.Error(f"Error llamando al backend: {exc}")

    df_out = pd.DataFrame(data["pairs"])
    return f"{data['n_returned']} registros devueltos", df_out


def get_popular_products(threshold):
    try:
        r = requests.get(f"{BACKEND_URL}/popular-products",
                         params={"threshold": float(threshold)}, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        return gr.Error(f"Error llamando al backend: {exc}")

    df_out = pd.DataFrame(data["products"])
    return (f"{data['n_products']} productos por encima de {threshold}",
            df_out)


# ------------------- interfaz Gradio ---------------------------------------
with gr.Blocks(title="Predicciones – semana 54") as demo:
    gr.Markdown("## 📈 Predicciones de compra (semana 54). \n"
                "Consulta a las predicciones **week_54.parquet**. La app nos permite ver que clientes-productos compraran con mayor probabilidad la proxima semana, el usuario puede ver los 10,20,30..etc. con mayor probabilidad de compra asi tambien ver que productos seran comprandos con cierta probabilidad para optimizar el stock "
                "Elige la pestaña que necesites:")

    with gr.Tab("🔝 Top clientes"):
        gr.Markdown("Introduce cuántos *pares cliente-producto* quieres ver:")
        top_n_input = gr.Number(value=10, precision=0, label="N (entero)")
        btn_top = gr.Button("Obtener")
        msg_top = gr.Text(interactive=False, label="Info")
        df_top = gr.Dataframe(label="Top clientes")
        btn_top.click(get_top_clients, inputs=top_n_input,
                      outputs=[msg_top, df_top])

    with gr.Tab("🛒 Productos populares para controlar el stock"):
        gr.Markdown("Filtra por probabilidad mínima de compra:")
        thr_input = gr.Slider(value=0.5, minimum=0.0, maximum=1.0,
                              step=0.05, label="Threshold")
        btn_prod = gr.Button("Obtener")
        msg_prod = gr.Text(interactive=False, label="Info")
        df_prod = gr.Dataframe(label="Productos populares")
        btn_prod.click(get_popular_products, inputs=thr_input,
                       outputs=[msg_prod, df_prod])

    gr.Markdown("> **Cómo usar la app**:  \n"
                "1. Ve a la pestaña que necesites.  \n"
                "2. Ajusta el valor (N o *threshold*).  \n"
                "3. Pulsa **Obtener** y espera la tabla.  \n"
                "Los datos provienen del backend FastAPI que lee "
                "`week_54.parquet que son las predicciones para la siguiente semana`.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
