import gradio as gr
import pandas as pd
import requests

API_URL = "http://backend:8000/predict/"

def predict_soda(customer_id, product_id, year_week, X, Y, size, num_deliver_per_week):
    payload = {
        "customer_id": customer_id,
        "product_id": product_id,
        "year_week": year_week,
        "X": X,
        "Y": Y,
        "size": size,
        "num_deliver_per_week": num_deliver_per_week
    }
    response = requests.post(API_URL, json=payload)
    resp = response.json()
    return resp["prediction"]

iface = gr.Interface(
    fn=predict_soda,
    inputs=[
        gr.Number(label="Customer ID"),
        gr.Number(label="Product ID"),
        gr.Textbox(label="Year Week (e.g. week_23)"),
        gr.Number(label="X"),
        gr.Number(label="Y"),
        gr.Number(label="Size"),
        gr.Number(label="Num Deliver per Week")
    ],
    outputs=gr.Number(label="Predicted Demand"),
    description="Introduce los datos y pulsa Predict para ver la demanda estimada."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
