# backend/main.py
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

# -------------------------------------------------------------------------
# Config & carga de dataset (una sola vez)
# -------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "week_54.parquet"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)        # columnas: customer_id, product_id,
                                       # week_num, prediction_proba, label

# -------------------------------------------------------------------------
# Modelos Pydantic
# -------------------------------------------------------------------------
class Pair(BaseModel):
    customer_id: int
    product_id: int
    prediction_proba: float

class ProductStat(BaseModel):
    product_id: int
    total_purchases: int

class TopCustomersResponse(BaseModel):
    n_returned: int
    pairs: List[Pair]

class PopularProductsResponse(BaseModel):
    threshold: float
    n_products: int
    products: List[ProductStat]

# -------------------------------------------------------------------------
# FastAPI
# -------------------------------------------------------------------------
app = FastAPI(
    title="Predicciones semana 54",
    description="API para consultar week_54.parquet",
    version="1.0.0"
)

@app.get("/top-customers", response_model=TopCustomersResponse)
def top_customers(top_n: int = Query(10, ge=1, le=10_000)):
    """
    Devuelve los *top_n* pares cliente-producto con mayor `prediction_proba`.
    """
    top_df = df.nlargest(top_n, "prediction_proba")[["customer_id",
                                                     "product_id",
                                                     "prediction_proba"]]

    result = [Pair(**row) for row in top_df.to_dict(orient="records")]
    return TopCustomersResponse(n_returned=len(result), pairs=result)


@app.get("/popular-products", response_model=PopularProductsResponse)
def popular_products(threshold: float = Query(0.5, ge=0.0, le=1.0)):
    """
    Para un *threshold*, cuenta cuántas compras (> threshold)
    tendrá cada producto y ordénalos de mayor a menor.
    """
    filt = df["prediction_proba"] >= threshold
    filtered = df.loc[filt]

    if filtered.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No hay registros con probability ≥ {threshold}"
        )

    counts = (filtered
              .groupby("product_id", as_index=False)
              .size()
              .rename(columns={"size": "total_purchases"})
              .sort_values("total_purchases", ascending=False))

    products = [ProductStat(**row)
                for row in counts.to_dict(orient="records")]

    return PopularProductsResponse(
        threshold=threshold,
        n_products=len(products),
        products=products
    )


# -------------------------------------------------------------------------
# Arranque local (útil para depurar fuera de Docker)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
