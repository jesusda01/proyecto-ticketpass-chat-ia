from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

R = Path(__file__).resolve().parents[1]
# a: modelo de clasificación (Sold Out Anticipado)
a = joblib.load(R / 'models/modelo_clasificacion_soldout.joblib')
# c: modelo de regresión (Venta de Entradas)
c = joblib.load(R / 'models/modelo_regresion_entradas.joblib')


app = FastAPI(title="ticketpass_prediction_api")


class ClasificacionSoldOut(BaseModel):
  artista: str
  oyentes_spotify_peru: int = Field(ge=0)
  precio_promedio_entrada: float = Field(gt=0)
  presupuesto_marketing_digital_usd: float = Field(ge=0)
  meses_desde_ultima_visita_pais: int = Field(ge=0)
  es_fin_de_semana_lanzamiento: int = Field(ge=0, le=1)


class RegresionVentas(BaseModel):
  artista: str
  oyentes_spotify_peru: int = Field(ge=0)
  recinto: str
  aforo_maximo: int = Field(gt=0)
  precio_minimo_pen: float = Field(gt=0)
  dias_antelacion: int = Field(ge=0)
  temporada_mes: str
  descuento_bancario: str


@app.get('/health')
def health():
  return {
      'status': 'ok',
      'modelos': ['clasificacion_sold_out', 'regresion_ventas'],
  }


@app.post('/sold_out')
def sold_out(x: ClasificacionSoldOut):
  d = pd.DataFrame([x.model_dump()])
  y = int(a.predict(d)[0])
  p = float(a.predict_proba(d)[0][1])
  return {
      'es_sold_out': bool(y),
      'clase': y,
      'probabilidad_exito': round(p * 100, 2),
  }


@app.post('/ventas')
def ventas(x: RegresionVentas):
  d = pd.DataFrame([x.model_dump()])
  y = float(c.predict(d)[0])
  return {
      'entradas_estimadas_vendidas': int(round(y)),
      'moneda': 'PEN',
  }