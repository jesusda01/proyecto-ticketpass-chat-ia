# Taller - Predicción de Entradas y Éxito de Conciertos (TicketPass)

## Objetivo
Comparar dos problemas de aprendizaje automático (Machine Learning) supervisado y publicar ambos modelos en una sola API REST FastAPI (`ticketpass_prediction_api`).

- **Clasificación:** predecir `es_sold_out` (Sí/No).
- **Regresión:** estimar `entradas_estimadas_vendidas` en unidades.

## Arquitectura
```text
        Predicción y Machine Learning - TicketPass
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
    CLASIFICACIÓN                       REGRESIÓN
 exito_conciertos.csv              ventas_conciertos.csv
     1,000 filas                       1,000 filas
          │                                 │
          ▼                                 ▼
 RandomForestClassifier             RandomForestRegressor
          │                                 │
          ▼                                 ▼
modelo_clasificacion_soldout.joblib  modelo_regresion_entradas.joblib
          │                                 │
          └────────────────┬────────────────┘
                           ▼
             ticketpass_prediction_api
                        FastAPI
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
   POST /sold_out                    POST /ventas
          │                                 │
    Sí / No (0/1)                   Entradas estimadas
                           │
                           ▼
                         Docker