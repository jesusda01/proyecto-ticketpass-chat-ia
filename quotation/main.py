from fastapi import FastAPI
import joblib
import numpy as np
import os


app = FastAPI(title="ticketpass_quotation_api")


# Cargar modelo al iniciar
if not os.path.exists("ticket_model.pkl"):
    import model
    model.train_and_save_model()


ml_model = joblib.load("ticket_model.pkl")


@app.post("/quotation")
def quotation_ticket(data: dict):
    # Mapeo de zonas a valores numéricos
    zona_map = {"General": 1, "Platea": 2, "VIP": 3, "Platinum": 4}
    zona_val = zona_map.get(data.get("zona"), 1)
    cantidad = float(data.get("cantidad", 1))
    
    features = np.array([[cantidad, zona_val]])
    cotizacion = ml_model.predict(features)[0]
    
    return {
        "evento": data.get("evento"),
        "zona": data.get("zona"),
        "cantidad": cantidad,
        "costo_estimado_pen": round(float(cotizacion), 2),
        "mensaje": "Cotización calculada exitosamente mediante modelo ML con Docker."
    }