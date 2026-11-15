from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)

R = Path(__file__).resolve().parents[1]
d = pd.read_csv(R / 'data/exito_conciertos.csv')

N = [
	'oyentes_spotify_peru',
	'precio_promedio_entrada',
	'presupuesto_marketing_digital_usd',
	'meses_desde_ultima_visita_pais',
]
C = ['artista', 'es_fin_de_semana_lanzamiento']

X = d[N + C]
y = d.es_sold_out

Xt, Xv, yt, yv = train_test_split(
	X,
	y,
	test_size=.2,
	random_state=42,
	stratify=y,
)

pre = ColumnTransformer([
	('num', StandardScaler(), N),
	('cat', OneHotEncoder(handle_unknown='ignore'), C),
])

m = Pipeline([
	('preprocessor', pre),
	('model', RandomForestClassifier(
		n_estimators=200,
		max_depth=8,
		class_weight='balanced',
		random_state=42,
	)),
])

m.fit(Xt, yt)
p = m.predict(Xv)
q = {
	'accuracy': accuracy_score(yv, p),
	'precision': precision_score(yv, p, zero_division=0),
	'recall': recall_score(yv, p, zero_division=0),
	'f1': f1_score(yv, p, zero_division=0),
	'confusion_matrix': confusion_matrix(yv, p).tolist(),
}

joblib.dump(m, R / 'models/modelo_clasificacion_soldout.joblib')
(R / 'models/metricas_clasificacion.json').write_text(
	json.dumps(q, indent=2),
)
print("Métricas Clasificación (Sold Out):")
print(json.dumps(q, indent=2))