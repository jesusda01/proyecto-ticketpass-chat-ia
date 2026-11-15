from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

R = Path(__file__).resolve().parents[1]
d = pd.read_csv(R / 'data/ventas_conciertos.csv')

N = [
	'oyentes_spotify_peru',
	'aforo_maximo',
	'precio_minimo_pen',
	'dias_antelacion',
]
C = ['artista', 'recinto', 'temporada_mes', 'descuento_bancario']

X = d[N + C]
y = d.entradas_vendidas

Xt, Xv, yt, yv = train_test_split(
	X,
	y,
	test_size=.2,
	random_state=42,
)

pre = ColumnTransformer([
	('num', StandardScaler(), N),
	('cat', OneHotEncoder(handle_unknown='ignore'), C),
])

m = Pipeline([
	('preprocessor', pre),
	('model', RandomForestRegressor(
		n_estimators=250,
		max_depth=12,
		random_state=42,
		n_jobs=-1,
	)),
])

m.fit(Xt, yt)
p = m.predict(Xv)
q = {
	'mae': mean_absolute_error(yv, p),
	'rmse': mean_squared_error(yv, p) ** .5,
	'r2': r2_score(yv, p),
}

joblib.dump(m, R / 'models/modelo_regresion_entradas.joblib')
(R / 'models/metricas_regresion.json').write_text(
	json.dumps(q, indent=2),
)
print("Métricas Regresión (Ventas):")
print(json.dumps(q, indent=2))