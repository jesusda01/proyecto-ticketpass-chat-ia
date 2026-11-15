import numpy as np
import pandas as pd
from pathlib import Path

# Configurar semilla para reproducibilidad
np.random.seed(42)
n_registros = 1000

# Listas base para simular variedad realista en Perú
artistas = ["Shakira", "Soda Stereo", "Juanes", "Bad Bunny", "Gian Marco", "Katy Perry", "Libido", "Daddy Yankee", "Karol G", "Marc Anthony"]
recintos = ["Estadio Nacional", "Estadio San Marcos", "Anfiteatro del Parque", "Arena 1", "Jockey Club"]
temporadas = ["Enero", "Febrero", "Julio", "Agosto", "Octubre", "Diciembre"]
descuentos = ["Si", "No"]
lanzamiento_fin_semana = [0, 1]

# ==========================================
# 1. GENERAR DATASET DE REGRESIÓN (ventas_conciertos.csv)
# ==========================================
print("Generando dataset de regresión (ventas de conciertos)...")
data_reg = {
    "artista": np.random.choice(artistas, n_registros),
    "oyentes_spotify_peru": np.random.randint(150000, 3000000, n_registros),
    "recinto": np.random.choice(recintos, n_registros),
    "aforo_maximo": np.random.choice([5000, 15000, 32000, 45000], n_registros),
    "precio_minimo_pen": np.random.randint(80, 350, n_registros),
    "dias_antelacion": np.random.randint(15, 180, n_registros),
    "temporada_mes": np.random.choice(temporadas, n_registros),
    "descuento_bancario": np.random.choice(descuentos, n_registros)
}

df_reg = pd.DataFrame(data_reg)

# Asignar una lógica matemática con algo de ruido aleatorio para que el modelo de regresión aprenda bien
df_reg["entradas_vendidas"] = (
    df_reg["oyentes_spotify_peru"] * 0.015 + 
    df_reg["aforo_maximo"] * 0.4 + 
    df_reg["dias_antelacion"] * 25 - 
    df_reg["precio_minimo_pen"] * 15 + 
    np.random.normal(0, 1500, n_registros)
).astype(int)

# Asegurar que las ventas no superen el aforo máximo ni sean negativas
df_reg["entradas_vendidas"] = df_reg[["entradas_vendidas", "aforo_maximo"]].min(axis=1)
df_reg["entradas_vendidas"] = df_reg["entradas_vendidas"].clip(lower=200)

# ==========================================
# 2. GENERAR DATASET DE CLASIFICACIÓN (exito_conciertos.csv)
# ==========================================
print("Generando dataset de clasificación (Sold Out)...")
data_clf = {
    "artista": np.random.choice(artistas, n_registros),
    "oyentes_spotify_peru": np.random.randint(150000, 3000000, n_registros),
    "precio_promedio_entrada": np.random.randint(90, 400, n_registros),
    "presupuesto_marketing_digital_usd": np.random.randint(1500, 30000, n_registros),
    "meses_desde_ultima_visita_pais": np.random.randint(2, 48, n_registros),
    "es_fin_de_semana_lanzamiento": np.random.choice(lanzamiento_fin_semana, n_registros)
}

df_clf = pd.DataFrame(data_clf)

# Asignar probabilidad de éxito (Sold Out = 1) en base a popularidad y marketing
score = (
    (df_clf["oyentes_spotify_peru"] > 800000).astype(int) * 3 +
    (df_clf["presupuesto_marketing_digital_usd"] > 10000).astype(int) * 2 +
    df_clf["es_fin_de_semana_lanzamiento"] +
    np.random.normal(0, 1.2, n_registros)
)

# Convertir a binario (0 o 1) usando un umbral
df_clf["es_sold_out"] = (score > 3).astype(int)

# ==========================================
# 3. GUARDAR EN LA CARPETA DATA
# ==========================================
base_dir = Path(__file__).resolve().parents[1]
data_dir = base_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

path_reg = data_dir / "ventas_conciertos.csv"
path_clf = data_dir / "exito_conciertos.csv"

df_reg.to_csv(path_reg, index=False)
df_clf.to_csv(path_clf, index=False)

print(f"¡Datasets generados con éxito!")
print(f" - {path_reg} ({len(df_reg)} filas)")
print(f" - {path_clf} ({len(df_clf)} filas)")