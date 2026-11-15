import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# Simulamos un modelo entrenado para predecir el costo total del ticket según cantidad y zona
def train_and_save_model():
    # X = [cantidad, zona_cod (1:General, 2:Platea, 3:VIP, 4:Platinum)]
    X_train = np.array([[1, 1], [2, 3], [4, 2], [1, 4], [3, 3]])
    # y = costo total en soles
    y_train = np.array([70.0, 360.0, 320.0, 250.0, 540.0])
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "ticket_model.pkl")

if __name__ == "__main__":
    train_and_save_model()