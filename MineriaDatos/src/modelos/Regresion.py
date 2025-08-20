"""
Clase: Regresion

Objetivo: Clase enfocada en el procesamiento de datos y generación de modelos de
Regresión Lineal Simple y Múltiple, incluyendo cálculo de métricas y visualización
de resultados.

Cambios:
1. Solo variables numéricas en los modelos - fcontreras 14-08-2025
2. Gráficos lineales simples en regresión múltiple - fcontreras 14-08-2025
3. Resultados en tarjetas coloridas - fcontreras 14-08-2025
4. Eliminación de regresión logística - fcontreras 17-08-2025
5. Uso de tabs - fcontreras 17-08-2025
6. Uso del control de excepciones - fcontreras 18-08-2025
7. Generación del código del gráfico - fcontreras 18-08-2025
8. Arreglo de no elegir la misma variable para el análisis - fcontreras 19-08-2025
9. Uso del control de excepciones (mejorado) 2.0 - fcontreras 20-08-2025
10. Agregación del EDA fcontreras 20-08-2025
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


class Regresion:
    @staticmethod
    def regresion_lineal_simple(df: pd.DataFrame, x: str, y: str, test_size=0.2, random_state=42):
        """
        Realiza regresión lineal simple con una sola variable predictora (x) y una dependiente (y).
        """
        try:
            if x not in df.columns or y not in df.columns:
                raise ValueError("Variables no encontradas en el dataframe.")

            sub = df[[x, y]].dropna()
            if sub.empty:
                raise ValueError("No hay suficientes datos después de eliminar valores nulos.")

            X = sub[[x]].to_numpy().reshape(-1, 1)
            Y = sub[y].to_numpy().reshape(-1, 1)

            if len(X) < 2:
                raise ValueError("No hay suficientes registros para entrenar el modelo.")

            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=test_size, random_state=random_state
            )

            model = LinearRegression()
            model.fit(X_train, Y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            r2_train = r2_score(Y_train, y_pred_train)
            mse_train = mean_squared_error(Y_train, y_pred_train)
            rmse_train = np.sqrt(mse_train)

            r2_test = r2_score(Y_test, y_pred_test)
            mse_test = mean_squared_error(Y_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)

            coef = model.coef_[0][0]
            intercept = model.intercept_[0]

            results = {
                "coeficiente": float(coef),
                "intercepto": float(intercept),
                "R2_entrenamiento": float(r2_train),
                "MSE_entrenamiento": float(mse_train),
                "RMSE_entrenamiento": float(rmse_train),
                "R2_prueba": float(r2_test),
                "MSE_prueba": float(mse_test),
                "RMSE_prueba": float(rmse_test),
            }

            return {
                "modelo": model,
                "X_test": X_test.flatten(),
                "Y_test": Y_test.flatten(),
                "y_pred_test": y_pred_test.flatten(),
                "X_train": X_train.flatten(),
                "Y_train": Y_train.flatten(),
                "y_pred_train": y_pred_train.flatten(),
                "resultados": results,
            }
        except Exception as e:
            raise RuntimeError(f"Error en regresión lineal simple: {e}")

    @staticmethod
    def regresion_lineal_multiple(df: pd.DataFrame, x_vars: list[str], y: str):
        """
        Realiza regresión lineal múltiple con métricas de entrenamiento y prueba.
        """
        try:
            if not x_vars:
                raise ValueError("Debe seleccionar al menos una variable predictora.")

            if y not in df.columns:
                raise ValueError("Variable objetivo no encontrada.")

            for xi in x_vars:
                if xi not in df.columns:
                    raise ValueError(f"Predictora '{xi}' no encontrada.")

            sub = df[[*x_vars, y]].dropna()
            if sub.empty:
                raise ValueError("No hay suficientes datos después de eliminar valores nulos.")

            X = sub[x_vars]
            Y = sub[y]

            if len(X) < 2:
                raise ValueError("No hay suficientes registros para entrenar el modelo.")

            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, Y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            r2_train = r2_score(Y_train, y_train_pred)
            mse_train = mean_squared_error(Y_train, y_train_pred)
            rmse_train = np.sqrt(mse_train)

            r2_test = r2_score(Y_test, y_test_pred)
            mse_test = mean_squared_error(Y_test, y_test_pred)
            rmse_test = np.sqrt(mse_test)

            coeficientes = dict(zip(x_vars, model.coef_.tolist()))
            intercept = model.intercept_

            resultados = {
                "coeficientes": coeficientes,
                "intercepto": float(intercept),
                "R2_entrenamiento": float(r2_train),
                "R2_prueba": float(r2_test),
                "MSE_entrenamiento": float(mse_train),
                "MSE_prueba": float(mse_test),
                "RMSE_entrenamiento": float(rmse_train),
                "RMSE_prueba": float(rmse_test),
            }

            return {
                "modelo": model,
                "X": X,
                "Y": Y,
                "X_train": X_train,
                "Y_train": Y_train,
                "X_test": X_test,
                "Y_test": Y_test,
                "y_train_pred": y_train_pred,
                "y_test_pred": y_test_pred,
                "resultados": resultados,
            }
        except Exception as e:
            raise RuntimeError(f"Error en regresión lineal múltiple: {e}")
