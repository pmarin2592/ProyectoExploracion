"""
Clase: RegresionLineal

Objetivo: Clase enfocada en el procesamiento de datos y generación de modelos de
Regresión Lineal Simple y Múltiple, incluyendo cálculo de métricas y visualización
de resultados.

Cambios:
1. Cambio para que solo acepte variables numéricas en los modelos fcontreras 14-08-2025
2. Agregación de gráficos lineales simples para las variables en el modelo de regresión lineal multiple fcontreras 14-08-2025
3. Cambios con los resultados en las tarjetas coloridas fcontreras 14-08-2025
"""

import pandas as pd  # Para manejo de DataFrames
import numpy as np  # Para cálculos numéricos y arreglos

# Modelos de regresión lineal y logística
from sklearn.linear_model import LinearRegression, LogisticRegression

# Métricas de evaluación para regresión y clasificación
from sklearn.metrics import (
    r2_score,               # R² para evaluar regresión lineal
    mean_squared_error,      # Error cuadrático medio (MSE)
    accuracy_score,          # Exactitud en clasificación
    confusion_matrix,        # Matriz de confusión
    roc_curve,               # Valores para curva ROC
    auc,                     # Área bajo la curva ROC
    classification_report,   # Métricas detalladas de clasificación
)

from sklearn.model_selection import train_test_split  # Divide datos en entrenamiento y prueba
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Codificación y estandarización
from sklearn.compose import make_column_transformer  # Transformaciones por tipo de columna
from sklearn.pipeline import make_pipeline  # Crear pipelines de preprocesamiento + modelo


class Regresion:
    @staticmethod
    def _prepare_features(df, x_vars, drop_na=True, for_logistica=False):
        """
        Prepara variables predictoras:
        - Elimina NaN si drop_na=True
        - Codifica categóricas con OneHotEncoder
        - Escala numéricas si es regresión logística
        """
        df = df.copy()  # Copiamos el DataFrame para no alterar el original

        if drop_na:  # Si se indica, eliminamos filas con NaN en variables x
            df = df.dropna(subset=x_vars)

        X = df[x_vars]  # Seleccionamos las columnas predictoras

        # Identificamos columnas categóricas y numéricas
        categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()

        transformers = []  # Lista para transformaciones

        # Si hay categóricas, agregamos OneHotEncoder
        if categorical:
            transformers.append(
                ("cat", OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore"), categorical)
            )

        # Si hay numéricas y es para regresión logística, las escalamos
        if numeric and for_logistica:
            transformers.append(("num", StandardScaler(), numeric))

        # Creamos transformador que aplica transformaciones a las columnas seleccionadas
        preprocessor = make_column_transformer(*transformers, remainder="passthrough")

        # Aplicamos las transformaciones
        X_transformed = preprocessor.fit_transform(X)

        # Recuperamos nombres de las variables transformadas
        feature_names = []
        if categorical:
            ohe: OneHotEncoder = preprocessor.named_transformers_.get("cat", None)
            if ohe:
                cat_names = ohe.get_feature_names_out(categorical).tolist()
                feature_names.extend(cat_names)
        if numeric:
            feature_names.extend(numeric)

        # Retornamos la matriz transformada, nombres de columnas y el preprocesador
        return X_transformed, feature_names, preprocessor

    @staticmethod
    def regresion_lineal_simple(df: pd.DataFrame, x: str, y: str, test_size=0.2, random_state=42):
        """
        Realiza regresión lineal simple con una sola variable predictora (x) y una dependiente (y).
        """
        # Validamos que las columnas existan
        if x not in df.columns or y not in df.columns:
            raise ValueError("Variables no encontradas en el dataframe.")

        # Eliminamos filas con NaN en x o y
        sub = df[[x, y]].dropna()

        # Convertimos a arreglos de numpy y reordenamos para sklearn
        X = sub[[x]].to_numpy().reshape(-1, 1)
        Y = sub[y].to_numpy().reshape(-1, 1)

        # Dividimos en entrenamiento y prueba
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        # Creamos y entrenamos el modelo
        model = LinearRegression()
        model.fit(X_train, Y_train)

        # Hacemos predicciones para train y test
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculamos métricas para entrenamiento
        r2_train = r2_score(Y_train, y_pred_train)
        mse_train = mean_squared_error(Y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)

        # Calculamos métricas para prueba
        r2_test = r2_score(Y_test, y_pred_test)
        mse_test = mean_squared_error(Y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)

        # Obtenemos coeficiente e intercepto
        coef = model.coef_[0][0]
        intercept = model.intercept_[0]

        # Guardamos resultados
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

        # Retornamos modelo, predicciones y métricas
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

    @staticmethod
    def regresion_lineal_multiple(df: pd.DataFrame, x_vars: list[str], y: str):
        """
        Realiza regresión lineal múltiple con métricas de entrenamiento y prueba.
        """
        # Validamos que la variable objetivo exista
        if y not in df.columns:
            raise ValueError("Variable objetivo no encontrada.")

        # Validamos que todas las x existan
        for xi in x_vars:
            if xi not in df.columns:
                raise ValueError(f"Predictora '{xi}' no encontrada.")

        # Filtramos datos sin NaN
        sub = df[[*x_vars, y]].dropna()
        X = sub[x_vars]
        Y = sub[y]

        # División en entrenamiento y prueba (80% - 20%)
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Creamos y ajustamos modelo
        model = LinearRegression()
        model.fit(X_train, Y_train)

        # Predicciones entrenamiento y prueba
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Métricas entrenamiento
        r2_train = r2_score(Y_train, y_train_pred)
        mse_train = mean_squared_error(Y_train, y_train_pred)
        rmse_train = np.sqrt(mse_train)

        # Métricas prueba
        r2_test = r2_score(Y_test, y_test_pred)
        mse_test = mean_squared_error(Y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)

        # Obtenemos coeficientes e intercepto
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

        # Retornamos modelo y resultados
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

    @staticmethod
    def regresion_logistica(df: pd.DataFrame, x_vars: list[str], y: str):
        """
        Realiza regresión logística para clasificación binaria.
        """
        # Validamos variable objetivo
        if y not in df.columns:
            raise ValueError("Variable objetivo no encontrada.")

        # Filtramos datos sin NaN
        sub = df[[*x_vars, y]].dropna()
        Y = sub[y]

        # Validamos que la variable objetivo tenga solo 2 clases
        unique = Y.dropna().unique()
        if len(unique) != 2:
            raise ValueError(f"La variable objetivo debe tener exactamente 2 clases. Tiene: {unique.tolist()}")

        # Mapeamos clases a {0, 1} si no están ya en ese formato
        if set(unique) != {0, 1}:
            mapping = {unique[0]: 0, unique[1]: 1}
            Y_mapped = Y.map(mapping)
        else:
            Y_mapped = Y
            mapping = {0: 0, 1: 1}

        # Preparamos variables predictoras con codificación y escalado
        X_transformed, feature_names, preprocessor = Regresion._prepare_features(
            sub, x_vars, drop_na=True, for_logistica=True
        )

        # Creamos y entrenamos el modelo
        model = LogisticRegression(max_iter=1000)
        model.fit(X_transformed, Y_mapped)

        # Probabilidades y predicciones
        y_pred_proba = model.predict_proba(X_transformed)[:, 1]  # Probabilidad de clase 1
        y_pred = model.predict(X_transformed)

        # Métricas de evaluación
        accuracy = accuracy_score(Y_mapped, y_pred)
        cm = confusion_matrix(Y_mapped, y_pred)
        fpr, tpr, thresholds = roc_curve(Y_mapped, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        report = classification_report(Y_mapped, y_pred, output_dict=True)

        resultados = {
            "accuracy": float(accuracy),
            "confusion_matrix": cm.tolist(),
            "roc_auc": float(roc_auc),
            "classification_report": report,
            "feature_names": feature_names,
            "label_mapping": mapping,
        }

        # Retornamos modelo, métricas y curvas para ROC
        return {
            "modelo": model,
            "y_true": Y_mapped.to_numpy(),
            "y_pred": y_pred,
            "y_score": y_pred_proba,
            "fpr": fpr,
            "tpr": tpr,
            "resultados": resultados,
        }