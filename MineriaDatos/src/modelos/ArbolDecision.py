"""
Clase: ArbolDecision

Objetivo: Clase enfocada en el procesamiento de datos para Árbol de Decisión
"""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.datos.EstadisticasBasicasDatos import EstadisticasBasicasDatos


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArbolDecision:

    def __init__(self, eda: EstadisticasBasicasDatos, df, target_col, aplicar_binning=True, n_bins=5):
        self._eda = eda
        self.df = df  # Usar el df que se pasa (ya filtrado)
        self.target_col = target_col
        self.aplicar_binning = aplicar_binning
        self.n_bins = n_bins
        self.modelo = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.datos_preparados = False

    def var_continua(self, serie):
        """
        Determina si una variable es continua:
        - Es de tipo numérico (int o float)
        - Tiene más de 15 valores únicos
        """
        return pd.api.types.is_numeric_dtype(serie) and serie.nunique() > 15 #si hay más de 15 val unicos en el target manda el error

    def hacer_binning(self, serie, nombre_col):
        """Convierte variable continua en categórica con bins""" #además, se deja para evitar overfitting
        try:
            bins = pd.qcut(serie, q=self.n_bins, duplicates='drop') # el duplicates drop evita errores si hay datos repetidos que impiden hacer cortes exactos
            labels = [f"{nombre_col}Bin{i + 1}" for i in range(len(bins.cat.categories))]
            return pd.qcut(serie, q=self.n_bins, duplicates='drop', labels=labels[:len(bins.cat.categories)])
            """Crea etiquetas como edad_bin_1, edad_bin_2… para cada rango y devuelve una serie categórica, donde cada valor es un bin que se asignó"""
        except:
            try:
                bins = pd.cut(serie, bins=self.n_bins, duplicates='drop') #si los valores van de 0 a 100 y n_bins = 4, entonces hace cortes en 0–25, 25–50, etc
                labels = [f"{nombre_col}Bin{i + 1}" for i in range(len(bins.cat.categories))]
                return pd.cut(serie, bins=self.n_bins, labels=labels[:len(bins.cat.categories)], duplicates='drop')
            except:
                logger.warning(f"No se pudo aplicar binning a {nombre_col}") #Devuelve la serie sin cambios, para no romper el flujo.
                return serie

    def limpiar_preparar_datos(self):
        """Limpia y prepara los datos para el modelo"""
        # Validar que la columna objetivo exista
        if self.target_col not in self.df.columns:
            raise ValueError(f"La columna objetivo '{self.target_col}' no está en el dataset.")

        # Binning en la variable target si se desea forzarla como categórica
        if self.aplicar_binning and self.var_continua(self.df[self.target_col]):
            logger.info(f"Aplicando binning a la variable objetivo '{self.target_col}'")
            self.df[self.target_col] = self.hacer_binning(self.df[self.target_col], self.target_col)
        else:
            # Validar que la variable objetivo NO sea continua
            if self.var_continua(self.df[self.target_col]):
                raise ValueError(f"La variable objetivo '{self.target_col}' es continua y no puede usarse "
                                f"en un clasificador de árbol de decisión. Usá una variable categórica.")

        # Recién acá trabajás sobre una copia limpia
        df_limpio = self.df.copy()
        df_limpio = df_limpio.dropna()

        # Separar X e y
        X = df_limpio.drop(columns=[self.target_col])
        y = df_limpio[self.target_col].astype('category')

        # Binning si es necesario
        if self.aplicar_binning:
            for col in X.columns:
                if self.var_continua(X[col]):
                    X[col] = self.hacer_binning(X[col], col)

        # Agrupar categorías poco frecuentes
        columnas_categoricas = X.select_dtypes(include=["object", "category"]).columns
        for col in columnas_categoricas:
            if X[col].nunique() > 10:
                comunes = X[col].value_counts(normalize=True)
                comunes = comunes[comunes >= 0.05].index
                X[col] = X[col].apply(lambda x: x if x in comunes else "Otro")

        # Codificación
        X = pd.get_dummies(X)

        # Partición
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.datos_preparados = True

    def entrenar_modelo(self):
        """Entrena el modelo de árbol de decisión"""
        if not self.datos_preparados:
            logger.error("Los datos no han sido preparados. Ejecuta limpiar_preparar_datos() primero.")
            raise ValueError("Los datos no han sido preparados.")

        self.modelo = DecisionTreeClassifier(random_state=42)

        try:
            self.modelo.fit(self.X_train, self.y_train)
            logger.info("Modelo entrenado exitosamente")
        except ValueError as e:
            raise ValueError(f"Error al entrenar el modelo: {e}")

    def evaluar_modelo(self):
        """Evalúa el modelo"""
        if self.modelo is None:
            raise ValueError("El modelo no ha sido entrenado.")

        y_pred = self.modelo.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        rep = classification_report(self.y_test, y_pred, output_dict=True)
        conf = confusion_matrix(self.y_test, y_pred)
        return acc, rep, conf

    def obtener_importancia_variables(self):
        """Devuelve la importancia de las variables"""
        if self.modelo is None:
            return None
        importancias = pd.Series(self.modelo.feature_importances_, index=self.X_train.columns)
        return importancias.sort_values(ascending=False)

    def graficar_arbol(self, max_depth=3):
        """Genera el gráfico del árbol de decisión"""
        if self.modelo is None:
            logger.error("El modelo no ha sido entrenado")
            raise ValueError("El modelo no ha sido entrenado.")

        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(
            self.modelo,
            feature_names=self.X_train.columns,
            class_names=[str(c) for c in self.modelo.classes_],
            filled=True,
            max_depth=max_depth,
            fontsize=10
        )
        plt.title(f"Árbol de Decisión (Profundidad máxima: {max_depth})", fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def graficar_matriz_confusion(self):
        """Genera la matriz de confusión"""
        if self.modelo is None:
            logger.error("El modelo no ha sido entrenado")
            raise ValueError("El modelo no ha sido entrenado.")

        y_pred = self.modelo.predict(self.X_test)
        conf = confusion_matrix(self.y_test, y_pred)

        clases_presentes = np.unique(np.concatenate([self.y_test, y_pred]))
        conf_df = pd.DataFrame(conf, index=clases_presentes, columns=clases_presentes)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(conf_df, annot=True, fmt='d', cmap="Blues", ax=ax,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title("Matriz de Confusión", fontsize=16, fontweight='bold')
        plt.ylabel('Valores Reales', fontsize=12)
        plt.xlabel('Valores Predichos', fontsize=12)
        plt.tight_layout()
        return fig

    def graficar_importancia_variables(self, top_n=10):
        """Grafica la importancia de las variables"""
        importancias = self.obtener_importancia_variables()
        if importancias is None:
            return None

        top_importancias = importancias.head(top_n)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(top_importancias)), top_importancias.values,
                       color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(top_importancias)))
        ax.set_yticklabels(top_importancias.index)
        ax.set_xlabel('Importancia', fontsize=12)
        ax.set_title(f'Top {top_n} Variables más Importantes', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2.,
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        return fig