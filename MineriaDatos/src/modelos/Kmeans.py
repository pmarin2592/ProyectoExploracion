"""
Archivo: Kmeans.py

Descripción: Clases para análisis de clustering K-means y clustering jerárquico manual,
incluye escalamiento, entrenamiento, evaluación y visualización básica.
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.datos.EstadisticasBasicasDatos import EstadisticasBasicasDatos

# Importar métricas para clustering jerárquico
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class Kmeans:
    def __init__(self, datos):
        """
        datos: array-like o dataframe con datos ya escalados para clustering K-means
        """
        self.datos = datos

    def entrenar(self, k):
        """
        Entrena el modelo KMeans con k clusters
        Retorna modelo entrenado y etiquetas asignadas
        """
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiquetas = modelo.fit_predict(self.datos)
        return modelo, etiquetas

    def evaluar_varios_k(self, max_k=10):
        """
        Evalúa varios k para KMeans calculando la inercia (suma de distancias al cuadrado)
        Retorna lista de inercias para cada k desde 2 hasta max_k
        """
        inercias = []
        for k in range(2, max_k + 1):
            modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
            modelo.fit(self.datos)
            inercias.append(modelo.inertia_)
        return inercias

    def graficar_metricas(self, inercias):
        """
        Grafica el codo de Jambu para encontrar número óptimo de clusters k
        Recibe lista de inercias
        Retorna figura matplotlib
        """
        k_range = range(2, len(inercias) + 2)
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(k_range, inercias, 'bo-')
        ax.set_title('Codo de Jambú (Número óptimo de clusters)')
        ax.set_xlabel('Número de clusters (k)')
        ax.set_ylabel('Inercia')
        ax.grid(True)

        plt.tight_layout()
        return fig


class Clustering:
    def __init__(self, df):
        """
        df: dataframe original con datos sin procesar
        """
        self.df_original = df
        self.df_numerico = None
        self.datos_escalados = None

    def limpiar_y_escalar(self, columnas_seleccionadas=None):
        """
        Limpia datos eliminando filas con NA y escala variables numéricas
        columnas_seleccionadas: lista de columnas numéricas para escalar; si None usa todas las numéricas
        Retorna datos escalados (numpy array) y nombres de columnas escaladas
        """
        df = self.df_original.dropna()
        if columnas_seleccionadas:
            df_numerico = df[columnas_seleccionadas]
        else:
            df_numerico = df.select_dtypes(include=['int64', 'float64'])
        self.df_numerico = df_numerico

        if df_numerico.empty or df_numerico.shape[1] < 2:
            return None, None

        scaler = StandardScaler()
        self.datos_escalados = scaler.fit_transform(df_numerico)

        return self.datos_escalados, df_numerico.columns

    def clustering_kmeans(self, k=None, max_clusters=10):
        """
        Realiza clustering jerárquico (Ward) y asigna etiquetas a clusters
        Si k es None, calcula métricas para k desde 2 hasta max_clusters y elige k óptimo
        Retorna diccionario con linkage, k_optimo, etiquetas y listas de métricas (silhouette, calinski, davies)
        """
        linkage_matrix = linkage(self.datos_escalados, method='ward')

        if k is not None:
            etiquetas_finales = fcluster(linkage_matrix, k, criterion='maxclust')
            return {
                "linkage": linkage_matrix,
                "k_optimo": k,
                "etiquetas": etiquetas_finales,
                "silhouette": None,
                "calinski": None,
                "davies": None
            }

        sil_scores, cal_scores, db_scores = [], [], []
        for i in range(2, max_clusters + 1):
            labels = fcluster(linkage_matrix, i, criterion='maxclust')
            sil_scores.append(silhouette_score(self.datos_escalados, labels))
            cal_scores.append(calinski_harabasz_score(self.datos_escalados, labels))
            db_scores.append(davies_bouldin_score(self.datos_escalados, labels))

        k_optimo = sil_scores.index(max(sil_scores)) + 2
        etiquetas_finales = fcluster(linkage_matrix, k_optimo, criterion='maxclust')

        return {
            "linkage": linkage_matrix,
            "k_optimo": k_optimo,
            "etiquetas": etiquetas_finales,
            "silhouette": sil_scores,
            "calinski": cal_scores,
            "davies": db_scores
        }

    def graficar_clusters_kmeans(self, etiquetas):
        """
        Grafica scatter plot de clusters (primeras dos variables numéricas)
        Recibe etiquetas cluster asignadas
        Retorna figura matplotlib
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            self.df_numerico.iloc[:, 0],
            self.df_numerico.iloc[:, 1],
            c=etiquetas,
            cmap='tab10',
            alpha=0.7
        )
        ax.set_title('Clusters')
        ax.set_xlabel(self.df_numerico.columns[0])
        ax.set_ylabel(self.df_numerico.columns[1])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def graficar_radar_por_cluster(self, etiquetas, titulo="Radar por Cluster"):
        """
        Grafica gráfico radar con promedio de variables por cluster
        etiquetas: etiquetas cluster asignadas
        titulo: título para la gráfica
        Retorna figura matplotlib
        """
        df = self.df_numerico.copy()
        df["cluster"] = etiquetas
        promedios = df.groupby("cluster").mean()

        categorias = list(promedios.columns)
        N = len(categorias)
        angulos = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angulos += angulos[:1]

        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

        for i, row in promedios.iterrows():
            valores = row.tolist()
            valores += valores[:1]
            ax.plot(angulos, valores, label=f"Cluster {i}")
            ax.fill(angulos, valores, alpha=0.1)

        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias)
        ax.set_title(titulo)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()

        return fig