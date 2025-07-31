"""
Clase: EstadisticasBasicasDatos

Objetivo: Clase enfocada a la codificacion de la data para el EDA basico

1. Creacion de clase pmarin 15-07-2024
"""
import numpy as np
import pandas as pd
import io
import logging
from typing import Dict, List, Optional, Union, Any
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
from scipy import stats
import plotly.graph_objects as go
# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Configuración global para mejor apariencia
plt.style.use('seaborn-v0_8-whitegrid')  # o 'default' si no funciona
sns.set_palette("husl")

class EstadisticasBasicasDatos:
    def __init__(self, df,numericas, categoricas):
        self.df = df
        self.numericas = numericas  # Lista de columnas numéricas
        self.categoricas = categoricas  # Lista de columnas categóricas

    def obtener_info_basico(self):
        """
        Obtiene información básica del DataFrame

        Returns:
            dict: Diccionario con información básica

        Raises:
            DataFrameAnalyzerError: Sí ocurre un error al obtener la información
        """
        try:
            memoria_mb = self.df.memory_usage(deep=True).sum() / 1024 ** 2

            return {
                'filas': self.df.shape[0],
                'columnas': self.df.shape[1],
                'memoria_mb': round(memoria_mb, 2)
            }

        except MemoryError:
            logger.error("Error de memoria al calcular el uso de memoria del DataFrame")
            return {
                'filas': 0,
                'columnas': 0,
                'memoria_mb': 0
            }
        except Exception as e:
            logger.error(f"Error al obtener información básica: {str(e)}")
            return {
                'filas':0,
                'columnas': 0,
                'memoria_mb': 0
            }

    def obtener_primeras_filas(self, n: int = 5):
        """
        Obtiene las primeras n filas del DataFrame

        Args:
            n (int): Número de filas a obtener

        Returns:
            pd.DataFrame: Primeras n filas

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener las filas
        """
        try:
            if n <= 0:
                logger.error("El número de filas debe ser positivo")

            if n > len(self.df):
                logger.warning(f"Se solicitaron {n} filas pero el DataFrame solo tiene {len(self.df)} filas")
                n = len(self.df)

            return self.df.head(n)

        except Exception as e:
            logger.error(f"Error al obtener las primeras {n} filas: {str(e)}")
            return None

    def obtener_ultimas_filas(self, n: int = 5):
        """
        Obtiene las últimas n filas del DataFrame

        Args:
            n (int): Número de filas a obtener

        Returns:
            pd.DataFrame: Últimas n filas

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener las filas
        """
        try:
            if n <= 0:
                logger.error("El número de filas debe ser positivo")

            if n > len(self.df):
                logger.warning(f"Se solicitaron {n} filas pero el DataFrame solo tiene {len(self.df)} filas")
                n = len(self.df)

            return self.df.tail(n)

        except Exception as e:
            logger.error(f"Error al obtener las últimas {n} filas: {str(e)}")
            return None

    def obtener_tipo_datos(self):
        """
        Obtiene información detallada sobre tipos de datos

        Returns:
            dict: Diccionario con información de tipos de datos

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener la información
        """
        try:
            # Capturar información detallada
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            info_str = buffer.getvalue()

            # Resumen de tipos de datos
            tipos_resumen = self.df.dtypes.value_counts().to_dict()

            return {
                'info_detallada': info_str,
                'tipos_resumen': tipos_resumen
            }

        except Exception as e:
            logger.error(f"Error al obtener información de tipos de datos: {str(e)}")
            return {
                'info_detallada': "NA",
                'tipos_resumen': "NA"
            }

    def obtener_info_columnas(self):
        """
        Obtiene información detallada de las columnas

        Returns:
            pd.DataFrame: DataFrame con información de columnas

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener la información
        """
        try:
            columnas = self.df.columns.tolist()

            info_columnas_data = []

            for col in columnas:
                try:
                    valores_nulos = self.df[col].isnull().sum()
                    valores_unicos = self.df[col].nunique()
                    porcentaje_nulos = round((valores_nulos / len(self.df)) * 100, 2)

                    info_columnas_data.append({
                        'Columna': col,
                        'Tipo': str(self.df[col].dtype),
                        'Valores_nulos': valores_nulos,
                        'Valores_únicos': valores_unicos,
                        'Porcentaje_nulos': porcentaje_nulos
                    })

                except Exception as e:
                    logger.warning(f"Error al procesar columna '{col}': {str(e)}")
                    info_columnas_data.append({
                        'Columna': col,
                        'Tipo': 'Error',
                        'Valores_nulos': 'N/A',
                        'Valores_únicos': 'N/A',
                        'Porcentaje_nulos': 'N/A'
                    })

            return pd.DataFrame(info_columnas_data)

        except Exception as e:
            logger.error(f"Error al obtener información de columnas: {str(e)}")


    def obtener_nombres_columnas(self):
        """
        Obtiene la lista de nombres de columnas

        Returns:
            list: Lista con nombres de columnas

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener los nombres
        """
        try:
            return self.df.columns.tolist()

        except Exception as e:
            logger.error(f"Error al obtener nombres de columnas: {str(e)}")


    def obtener_resumen_estadistico(self):
        """
        Obtiene estadísticas resumidas del DataFrame

        Returns:
            pd. DataFrame: Estadísticas descriptivas o None si hay error

        Raises:
            DataFrameAnalyzerError: Sí ocurre un error al obtener las estadísticas
        """
        try:
            return self.df.describe(include='all')

        except Exception as e:
            logger.error(f"Error al obtener estadísticas descriptivas: {str(e)}")


    def obetener_resumen_valores_faltantes(self):
        """
        Obtiene resumen de valores faltantes

        Returns:
            pd.DataFrame: Resumen de valores faltantes

        Raises:
            DataFrameAnalyzerError: Si ocurre un error al obtener el resumen
        """
        try:
            missing_data_list = []

            for col in self.df.columns:
                try:
                    valores_faltantes = self.df[col].isnull().sum()
                    porcentaje = round((valores_faltantes / len(self.df)) * 100, 2)

                    missing_data_list.append({
                        'Columna': col,
                        'Valores_faltantes': valores_faltantes,
                        'Porcentaje': porcentaje
                    })

                except Exception as e:
                    logger.warning(f"Error al procesar valores faltantes en columna '{col}': {str(e)}")
                    missing_data_list.append({
                        'Columna': col,
                        'Valores_faltantes': 'Error',
                        'Porcentaje': 'Error'
                    })

            missing_df = pd.DataFrame(missing_data_list)
            return missing_df.sort_values('Valores_faltantes', ascending=False)

        except Exception as e:
            logger.error(f"Error al obtener resumen de valores faltantes: {str(e)}")


    def obtener_analisis_completo(self):
        """
        Obtiene análisis completo del DataFrame

        Returns:
            dict: Diccionario con todo el análisis

        Raises:
            DataFrameAnalyzerError: Si ocurre un error crítico
        """
        analysis = {}
        errors = []

        # Ejecutar cada análisis y capturar errores
        analysis_methods = [
            ('info_basica', self.obtener_info_basico()),
            ('primeras_filas', lambda: self.obtener_primeras_filas()),
            ('ultimas_filas', lambda: self.obtener_ultimas_filas()),
            ('tipos_datos', self.obtener_tipo_datos()),
            ('info_columnas', self.obtener_info_columnas()),
            ('nombres_columnas', self.obtener_nombres_columnas()),
            ('estadisticas', self.obtener_resumen_estadistico()),
            ('valores_faltantes', self.obetener_resumen_valores_faltantes())
        ]

        for key, method in analysis_methods:
            try:
                analysis[key] = method()
            except Exception as e:
                logger.error(f"Error en análisis '{key}': {str(e)}")
                errors.append(f"{key}: {str(e)}")
                analysis[key] = None

        # Agregar información de errores si los hay
        if errors:
            analysis['errores'] = errors
            logger.warning(f"Análisis completado con {len(errors)} errores")

        return analysis

    def obtener_analisis_distribucion(self,col):
        # fig, ax = plt.subplots(figsize=(8, 6))
        #
        # sns.histplot(data=self.df, x=col, kde=True, ax=ax)
        # ax.set_title(f'Distribución de {col}')
        #
        # return fig

        """Histograma mejorado que funciona tanto para datos numéricos como categóricos"""
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        # Detectar si es numérico o categórico
        if pd.api.types.is_numeric_dtype(self.df[col]):
            # DATOS NUMÉRICOS
            sns.histplot(data=self.df, x=col, kde=True, ax=ax,
                         alpha=0.7, edgecolor='black', linewidth=0.5)

            # Líneas de estadísticas verticales
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()

            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8,
                       label=f'Media: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', alpha=0.8,
                       label=f'Mediana: {median_val:.2f}')

            # Estadísticas para numéricos
            stats_text = f'N: {len(self.df[col].dropna())}\nStd: {self.df[col].std():.2f}\nMin: {self.df[col].min():.2f}\nMax: {self.df[col].max():.2f}'
            ax.set_ylabel('Frecuencia', fontsize=12)
            ax.grid(True, alpha=0.3)

        else:
            # DATOS CATEGÓRICOS
            value_counts = self.df[col].value_counts()

            # Decidir orientación basada en la longitud de las etiquetas
            max_label_length = max(len(str(label)) for label in value_counts.index)

            if max_label_length > 15 or len(value_counts) > 10:
                # HORIZONTAL para etiquetas largas o muchas categorías
                bars = ax.barh(range(len(value_counts)), value_counts.values,
                               alpha=0.7, edgecolor='black', linewidth=0.5,
                               color=plt.cm.Set3(np.linspace(0, 1, len(value_counts))))

                # Líneas de estadísticas verticales (para gráfico horizontal)
                mean_val = value_counts.mean()
                median_val = value_counts.median()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8,
                           label=f'Media frecuencias: {mean_val:.1f}')
                ax.axvline(median_val, color='green', linestyle='--', alpha=0.8,
                           label=f'Mediana frecuencias: {median_val:.1f}')

                # Configurar etiquetas
                ax.set_yticks(range(len(value_counts)))
                ax.set_yticklabels(value_counts.index)
                ax.set_xlabel('Frecuencia', fontsize=12)
                ax.grid(True, alpha=0.3, axis='x')

                # Valores en las barras
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2.,
                            f'{int(width)}', ha='left', va='center', fontweight='bold')

            else:
                # VERTICAL para etiquetas cortas
                bars = ax.bar(range(len(value_counts)), value_counts.values,
                              alpha=0.7, edgecolor='black', linewidth=0.5,
                              color=plt.cm.Set3(np.linspace(0, 1, len(value_counts))))

                # Líneas de estadísticas horizontales
                mean_val = value_counts.mean()
                median_val = value_counts.median()
                ax.axhline(mean_val, color='red', linestyle='--', alpha=0.8,
                           label=f'Media frecuencias: {mean_val:.1f}')
                ax.axhline(median_val, color='green', linestyle='--', alpha=0.8,
                           label=f'Mediana frecuencias: {median_val:.1f}')

                # Configurar etiquetas
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.set_ylabel('Frecuencia', fontsize=12)
                ax.grid(True, alpha=0.3, axis='y')

                # Valores en las barras
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

            # Estadísticas para categóricos
            total_count = len(self.df[col].dropna())
            unique_count = len(value_counts)
            most_frequent = value_counts.index[0]
            most_frequent_pct = (value_counts.iloc[0] / total_count) * 100

            stats_text = f'Total: {total_count}\nCategorías: {unique_count}\nMás frecuente: {most_frequent}\n({most_frequent_pct:.1f}%)'

        # Configuración común
        ax.set_title(f'Distribución de {col}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(col, fontsize=12)
        ax.legend(loc='best')

        # Añadir caja de estadísticas
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return fig



    def obtener_analisis_boxplot(self,col):
        # Se genera el boxplot con Plotly
        fig = px.box(self.df, y=col, title=f"Boxplot de {col}")
        return fig


    def obtener_analisis_correlaccion(self):
        # corr = self.df[self.numericas].corr()
        # fig = plt.figure(figsize=(12, 8))
        # mask = np.triu(np.ones_like(corr, dtype=bool))
        # sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu_r', center=0, mask=mask, square=True)
        # plt.title("Matriz de Correlación")
        # plt.tight_layout()
        #
        # return fig
        """Matriz de correlación mejorada"""
        corr = self.df[self.numericas].corr()

        # Crear máscara para triángulo superior
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

        # Crear heatmap con mejor paleta de colores
        sns.heatmap(corr,
                    annot=True,
                    fmt=".2f",
                    cmap='RdBu_r',
                    center=0,
                    mask=mask,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": .8},
                    annot_kws={'size': 10})

        ax.set_title("Matriz de Correlación", fontsize=18, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        return fig

    def _grafico_categorico(self,col):
        # value_counts = self.df[col].value_counts()
        #
        # fig=plt.figure(figsize=(8, 8))
        # plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        # plt.title(f'Distribución de {col}')
        #
        # return fig

        """Gráfico de pie mejorado con opciones adicionales"""
        value_counts = self.df[col].value_counts()

        # Crear figura con subplots para pie + barra
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

        # Gráfico de pie mejorado
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        wedges, texts, autotexts = ax1.pie(value_counts.values,
                                           labels=value_counts.index,
                                           autopct='%1.1f%%',
                                           colors=colors,
                                           explode=[0.05 if i == 0 else 0 for i in range(len(value_counts))],
                                           shadow=True,
                                           startangle=90)

        # Mejorar texto del pie
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax1.set_title(f'Distribución de {col}', fontsize=14, fontweight='bold')

        # Gráfico de barras complementario
        bars = ax2.bar(range(len(value_counts)), value_counts.values,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Categorías', fontsize=12)
        ax2.set_ylabel('Frecuencia', fontsize=12)
        ax2.set_title(f'Frecuencia de {col}', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(value_counts)))
        ax2.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        return fig

    def _grafico_qplot(self,col):

        # fig = plt.figure(figsize=(8, 8))
        #
        # stats.probplot(self.df[col].dropna(), dist="norm", plot=plt)
        # plt.title(f'Q-Q Plot de {col}')
        # plt.grid(True, alpha=0.3)
        # return fig

        """Q-Q Plot mejorado con mejor visualización"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

        # Q-Q Plot
        stats.probplot(self.df[col].dropna(), dist="norm", plot=ax1)
        ax1.set_title(f'Q-Q Plot de {col}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Cuantiles Teóricos', fontsize=12)
        ax1.set_ylabel('Cuantiles de la Muestra', fontsize=12)

        # Añadir línea de referencia más visible
        ax1.get_lines()[0].set_markerfacecolor('lightblue')
        ax1.get_lines()[0].set_markeredgecolor('darkblue')
        ax1.get_lines()[0].set_markersize(4)
        ax1.get_lines()[1].set_color('red')
        ax1.get_lines()[1].set_linewidth(2)

        # Histograma complementario para comparar con distribución normal
        data_clean = self.df[col].dropna()
        ax2.hist(data_clean, bins=30, density=True, alpha=0.7,
                 color='lightblue', edgecolor='black', label='Datos')

        # Superponer distribución normal teórica
        x = np.linspace(data_clean.min(), data_clean.max(), 100)
        normal_curve = stats.norm.pdf(x, data_clean.mean(), data_clean.std())
        ax2.plot(x, normal_curve, 'r-', linewidth=2, label='Normal Teórica')

        ax2.set_title(f'Comparación con Distribución Normal', fontsize=14, fontweight='bold')
        ax2.set_xlabel(col, fontsize=12)
        ax2.set_ylabel('Densidad', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Añadir test de normalidad
        shapiro_stat, shapiro_p = stats.shapiro(data_clean)
        ax2.text(0.02, 0.98, f'Shapiro-Wilk p-value: {shapiro_p:.4f}',
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return fig

    def obtener_analisis_univariados(self, col):
        if col in self.numericas:
            return self._grafico_qplot(col)
        elif col in self.categoricas:
            return self._grafico_categorico(col)
        return None



