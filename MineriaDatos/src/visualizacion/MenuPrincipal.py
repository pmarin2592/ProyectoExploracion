"""
Clase: MenuPrincipal

Objetivo: Clase que mantiene el menu principal del proyecto

Cambios:

    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
"""
import os
import streamlit as st
from PIL import Image

from src.visualizacion.PaginaACP import PaginaACP
from src.visualizacion.PaginaAFC import PaginaAFC
from src.visualizacion.PaginaAcerca import PaginaAcerca
from src.visualizacion.PaginaCluster import PaginaCluster
from src.visualizacion.PaginaDatos import PaginaDatos
from src.visualizacion.PaginaEstadisticas import PaginaEstadisticas
from src.visualizacion.PaginaKmeans import PaginaKmeans


class MenuPrincipal:
    def __init__(self, base_dir):
        self.__menu ={
                    "🗂️ Datos":PaginaDatos(),                               # Representa archivos/datasets
                    "📈 Estadísticas Básicas": PaginaEstadisticas(),        # Para resúmenes y gráficos básicos
                    "🧮 ACP (Componentes Principales)": PaginaACP(),        # Matemática / reducción dimensional
                    "🔠 AFC (Correspondencias)": PaginaAFC(),               # Relacionado con datos categóricos
                    "🧬 Clúster Jerárquico": PaginaCluster(),               # Jerarquía y análisis agrupado
                    "🔢 K-Means": PaginaKmeans(),                           # Algoritmo de clustering
                    "💡 Acerca de": PaginaAcerca()                          # Información general / autores / contexto
                }
        self.__base_dir = base_dir

    def menu_principal(self):
        # Abre la imagen desde la misma carpeta que main.py
        logo_path = os.path.join(self.__base_dir, "logo-cuc.png")
        logo = Image.open(logo_path)

        st.set_page_config(page_title="Aplicativo de Mineria", layout="wide", page_icon="📊")
        st.sidebar.image(logo, width=120)

        st.sidebar.title("Menu Principal")
        opcion = st.sidebar.radio("Ir a:", self.__menu.keys())
        pagina = self.__menu[opcion]
        pagina.render()
        st.sidebar.markdown(
            """
            <hr style='margin-top: 50px;'>
            <div style='font-size: 0.8em; text-align: center;'>
                © 2025 | Proyecto Final Programación II<br>
                Big Data CUC
            </div>
            """,
            unsafe_allow_html=True
        )
