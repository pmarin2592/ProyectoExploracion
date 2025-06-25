"""
Clase: PaginaDatos

Objetivo: Clase que mantiene la pagina de la carga de datos

Cambios:

    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
"""

import streamlit as st
import pandas as pd
import time

class PaginaDatos:
    def __init__(self):
        self.archivo_cargado = None
        self.analisis_realizado = False

    def render(self):
        try:
            st.title("🗃️ Carga de Datos")
            col1, col2 = st.columns([1, 2])
            with col1:
                delimitador = st.selectbox(
                    "Selecciona el delimitador de tu archivo CSV:",
                    options=[",", ";", "."],
                    format_func=lambda x: f"{x}"
                )
            with col2:
                archivo = st.file_uploader(
                    "Sube tu archivo (.csv o .xlsx)",
                    type=["csv", "xlsx"]
                )

            if archivo:
                if self.archivo_cargado is not None:
                    st.warning("Ya has cargado un archivo. Reinicia la app para cargar otro.")
                    return

                try:
                    nombre = archivo.name.lower()

                    if nombre.endswith('.csv'):
                        df = pd.read_csv(archivo, delimiter=delimitador)
                    elif nombre.endswith('.xlsx'):
                        df = pd.read_excel(archivo, engine='openpyxl')
                    else:
                        st.error("Solo se permiten archivos .csv o .xlsx.")
                        return

                    self.archivo_cargado = df
                    st.success("Archivo cargado exitosamente.")
                    st.subheader("Vista previa del dataset")
                    st.dataframe(df)
                    col3, col4, col5 = st.columns([3, 2, 3])
                    # Mostrar botón para generar análisis
                    with col4:
                        if st.button("📊 Generar análisis"):
                            progreso = st.progress(0, text="Iniciando análisis...")

                            for i in range(1, 101):
                                time.sleep(0.02)  # Simula tiempo de proceso
                                progreso.progress(i, text=f"Analizando datos... {i}%")

                            st.success("✅ Análisis completado exitosamente.")
                            self.analisis_realizado = True

                except Exception as e:
                    st.error(f"Error al cargar el archivo: {e}")
            else:
                st.info("Por favor sube un archivo CSV o Excel (.xlsx).")

        except Exception as e:
            st.title("🗃️ Carga de Datos")

