"""
Clase: PaginaAcerca

Objetivo: Clase que mantiene la pagina para mostrar los datos del app e integrantes

Cambios:

    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
"""
import streamlit as st

class PaginaAcerca:
    def __init__(self):
        pass
    def render(self):
        st.title("ℹ️ Acerca de esta aplicación")

        st.markdown("""
        🧠 Esta aplicación es una demostración del trabajo realizado por estudiantes del curso **BD-152 Mineria de datos I** del  
        🎓 **Diplomado en Big Data** del **Colegio Universitario de Cartago (CUC)** durante el **II Cuatrimestre 2025**,  
        como parte de su aprendizaje y desarrollo de habilidades en el campo del análisis de datos a gran escala.
        """)

        st.markdown("### 👩‍💻 Autores:")
        st.markdown("""
        - 👩‍🎓 Nubia Brenes Valerín  
        - 👨‍🎓 Pablo Marín Castillo
        - 👨‍🎓 Alejandro 
        - 👨‍🎓 Fiorella 
        - 👨‍🎓 Gilary 
        - 👨‍🎓 Fernando 
        - 👨‍🎓 Jhoel
        """)

        st.markdown("### 🛠️ Herramientas y Tecnologías Utilizadas:")
        st.markdown("""
        - 🐍 **Lenguaje de Programación:** Python  
        - 📊 **Librerías de Análisis de Datos:** Pandas, NumPy  
        - 📈 **Visualización de Datos:** Matplotlib, Seaborn, Plotly  
        - 🚀 **Framework/Entorno:** Streamlit  
        - 🗄️ **Base de Datos:** PostgreSQL  
        - 💻 **Otras Herramientas:** Jupyter Notebooks, Microsoft Azure
        """)