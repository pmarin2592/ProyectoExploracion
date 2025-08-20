"""
Clase: PaginaAcerca

Objetivo: Clase que mantiene la pagina para mostrar los datos del app e integrantes

Cambios:

    1. Creacion de la clase y cascarazon visual pmarin 24-06-2025
    2. Colocar info en dos columnas para aprovechar espacio en pantalla aquesada 28-06-2025
"""
import streamlit as st

class PaginaAcerca:
    def __init__(self):
        pass
    def render(self):
        st.markdown("""
                                              <h1 style='
                                                  text-align: center;
                                                  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                                                  -webkit-background-clip: text;
                                                  -webkit-text-fill-color: transparent;
                                                  font-size: 3rem;
                                                  margin-bottom: 2rem;
                                                  font-weight: bold;
                                              '>
                                                  Acerca de esta aplicación
                                              </h1>
                                              """, unsafe_allow_html=True)

        st.markdown("""
         Esta aplicación es una demostración del trabajo realizado por estudiantes del curso **BD-152 Mineria de datos I** del  
         **Diplomado en Big Data** del **Colegio Universitario de Cartago (CUC)** durante el **II Cuatrimestre 2025**,  
        como parte de su aprendizaje y desarrollo de habilidades en el campo del análisis de datos a gran escala.
        """)

        # Crear dos columnas para mostrar Autores y Herramientas lado a lado
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👩‍💻 Autores:")
            st.markdown("""
                   - Nubia Brenes Valerín  
                   - Pablo Marín Castillo
                   - Alejandro Quesada Leiva
                   - Fiorella Abarca Valverde
                   - Gilary Granados Calvo
                   - Fernando Contreras Artavia
                   - Johel Barquero Carvajal
                   """)

        with col2:
            st.markdown("### 🛠️ Herramientas y Tecnologías Utilizadas:")
            st.markdown("""
                   - 🐍 **Lenguaje de Programación:** Python  
                   - 📊 **Librerías de Análisis de Datos:** Pandas, NumPy  
                   - 📈 **Visualización de Datos:** Matplotlib, Seaborn, Plotly  
                   - 🚀 **Framework/Entorno:** Streamlit   
                   - 💻 **Otras Herramientas:** Jupyter Notebooks, Microsoft Azure
                   """)