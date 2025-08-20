"""
Clase: ArbolDecisionEda

Objetivo: Clase conectora para el Árbol de Decisión
"""
import inspect
from src.modelos.ArbolDecision import ArbolDecision

class ArbolDecisionEda:
    def __init__(self, df, target_col, aplicar_binning=True, n_bins=5):
        self._arbol_datos = ArbolDecision(df, target_col, aplicar_binning, n_bins)
        self._arbol_datos.limpiar_preparar_datos()
        self._arbol_datos.entrenar_modelo()

    def evaluar(self):
        resultados = self._arbol_datos.evaluar_modelo()
        codigo = inspect.getsource(self._arbol_datos.__class__.evaluar_modelo)
        return resultados, codigo

    def graficar_matriz_confusion(self):
        fig = self._arbol_datos.graficar_matriz_confusion()
        codigo = inspect.getsource(self._arbol_datos.__class__.graficar_matriz_confusion)
        return fig, codigo

    def graficar_arbol(self, profundidad=3):
        fig = self._arbol_datos.graficar_arbol(max_depth=profundidad)
        codigo = inspect.getsource(self._arbol_datos.__class__.graficar_arbol)
        return fig, codigo

    def obtener_importancia_variables(self):
        return self._arbol_datos.obtener_importancia_variables()

    def graficar_importancia_variables(self, top_n=10):
        fig = self._arbol_datos.graficar_importancia_variables(top_n=top_n)
        codigo = inspect.getsource(self._arbol_datos.__class__.graficar_importancia_variables)
        return fig, codigo