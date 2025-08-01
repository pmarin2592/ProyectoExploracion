"""
Clase: ArbolDecisionEda

Objetivo: Clase conectora para el Árbol de Decisión

"""

from src.modelos.ArbolDecision import ArbolDecision

class ArbolDecisionEda:
    def __init__(self, eda, df, target_col, aplicar_binning=True, n_bins=5):
        self._eda = eda
        self._arbol_datos = ArbolDecision(self._eda.eda, df, target_col, aplicar_binning, n_bins)
        self._arbol_datos.limpiar_preparar_datos()
        self._arbol_datos.entrenar_modelo()

    def evaluar(self):
        return self._arbol_datos.evaluar_modelo()

    def graficar_matriz_confusion(self):
        return self._arbol_datos.graficar_matriz_confusion()

    def graficar_arbol(self, profundidad=3):
        return self._arbol_datos.graficar_arbol(max_depth=profundidad)

    def obtener_importancia_variables(self):
        return self._arbol_datos.obtener_importancia_variables()

    def graficar_importancia_variables(self, top_n=10):
        return self._arbol_datos.graficar_importancia_variables(top_n=top_n)