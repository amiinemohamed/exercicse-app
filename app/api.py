import numpy as np


class ClinicalPredictor:
    """Utilise un modèle DÉJÀ entraîné pour poser un diagnostic.

    Elle ne connaît pas le type du modèle : il suffit qu'il ait predict().
    """

    def __init__(self, model):
        self.model = model

    def diagnose(self, patient_data):
        x = np.asarray(patient_data, dtype=float).reshape(1, -1)
        prediction = float(self.model.predict(x)[0])
        return "Infecté" if prediction >= 0.5 else "Sain"
