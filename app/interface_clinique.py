import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api import ClinicalPredictor
from core.dataset import Dataset
from core.logisticRegression import LogisticRegression
from core.neuralNetwork import NeuralNetwork
from core.optimizer import GradientDescent
from pipeline.evaluator import Evaluator
from pipeline.trainer import Trainer

# Patients à diagnostiquer : [température, toux (0-10), oxygène (%)]
PATIENTS = {
    "Patient A": [36.7, 1, 98],
    "Patient B": [39.0, 8, 87],
}


def main():
    data = Dataset("data/patient_data.csv")
    X_train, y_train, X_test, y_test = data.split()

    # Le même ClinicalPredictor fonctionne avec n'importe quel modèle
    for model in (LogisticRegression(), NeuralNetwork()):
        name = type(model).__name__
        trained = Trainer(model, GradientDescent(lr=0.1)).train(X_train, y_train)
        print(f"\n=== {name} ===")
        print("Évaluation :", Evaluator().evaluate(trained, X_test, y_test))

        predictor = ClinicalPredictor(trained)
        for patient, values in PATIENTS.items():
            print(f"{patient} {values} -> {predictor.diagnose(values)}")


if __name__ == "__main__":
    main()
