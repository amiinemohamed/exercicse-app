### Quel concept de Programmation Orientée Objet permet à ClinicalPredictor de fonctionner avec n’importe quel modèle IA, tant qu’il dispose d'une méthode predict() ?

le polymorphisme (en Python, le duck typing). ClinicalPredictor appelle seulement model.predict(), sans connaître le type du modèle. Tous les modèles partagent la même interface (classe de base Model), donc LogisticRegression et NeuralNetwork fonctionnent sans changer le code du prédicteur.

### Expliquez pourquoi il est important de séparer la prédiction (ClinicalPredictor) de l’entraînement du modèle (Trainer) dans une architecture IA.

Pour la séparation des responsabilités.

- L’entraînement est long et ponctuel, la prédiction est rapide et répétée pour chaque patient.
- Le prédicteur ne peut pas modifier ou ré-entraîner le modèle par erreur, ce qui garde les diagnostics stables.
- Il n’a besoin ni des données d’entraînement, ni de l’optimiseur.
- On peut changer d’algorithme ou ré-entraîner sans toucher à l’application clinique.
- Chaque classe se teste séparément.