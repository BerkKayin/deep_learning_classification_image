# Projet de Reconnaissance d'Objets avec CIFAR-10 en utilisant PyTorch

### Objectif du Projet

Le but de ce projet est de développer et d'évaluer des modèles de deep learning pour la reconnaissance et la classification d'objets en utilisant l'ensemble de données CIFAR-10. 
CIFAR-10 est un ensemble de données bien connu composé de 60 000 images couleur de 32x32 pixels réparties en 10 classes (comme les avions, les voitures, les oiseaux, etc.). 
Les modèles à implémenter incluent la régression logistique, les réseaux de neurones convolutifs (CNN), et une architecture avancée comme VGG2.

## Étapes du Projet

### Préparation des Données
Charger l'ensemble de données CIFAR-10 en utilisant PyTorch / Keras.
Prétraitement des données : normalisation, augmentation des données (optionnelle).
Divisions des données en deux ensembles d'entraînement et de test.

### Modèle de Régression Logistique
Implémentation d'une régression logistique comme point de départ simple pour la classification.
Formation du modèle sur les données d'entraînement et évaluer la précision sur les données de test.

### Réseaux de Neurones Convolutifs (CNN)
Construction d'un CNN simple avec quelques couches de convolution (2 max) et de pooling.
Formation du CNN sur les données d'entraînement.
Évaluation de la performance du CNN sur les données de test et comparer avec la régression logistique.

### Architecture VGG2
Implémentation ou utilisation d'une version pré-implémentée de l'architecture VGG (par exemple VGG-16 ou une version simplifiée adaptée aux images 32x32).
Formation du modèle VGG sur les données d'entraînement.
Évaluation de la performance du modèle VGG sur les données de test.

### Évaluation et Comparaison
Comparaison les performances des différents modèles (régression logistique, CNN, VGG2) en termes de précision, de courbe ROC, de matrice de confusion, etc.
Analyse des erreurs de classification pour identifier les classes d'objets les plus difficiles à prédire.

# FIN
