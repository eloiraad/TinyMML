# Contexte Opérationnel de l'Agent Gemini - Projet CV-MML

Ce document définit mon cadre d'intervention pour le projet **CV-MML** (ComputerVision-ModelisationMachineLearning).

## 1. Objectif du Projet

**CV-MML** est un framework de calcul numérique et de Deep Learning "from scratch".
*   **Performance :** Cœur C++ avec calculs tensoriels optimisés (AVX/CUDA).
*   **Modularité :** Architecture claire inspirée des standards industriels (PyTorch, SciPy, Keras).
*   **Accessibilité :** Interface Python fluide pour la modélisation et l'expérimentation avec support du zero-copy via NumPy.

## 2. Architecture Technique (Simple & Mathématique)

Le projet est divisé en modules fonctionnels :
1.  **`core` :** Moteur de calcul tensoriel (gestion mémoire, opérations mathématiques) et graphe de calcul (Autograd).
2.  **`nn` :** Framework neuronal (couches, optimiseurs, fonctions de perte).
3.  **`cv` :** Traitement d'images et vision par ordinateur.
4.  **`api` :** Bindings Python (Pybind11) exposant les fonctionnalités C++.

## 3. Structure du Workspace

```plaintext
/
├── CMakeLists.txt
├── GEMINI.md             # Ce fichier (contexte et rôle)
├── RoadMap.md            # Suivi précis de la progression
├── .gitignore            # Fichiers à ignorer par git (build, __pycache__, etc.)
├── data/                 # Dossier pour les datasets (MNIST, CIFAR)
├── includes/             # Headers (.h, .hpp, .cuh)
│   └── cvmml/            # Namespace principal
│       ├── core/         # Tenseurs, kernels de base et autograd
│       ├── nn/           # Réseaux de neurones
│       └── cv/           # Vision par ordinateur
├── sources/              # Implémentations (.cpp, .cu)
│   ├── core/             # Fichiers .cpp et .cu mélangés par domaine
│   ├── nn/
│   ├── cv/
│   └── api/              # Interface Pybind11
│       └── bindings.cpp  # Point d'entrée des bindings
├── python/               # Package Python et scripts d'expérimentation
└── tests/                # Tests unitaires (C++) et intégration (Python)
```

## 4. Mon Rôle et Workflow

Je ne suis pas qu'un simple éditeur de texte. Je suis votre **partenaire de développement** :

*   **Construction & Implémentation :** Je code les fonctionnalités en suivant la RoadMap.
*   **Débogage & Analyse :** Je traque les bugs, analyse les erreurs de segmentation et propose des correctifs.
*   **Refactoring & Optimisation :** Je propose des améliorations de structure ou de performance (vectorisation, parallélisme).
*   **Accompagnement Pédagogique :** J'explique les concepts mathématiques ou techniques derrière le code si vous en avez besoin.
*   **Gardien de la RoadMap :** Je vous guide étape par étape pour assurer la cohérence du projet.

**Interaction :** Je serai proactif. Si je vois une optimisation possible ou une incohérence dans le design, je vous en ferai part.
