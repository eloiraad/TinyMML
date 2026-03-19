# RoadMap du Projet CV-MML

Ce document trace la progression séquentielle du projet. Chaque étape doit être validée avant de passer à la suivante.

---

### Phase 1 : Core Engine (Tenseurs, CPU & Autograd)

**Objectif :** Créer une base solide pour la manipulation de données multidimensionnelles et le calcul de gradients.

*   **1.1 : Initialisation**
    *   [x] Création de la structure complète : `includes/cvmml/core`, `sources/core`, `sources/api`, `python/`, `tests/`, `data/`.
    *   [x] Initialisation du fichier `.gitignore`.
    *   [x] `CMakeLists.txt` de base pour le projet.
*   **1.2 : Classe `Tensor`**
    *   [x] Gestion de la mémoire CPU, shape et strides.
    *   [x] Support des types de données (float par défaut).
*   **1.3 : Opérations Fondamentales**
    *   [x] Opérations arithmétiques élément par élément.
    *   [x] Multiplication matricielle (GEMM) naïve.
    *   [x] Tests unitaires C++.
*   **1.4 : Autograd (Graphe de Calcul)**
    *   [x] Implémentation du suivi des opérations (graphe acyclique dirigé).
    *   [x] Propagation arrière (Backpropagation) pour le calcul automatique des gradients.

---

### Phase 2 : Python API (Bridge)

**Objectif :** Rendre le moteur utilisable dans l'écosystème Python.

*   **2.1 : Bindings Pybind11**
    *   [x] Export de la classe `Tensor` et de l'Autograd vers Python dans `sources/api/bindings.cpp`.
*   **2.2 : Interopérabilité NumPy**
    *   [x] Implémentation du protocole buffer (Zero-copy) pour une conversion fluide NumPy <-> Tensor.
*   **2.3 : Validation**
    *   [x] Scripts de test d'intégrité Python/C++.

---

### Phase 3 : Accélération CUDA

**Objectif :** Déporter les calculs lourds sur le GPU.

*   **3.1 : Gestion Mémoire Device**
    *   [x] Allocation VRAM et transferts Host <-> Device.
*   **3.2 : Kernels de Calcul**
    *   [x] Implémentation CUDA pour les opérations de base et GEMM (`.cu` dans `sources/core`).
*   **3.3 : Intégration API**
    *   [x] Méthodes `.to_cuda()` et `.to_cpu()` exposées en Python.

- [x] Enlever synchro systematique
- [x] Ajouter multithreading cpu (OpenMP appliqué sur reduce, matmult, im2col, transpose backward)
- [x] revoir transposition (backward optimisé)

---

### Phase 4 : Neural Network Framework (`nn`)

**Objectif :** Construire des modèles de Deep Learning avec une API haut niveau type Keras.

*   **4.0 : Prérequis Tensor (bloquant)**
    *   [x] Ajouter les ops nécessaires: `exp`, `log`, `sum(axis, keepdim)`, `max(axis, keepdim)`, `min(axis, keepdim)`, broadcast fiable.
    *   [x] Vérifier l’autograd de ces ops (CPU/CUDA).

*   **4.1 : Abstractions de base**
    *   [x] Classe abstraite `Layer` (`forward`, `parameters`, `train`, `eval`).
    *   [x] Classe `Model` avec liste de `Layer`, `forward`, `__call__`, `parameters`.
    *   [x] Build du modèle: `input_shape` explicite **ou** inférence au premier batch.

*   **4.2 : Layers**
    *   [x] `Linear` (poids, biais, init Xavier/He simple).
    *   [x] `ReLU`.
    *   [x] `Softmax` (version stable numériquement).
    *   [x] `Dropout` (masque aléatoire, gestion du flag `training`).
    *   [x] `Conv2D` (MVP: stride=1, padding=0, dilation=1, groups=1).
    *   [x] `BatchNorm` & `LayerNorm`
    *   [x] Patterns avancés : Skip Connections (`ResidualBlock` utilisant l'opérateur tensoriel `+`).

*   **4.3 : Losses**
    *   [x] `CrossEntropyLoss` (fusion Softmax + NLL recommandée pour classification).
    *   [ ] `MSELoss` (Mean Squared Error, utile pour la régression et les tests simples).
    *   [x] Interface `LossFunction` générique.

*   **4.4 : Optimizers**
    *   [x] `SGD` complet (lr, weight decay).
    *   [ ] Ajouter le paramètre **Momentum** à l'optimiseur SGD.
    *   [ ] `Adam` (beta1, beta2, eps).

*   **4.5 : Entraînement & Évaluation**
    *   [ ] `model.train(...)` (epochs, batch_size, shuffle, lr).
    *   [ ] `model.evaluate(...)` (loss moyenne, accuracy).
    *   [ ] Affichage de la loss par epoch.
    *   [ ] Matrice de confusion (TP/TN/FP/FN + multi-classes).

*   **4.6 : Validation**
    *   [ ] Pipeline complet sur MNIST (Linear + ReLU + Softmax + CrossEntropy).
    *   [ ] Test de non-régression CPU/CUDA.

---

### Phase 5 : Computer Vision & Modèles Avancés (`cv`)

**Objectif :** Traitement d'image et modèles génératifs.

*   **5.1 : Ops de Vision**
    *   [ ] Grayscale, Resizing, Normalisation en C++.
*   **5.2 : Convolutional Networks**
    *   [ ] Entraînement sur CIFAR-10.
*   **5.3 : Pipeline Génératif**
    *   [ ] Super-résolution ou modèles de diffusion simplifiés.

---

### Notes de structure (refactor global)

*   [x] Split `sources/core/tensor.cpp` en modules `tensor/` par responsabilité.
*   [x] Split `sources/core/cuda_kernels.cu` en modules `cuda/` par responsabilité.
*   [x] Conserver API publique `Tensor` stable tout en ajoutant `min`.
*   [x] Reporter explicitement le multithreading CPU (décision produit) en attente de profiling.
