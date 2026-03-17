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
    *   [ ] Allocation VRAM et transferts Host <-> Device.
*   **3.2 : Kernels de Calcul**
    *   [ ] Implémentation CUDA pour les opérations de base et GEMM (`.cu` dans `sources/core`).
*   **3.3 : Intégration API**
    *   [ ] Méthodes `.to_cuda()` et `.to_cpu()` exposées en Python.

---

### Phase 4 : Neural Network Framework (`nn`)

**Objectif :** Construire des modèles de Deep Learning.

*   **4.1 : Architecture Layers**
    *   [ ] Classe de base `Layer`, `Linear`, `Conv2D`.
*   **4.2 : Activations & Pertes**
    *   [ ] `ReLU`, `Softmax`, `CrossEntropy`.
*   **4.3 : Optimiseurs**
    *   [ ] `SGD`, `Adam` (optionnel au début).
*   **4.4 : Application MNIST**
    *   [ ] Entraînement complet d'un MLP sur MNIST en Python.

---

### Phase 5 : Computer Vision & Modèles Avancés (`cv`)

**Objectif :** Traitement d'image et modèles génératifs.

*   **5.1 : Ops de Vision**
    *   [ ] Grayscale, Resizing, Normalisation en C++.
*   **5.2 : Convolutional Networks**
    *   [ ] Entraînement sur CIFAR-10.
*   **5.3 : Pipeline Génératif**
    *   [ ] Super-résolution ou modèles de diffusion simplifiés.
