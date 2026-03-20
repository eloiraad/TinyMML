from __future__ import annotations
import math
import sys
import numpy as np
import cvmml_api as cvmml
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from .layer import Layer


class Model(Layer):
	"""
	[brief] Conteneur séquentiel Keras-style pour structurer un modèle de Deep Learning.

	[details]
	Regroupe les couches, propage les données séquentiellement, et fournit les méthodes
	haut-niveau `fit`, `evaluate`, `predict` calquées sur l'API Keras.
	Le paramètre `device` permet de placer globalement tous les calculs sur CPU ou GPU.

	Args:
		layers (Optional[Iterable[Layer]]): Liste ordonnée de couches.
		device (str): Device global, "cpu" ou "cuda" (défaut "cpu").
	"""
	def __init__(self, layers: Optional[Iterable[Layer]] = None, device: str = "cpu") -> None:
		super().__init__()
		self.layers: List[Layer] = list(layers) if layers is not None else []
		self.device = device.lower()

	# ─── Layer protocol ──────────────────────────────────────────

	def add(self, layer: Layer) -> None:
		self.layers.append(layer)
		self._built = False

	def _build(self, input_shape):
		current_shape = list(input_shape) if input_shape is not None else None
		for layer in self.layers:
			if not layer.built:
				current_shape = layer.build(current_shape)
			else:
				current_shape = layer.output_shape
		return current_shape

	def forward(self, x):
		out = x
		for layer in self.layers:
			out = layer(out)
		return out

	def __call__(self, x):
		if not self.built:
			inferred_shape = x.shape() if hasattr(x, "shape") else None
			self.build(inferred_shape)
		return self.forward(x)

	def parameters(self) -> List:
		params: List = []
		for layer in self.layers:
			params.extend(layer.parameters())
		return params

	def train(self) -> "Model":
		"""
		[brief] Active le mode entraînement sur le modèle et toutes ses couches.

		[details]
		Toggle utilisé par Dropout, BatchNorm, etc. pour distinguer train vs inférence.
		Ne lance PAS l'entraînement — utiliser `fit()` pour cela.
		"""
		super().train()
		for layer in self.layers:
			layer.train()
		return self

	def eval(self) -> "Model":
		"""
		[brief] Active le mode évaluation sur le modèle et toutes ses couches.

		[details]
		Désactive Dropout, fige les statistiques BatchNorm.
		"""
		super().eval()
		for layer in self.layers:
			layer.eval()
		return self

	# ─── Helpers internes ─────────────────────────────────────────

	def _np_to_tensor(self, arr: np.ndarray) -> cvmml.Tensor:
		"""
		[brief] Convertit un np.ndarray en Tensor sur le device du modèle.

		[details]
		Crée un Tensor C++, copie les données via le buffer protocol NumPy,
		puis transfère vers CUDA si le modèle est configuré en mode GPU.

		Args:
			arr (np.ndarray): Données source, converties en float32 si nécessaire.

		Returns:
			Tensor: Tensor sur le device du modèle.
		"""
		arr = np.ascontiguousarray(arr, dtype=np.float32)
		t = cvmml.Tensor(list(arr.shape))
		np.asarray(t)[:] = arr
		if self.device == "cuda":
			t = t.to_cuda()
		return t

	def _tensor_to_np(self, t) -> np.ndarray:
		"""
		[brief] Convertit un Tensor en np.ndarray CPU.

		Args:
			t: Tensor C++ (CPU ou CUDA).

		Returns:
			np.ndarray: Copie des données sur CPU.
		"""
		if t.device() == cvmml.Device.CUDA:
			t = t.to_cpu()
		return np.array(t, dtype=np.float32, copy=True)

	@staticmethod
	def _compute_accuracy(preds_np: np.ndarray, targets_np: np.ndarray) -> float:
		"""
		[brief] Calcule l'accuracy en classification multi-classes ou binaire.

		[details]
		Détecte automatiquement le cas binaire (sortie 1D ou colonne unique)
		vs multi-classes (sortie [B, C] avec C > 1) et applique la bonne logique.

		Args:
			preds_np (np.ndarray): [B, C] ou [B, 1] ou [B] — scores/probabilités.
			targets_np (np.ndarray): [B, C] one-hot ou [B, 1] / [B] labels binaires.

		Returns:
			float: Fraction d'exemples correctement classifiés.
		"""
		preds_flat = preds_np.reshape(preds_np.shape[0], -1)
		targets_flat = targets_np.reshape(targets_np.shape[0], -1)

		if preds_flat.shape[1] > 1:
			# Multi-classes : argmax
			pred_labels = np.argmax(preds_flat, axis=1)
			true_labels = np.argmax(targets_flat, axis=1)
		else:
			# Binaire : seuil 0.5
			pred_labels = (preds_flat[:, 0] >= 0.5).astype(np.int32)
			true_labels = targets_flat[:, 0].astype(np.int32)

		return float(np.mean(pred_labels == true_labels))

	def _eval_metric(self, preds_np: np.ndarray, targets_np: np.ndarray,
					  metric: Optional[Union[str, Callable]]) -> Optional[float]:
		"""
		[brief] Dispatche le calcul de métrique selon le type fourni.

		Args:
			preds_np (np.ndarray): Prédictions du modèle.
			targets_np (np.ndarray): Cibles réelles.
			metric: "accuracy", callable(preds, targets) → float, ou None.

		Returns:
			float ou None: Valeur de la métrique, None si metric is None.
		"""
		if metric is None:
			return None
		if metric == "accuracy":
			return self._compute_accuracy(preds_np, targets_np)
		if callable(metric):
			return float(metric(preds_np, targets_np))
		raise ValueError(f"Metric inconnue : {metric}. Utiliser 'accuracy' ou un callable.")

	# ─── Keras-style public API ──────────────────────────────────

	def fit(self,
			x_data: np.ndarray,
			y_data: np.ndarray,
			epochs: int,
			batch_size: int,
			optimizer,
			loss_fn,
			shuffle: bool = True,
			metric: Optional[Union[str, Callable]] = None) -> Dict[str, List[float]]:
		"""
		[brief] Entraîne le modèle sur les données fournies pendant un nombre fixe d'epochs.

		[details]
		Boucle d'entraînement complète calquée sur Keras `model.fit()`:
		découpe en mini-batches, forward, calcul de la loss, backward, optimizer.step.
		Collecte un historique de loss (et optionnellement de métrique) par epoch.

		Args:
			x_data (np.ndarray): [N, ...] Données d'entrée (converties en float32).
			y_data (np.ndarray): [N, ...] Cibles (one-hot pour classification, valeurs pour régression).
			epochs (int): Nombre de passes complètes sur le dataset.
			batch_size (int): Taille des mini-batches.
			optimizer (Optimizer): Instance d'optimiseur déjà configurée.
			loss_fn (LossFunction): Fonction de perte à minimiser.
			shuffle (bool): Si True, mélange les indices à chaque epoch.
			metric (str | callable | None): "accuracy" pour classification, callable(preds, targets) → float, None pour ignorer.

		Returns:
			dict: {"loss": [float], "metric": [float]} historique par epoch.
		"""
		x_data = np.asarray(x_data, dtype=np.float32)
		y_data = np.asarray(y_data, dtype=np.float32)
		n_samples = x_data.shape[0]

		history: Dict[str, List[float]] = {"loss": []}
		if metric is not None:
			history["metric"] = []

		self.train()

		for epoch in range(epochs):
			indices = np.arange(n_samples)
			if shuffle:
				np.random.shuffle(indices)

			epoch_loss = 0.0
			epoch_preds = []
			epoch_targets = []
			n_batches = 0

			for start in range(0, n_samples, batch_size):
				end = min(start + batch_size, n_samples)
				batch_idx = indices[start:end]

				x_batch = self._np_to_tensor(x_data[batch_idx])
				y_batch = self._np_to_tensor(y_data[batch_idx])

				optimizer.zero_grad()

				y_pred = self(x_batch)
				loss = loss_fn(y_pred, y_batch)
				loss.backward()
				optimizer.step()

				# Collecter la loss (scalaire)
				loss_val = self._tensor_to_np(loss).item()
				epoch_loss += loss_val
				n_batches += 1

				# Collecter les prédictions pour la métrique
				if metric is not None:
					epoch_preds.append(self._tensor_to_np(y_pred))
					epoch_targets.append(y_data[batch_idx])

			avg_loss = epoch_loss / n_batches
			history["loss"].append(avg_loss)

			if metric is not None:
				all_preds = np.concatenate(epoch_preds, axis=0)
				all_targets = np.concatenate(epoch_targets, axis=0)
				metric_val = self._eval_metric(all_preds, all_targets, metric)
				history["metric"].append(metric_val)
				print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f} — {metric if isinstance(metric, str) else 'metric'}: {metric_val:.4f}")
			else:
				print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

		return history

	def evaluate(self,
				 x_data: np.ndarray,
				 y_data: np.ndarray,
				 loss_fn,
				 batch_size: Optional[int] = None,
				 metric: Optional[Union[str, Callable]] = None,
				 return_outputs: bool = False) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray]]:
		"""
		[brief] Évalue le modèle sur un jeu de données en mode inférence.

		[details]
		Passe le modèle en mode `eval`, forward par batches, calcule la loss moyenne
		et optionnellement une métrique. Peut retourner les vecteurs de sortie bruts.

		Args:
			x_data (np.ndarray): [N, ...] Données d'entrée.
			y_data (np.ndarray): [N, ...] Cibles.
			loss_fn (LossFunction): Fonction de perte.
			batch_size (int | None): Taille de batch (défaut : tout le dataset).
			metric (str | callable | None): Métrique à calculer.
			return_outputs (bool): Si True, retourne aussi les prédictions.

		Returns:
			tuple: (loss_moy, metric_moy) ou (loss_moy, metric_moy, outputs_np).
		"""
		x_data = np.asarray(x_data, dtype=np.float32)
		y_data = np.asarray(y_data, dtype=np.float32)
		n_samples = x_data.shape[0]
		if batch_size is None:
			batch_size = n_samples

		self.eval()

		total_loss = 0.0
		all_preds = []
		n_batches = 0

		for start in range(0, n_samples, batch_size):
			end = min(start + batch_size, n_samples)

			x_batch = self._np_to_tensor(x_data[start:end])
			y_batch = self._np_to_tensor(y_data[start:end])

			y_pred = self(x_batch)
			loss = loss_fn(y_pred, y_batch)

			total_loss += self._tensor_to_np(loss).item()
			all_preds.append(self._tensor_to_np(y_pred))
			n_batches += 1

		avg_loss = total_loss / n_batches
		outputs_np = np.concatenate(all_preds, axis=0)

		metric_val = self._eval_metric(outputs_np, y_data, metric)
		if metric_val is None:
			metric_val = 0.0

		if return_outputs:
			return avg_loss, metric_val, outputs_np
		return avg_loss, metric_val

	def predict(self,
				x_data: np.ndarray,
				batch_size: Optional[int] = None) -> np.ndarray:
		"""
		[brief] Effectue une inférence pure sur les données et retourne les sorties.

		[details]
		Passe le modèle en mode `eval`, exécute le forward par batches, concatène
		les résultats et retourne un np.ndarray.

		Args:
			x_data (np.ndarray): [N, ...] Données d'entrée.
			batch_size (int | None): Taille de batch (défaut : tout le dataset).

		Returns:
			np.ndarray: [N, ...] Sorties du modèle concaténées.
		"""
		x_data = np.asarray(x_data, dtype=np.float32)
		n_samples = x_data.shape[0]
		if batch_size is None:
			batch_size = n_samples

		self.eval()
		all_preds = []

		for start in range(0, n_samples, batch_size):
			end = min(start + batch_size, n_samples)
			x_batch = self._np_to_tensor(x_data[start:end])
			y_pred = self(x_batch)
			all_preds.append(self._tensor_to_np(y_pred))

		return np.concatenate(all_preds, axis=0)

	def get_layer_weights(self) -> Dict[str, np.ndarray]:
		"""
		[brief] Extrait les matrices de poids et biais de chaque couche pour visualisation.

		[details]
		Parcourt toutes les couches du modèle et construit un dictionnaire
		nommant chaque paramètre `"idx_ClassName_weight"` / `"idx_ClassName_bias"`.
		Seules les couches possédant un attribut `weight` ou `bias` sont incluses.

		Returns:
			dict[str, np.ndarray]: Dictionnaire {nom → matrice numpy} des poids du modèle.
		"""
		weights: Dict[str, np.ndarray] = {}
		for idx, layer in enumerate(self.layers):
			name = type(layer).__name__
			if hasattr(layer, "weight") and layer.weight is not None:
				w = layer.weight
				if w.device() == cvmml.Device.CUDA:
					w = w.to_cpu()
				weights[f"{idx}_{name}_weight"] = np.array(w, dtype=np.float32, copy=True)
			if hasattr(layer, "bias") and layer.bias is not None:
				b = layer.bias
				if b.device() == cvmml.Device.CUDA:
					b = b.to_cpu()
				weights[f"{idx}_{name}_bias"] = np.array(b, dtype=np.float32, copy=True)
		return weights
