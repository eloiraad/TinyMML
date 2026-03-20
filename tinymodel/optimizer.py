from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import tinytensor as tt

class Optimizer(ABC):
	"""
	[brief] Classe de base abstraite pour tous les optimiseurs de gradients.

	[details]
	Définit l'interface commune et stocke les paramètres à optimiser pour faciliter le polymorphisme.
	Gère la liste des paramètres modifiables du modèle et fournit la méthode universelle `zero_grad()` pour effacer les gradients avant chaque itération d'entraînement.

	Args:
		parameters (List): Liste des objets `cvmml.Tensor` (paramètres du modèle).
		lr (float): Taux d'apprentissage de base.

	Returns:
		Aucun retour pour l'initialisation.
	"""
	def __init__(self, parameters: List, lr: float):
		self.parameters = list(parameters)
		self.lr = lr

	def zero_grad(self):
		for param in self.parameters:
			param.zero_grad()

	@abstractmethod
	def step(self):
		pass


class SGD(Optimizer):
	"""
	[brief] Optimiseur par Descente de Gradient Stochastique (Stochastic Gradient Descent).

	[details]
	Met à jour les poids pour minimiser l'erreur selon la direction du gradient avec support de l'inertie (momentum).
	Soustrait à chaque paramètre la valeur de son gradient (pondéré par `lr`). Utilise des arrays `numpy` internes pour calculer les inerties sans allouer de nouveaux Tensors. Les opérations de mise à jour s'effectuent via `.subtract_()` (in-place) pour rester sur le device (C++ / CUDA) et éviter l'explosion de l'autograd.

	Args:
		parameters (List): Paramètres à optimiser.
		lr (float): Taux d'apprentissage.
		momentum (float): Constante d'inertie accélérant la descente dans les directions constantes.
		weight_decay (float): Pénalité L2 pour régulariser les poids.

	Returns:
		Aucun retour. `step()` met à jour les tenseurs en-place.
	"""
	def __init__(self, parameters: List, lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
		super().__init__(parameters, lr)
		self.momentum = momentum
		self.weight_decay = weight_decay
		self.velocities = []
		for p in self.parameters:
			self.velocities.append(np.zeros(p.shape(), dtype=np.float32))

	def step(self):
		for idx, param in enumerate(self.parameters):
			grad = param.grad()
			if grad is None:
				continue

			g = np.array(grad, dtype=np.float32, copy=False)
			if self.weight_decay != 0.0:
				g += self.weight_decay * np.asarray(param)
			if self.momentum != 0.0:
				self.velocities[idx] = self.momentum * self.velocities[idx] + g
				g = self.velocities[idx]

			grad_tensor = tt.Tensor(list(param.shape()))
			np.asarray(grad_tensor)[:] = g.reshape(param.shape())
			scaled = grad_tensor * self.lr
			if param.device() == tt.Device.CUDA:
				scaled = scaled.to_cuda()
			param.subtract_(scaled)

class Adam(Optimizer):
	"""
	[brief] Optimiseur Adaptive Moment Estimation (Adam).

	[details]
	Ajuste individuellement le taux d'apprentissage de chaque paramètre, offrant une convergence très rapide et robuste.
	Calcule des estimations glissantes du premier moment (moyenne locale du gradient) et du second moment (variance locale non centrée), avec correction de biais selon l'itération `t`. Les buffers modifiés (arrays numpy) s'appliquent sur les `Tensor` in-place.

	Args:
		parameters (List): Paramètres à optimiser.
		lr (float): Taux d'apprentissage asymptotique maximum.
		beta1 (float): Taux de décroissance pour le premier moment (momentum).
		beta2 (float): Taux de décroissance pour le second moment (RMSprop).
		eps (float): Terme de stabilité pour diviser sans erreur par zéro.
		weight_decay (float): Pénalité de régularisation L2.

	Returns:
		Aucun retour. `step()` met à jour les poids in-place.
	"""
	def __init__(self, parameters: List, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
		super().__init__(parameters, lr)
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps
		self.weight_decay = weight_decay
		self.t = 0
		self.m = []
		self.v = []
		for p in self.parameters:
			self.m.append(np.zeros(p.shape(), dtype=np.float32))
			self.v.append(np.zeros(p.shape(), dtype=np.float32))

	def step(self):
		self.t += 1
		for idx, param in enumerate(self.parameters):
			grad = param.grad()
			if grad is None:
				continue

			g = np.array(grad, dtype=np.float32, copy=False)
			if self.weight_decay != 0.0:
				g += self.weight_decay * np.asarray(param)

			self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * g
			self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * (g ** 2)

			m_hat = self.m[idx] / (1.0 - (self.beta1 ** self.t))
			v_hat = self.v[idx] / (1.0 - (self.beta2 ** self.t))
			
			update = m_hat / (np.sqrt(v_hat) + self.eps)

			grad_tensor = tt.Tensor(list(param.shape()))
			np.asarray(grad_tensor)[:] = update.reshape(param.shape())

			scaled = grad_tensor * self.lr
			if param.device() == tt.Device.CUDA:
				scaled = scaled.to_cuda()
			param.subtract_(scaled)
