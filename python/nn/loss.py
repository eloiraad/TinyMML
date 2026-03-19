from __future__ import annotations

from abc import ABC, abstractmethod
import cvmml_api as cvmml


class LossFunction(ABC):
	"""
	[brief] Interface de base pour les fonctions d'erreur (Loss).

	[details]
	Uniformise l'évaluation des erreurs entre les prédictions du modèle et les cibles (targets) réelles.
	Définit un constructeur universel par la méthode `__call__` qui renvoie la logique calculatoire abstraite `forward()`.

	Args:
		Aucun paramètre d'initialisation global.

	Returns:
		Aucun retour d'init. `__call__` retournera le Tenseur scalaire de perte.
	"""

	@abstractmethod
	def forward(self, predictions, targets):
		pass

	def __call__(self, predictions, targets):
		return self.forward(predictions, targets)


class CrossEntropyLoss(LossFunction):
	"""
	[brief] Calcule l'erreur d'Entropie Croisée (fusion Softmax + Negative Log-Likelihood).

	[details]
	L'estimateur de perte standard et le plus performant pour la classification multi-classes.
	Fusionne logistiquement l'application de Softmax (avec soustraction du maximum) et NLLLoss. Cette fusion garantit que des logits extrêmes n'induisent pas de problèmes de NaN.

	Args:
		Aucun paramètre d'initialisation.

	Returns:
		Tensor: [1] Un scalaire moyenné de la perte du batch, calculée lors du forward prenant `logits` [B, C] et `targets` [B, C].
	"""

	def forward(self, logits, targets):
		max_logits = logits.max([1], True)
		shifted = logits - max_logits

		exp_shifted = shifted.exp()
		sum_exp = exp_shifted.sum([1], True)
		log_sum_exp = sum_exp.log()
		log_softmax = shifted - log_sum_exp

		nll = (targets * log_softmax).sum([1], False)
		loss = nll.sum([0], False) * (-1.0 / logits.shape()[0])
		return loss

class MSELoss(LossFunction):
	"""
	[brief] Calcule l'erreur quadratique moyenne (Mean Squared Error).

	[details]
	Optimise les tâches de régression en pénalisant fortement les grandes déviations par rapport aux cibles.
	Applique `(pred - target)^2` élément par élément, puis calcule la moyenne totale avec l'outil tensoriel `mean`.

	Args:
		Aucun paramètre d'initialisation.

	Returns:
		Tensor: [1] Tenseur scalaire contenant la moyenne des erreurs quadratiques successives de la passe forward.
	"""
	def forward(self, predictions, targets):
		diff = predictions - targets
		sq = diff * diff
		axes = list(range(len(sq.shape())))
		loss = sq.mean(axes, False)
		return loss

