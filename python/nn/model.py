from __future__ import annotations
import math
import cvmml_api as cvmml
from typing import Iterable, List, Optional
from .layer import Layer

#* --- Abstract Class --- *
class Model(Layer):
	"""
	[brief] Conteneur séquentiel maître pour structurer un modèle de Deep Learning.

	[details]
	Regroupe la modularité, stocke récursivement les paramètres, et propage les données.
	Délègue l'exécution à chaque sous-couche successivement via une simple boucle `for layer in self.layers`. Induit automatiquement la dimensionnalité (`build`) lors du premier appel.

	Args:
		layers (Optional[Iterable[Layer]]): Liste ordonnée de couches à exécuter consécutivement.

	Returns:
		Tensor: [Dimensions sortantes] Résultat complet du traitement réseau.
	"""
	def __init__(self, layers: Optional[Iterable[Layer]] = None) -> None:
		super().__init__()
		self.layers: List[Layer] = list(layers) if layers is not None else []

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
		super().train()
		for layer in self.layers:
			layer.train()
		return self

	def eval(self) -> "Model":
		super().eval()
		for layer in self.layers:
			layer.eval()
		return self
