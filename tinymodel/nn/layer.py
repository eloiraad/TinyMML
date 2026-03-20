from __future__ import annotations

import math
import cvmml_api as cvmml
import numpy as np
from abc import ABC, abstractmethod
from typing import List


class Layer(ABC):
	def __init__(self) -> None:
		self._training = True
		self._built = False
		self._input_shape = None
		self._output_shape = None

	@property
	def training(self) -> bool:
		return self._training

	@property
	def built(self) -> bool:
		return self._built

	@property
	def input_shape(self):
		return self._input_shape

	@property
	def output_shape(self):
		return self._output_shape

	def train(self) -> "Layer":
		self._training = True
		return self

	def eval(self) -> "Layer":
		self._training = False
		return self

	def build(self, input_shape) -> object:
		if self._built:
			return self._output_shape
		self._input_shape = list(input_shape) if input_shape is not None else None
		output_shape = self._build(input_shape)
		self._output_shape = list(output_shape) if output_shape is not None else None
		self._built = True
		return self._output_shape

	def __call__(self, x):
		if not self._built:
			inferred_shape = x.shape() if hasattr(x, "shape") else None
			self.build(inferred_shape)
		return self.forward(x)

	def parameters(self) -> List:
		return []

	@abstractmethod
	def _build(self, input_shape):
		pass

	@abstractmethod
	def forward(self, x):
		pass


#* --- Linear Layer Class --- *
class Linear(Layer):
	"""
	[brief] Applique une transformation linéaire aux données entrantes : y = xW^T + b.

	[details]
	Construit les couches denses (fully connected) fondamentales dans les réseaux de neurones.
	Implémente la multiplication matricielle de l'entrée avec des poids apprenables, optionnellement suivie d'une addition de biais.

	Args:
		out_features (int): Taille de chaque échantillon de sortie.
		bias (bool): Si True, ajoute un biais apprenable de dimension [1, out_features].
		init (str): Méthode d'initialisation des poids ("xavier" ou "he").

	Returns:
		Tensor: [..., out_features] Tensor transformé lors du forward.
	"""
	def __init__(self, out_features: int, bias: bool = True, init: str = "xavier"):
		super().__init__()
		self.out_features = out_features
		self.use_bias = bias
		self.init = init
		self.weight = None
		self.bias = None

	def _build(self, input_shape):
		if input_shape is None or len(input_shape) < 2:
			raise ValueError("Linear input shape must be [..., in_features]")
		in_features = input_shape[-1]

		self.weight = cvmml.Tensor.randn([in_features, self.out_features], 0.0, 1.0)
		if self.init == "he":
			scale = math.sqrt(2.0 / in_features)
		else:
			scale = math.sqrt(1.0 / in_features)
		self.weight = self.weight * scale
		self.weight.set_requires_grad(True)

		if self.use_bias:
			self.bias = cvmml.Tensor.zeros([1, self.out_features])
			self.bias.set_requires_grad(True)

		out_shape = list(input_shape)
		out_shape[-1] = self.out_features
		return out_shape

	def forward(self, x):
		y = x.mm(self.weight)
		if self.bias is not None:
			y = y + self.bias
		return y

	def parameters(self):
		if self.bias is None:
			return [self.weight]
		return [self.weight, self.bias]

#* --- SoftMax Layer Class --- *
class Softmax(Layer):
	"""
	[brief] Applique la fonction Softmax sur un axe spécifique.

	[details]
	Normalise les scores (logits) pour générer une distribution de probabilité.
	Stabilisé numériquement en soustrayant le maximum de l'axe avant l'exponentiation pour éviter l'overflow.

	Args:
		axis (int): L'axe le long duquel appliquer Softmax (par défaut -1).

	Returns:
		Tensor: [Mêmes dimensions] Valeurs comprises entre 0 et 1, sommant à 1.
	"""
	def __init__(self, axis: int = -1):
		super().__init__()
		self.axis = axis

	def _build(self, input_shape):
		if input_shape is None:
			raise ValueError("Softmax need an input shape")
		rank = len(input_shape)
		axis = self.axis if self.axis >= 0 else self.axis + rank
		if axis < 0 or axis >= rank:
			raise ValueError("axis out of range")
		self.axis = axis
		return input_shape

	def forward(self, x):
		m = x.max([self.axis], True)
		z = x - m
		e = z.exp()
		s = e.sum([self.axis], True)
		return e / s

	def parameters(self):
		return []

#* --- ReLU --- *
class ReLU(Layer):
	"""
	[brief] Applique la fonction Rectified Linear Unit (ReLU) élément par élément.

	[details]
	Introduit de la non-linéarité dans le réseau pour apprendre des motifs complexes.
	Remplace toutes les valeurs négatives par zéro in-place ou via copie selon l'implémentation du tenseur (max(0, x)).

	Args:
		Aucun paramètre d'initialisation.

	Returns:
		Tensor: [Mêmes dimensions] Tenseur activé.
	"""
	def _build(self, input_shape):
		return input_shape

	def forward(self, x):
		return x.relu()

	def parameters(self):
		return []

#* --- Dropout Layer Class --- *
class Dropout(Layer):
	"""
	[brief] Inactive aléatoirement un pourcentage des neurones d'entrée.

	[details]
	Technique de régularisation efficace qui prévient la co-adaptation et freine le sur-apprentissage.
	Applique un "Inverted Dropout" : les neurones conservés sont amplifiés par (1 / (1-rate)) pour que l'échelle des valeurs au test reste identique. Le masque est généré par `Tensor.bernoulli`.

	Args:
		rate (float): Probabilité (entre 0.0 et 1.0) de mise à zéro.

	Returns:
		Tensor: [Mêmes dimensions] Tenseur avec masque appliqué.
	"""

	def __init__(self, rate: float = 0.5):
		super().__init__()
		if not 0.0 <= rate < 1.0:
			raise ValueError("Dropout rate must be in [0, 1).")
		self.rate = rate

	def _build(self, input_shape):
		return input_shape

	def forward(self, x):
		if not self._training or self.rate == 0.0:
			return x
		keep_prob = 1.0 - self.rate
		mask = cvmml.Tensor.bernoulli(list(x.shape()), keep_prob)
		if x.device() == cvmml.Device.CUDA:
			mask = mask.to_cuda()
		return (x * mask) * (1.0 / keep_prob)

	def parameters(self):
		return []


#* --- Conv2D Layer Class --- *
class Conv2D(Layer):
	"""
	[brief] Applique une convolution spatiale 2D sur un tenseur d'entrée paramétré.

	[details]
	Module de base des réseaux convolutifs utilisé pour l'extraction de caractéristiques locales.
	Utilise `im2col` pour extraire des patchs, les transpose, puis calcule le résultat via une unique multiplication matricielle.

	Args:
		out_channels (int): Nombre de filtres produits en sortie.
		kernel_size (int): Taille du noyau (carré de kernel_size x kernel_size).
		stride (int): Pas de la fenêtre de balayage.
		padding (int): Nombre de zéros artificiels ajoutés bord à bord.
		bias (bool): Si True, inclut un biais indépendant [1, out_channels, 1, 1].
		init (str): Méthode d'initialisation ("xavier" ou "he").

	Returns:
		Tensor: [B, out_channels, H_out, W_out] Cartes de caractéristiques convoluées.
	"""

	def __init__(self, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True, init: str = "xavier"):
		super().__init__()
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.use_bias = bias
		self.init = init
		self.weight = None
		self.bias = None

	def _build(self, input_shape):
		if input_shape is None or len(input_shape) != 4:
			raise ValueError("Conv2D expects input shape [B, C_in, H, W]")

		C_in = input_shape[1]
		H = input_shape[2]
		W = input_shape[3]
		kH = kW = self.kernel_size

		H_out = (H + 2 * self.padding - kH) // self.stride + 1
		W_out = (W + 2 * self.padding - kW) // self.stride + 1

		fan_in = C_in * kH * kW
		self.weight = cvmml.Tensor.randn([self.out_channels, fan_in], 0.0, 1.0)
		if self.init == "he":
			scale = math.sqrt(2.0 / fan_in)
		else:
			scale = math.sqrt(1.0 / fan_in)
		self.weight = self.weight * scale
		self.weight.set_requires_grad(True)

		if self.use_bias:
			self.bias = cvmml.Tensor.zeros([1, self.out_channels, 1, 1])
			self.bias.set_requires_grad(True)

		return [input_shape[0], self.out_channels, H_out, W_out]

	def forward(self, x):
		B = x.shape()[0]
		kH = kW = self.kernel_size
		H_out = (x.shape()[2] + 2 * self.padding - kH) // self.stride + 1
		W_out = (x.shape()[3] + 2 * self.padding - kW) // self.stride + 1

		cols = x.im2col(kH, kW, self.stride, self.padding)

		cols_t = cols.T(1, 2)
		wt = self.weight.T(0, 1)
		out = cols_t.mm(wt)
		out = out.T(1, 2)
		out = out.view([B, self.out_channels, H_out, W_out])

		if self.bias is not None:
			out = out + self.bias
		return out

	def parameters(self):
		if self.bias is None:
			return [self.weight]
		return [self.weight, self.bias]


#* --- BatchNorm Layer Class --- *
class BatchNorm(Layer):
	"""
	[brief] Normalise indépendamment le batch sur la dimension des caractéristiques.

	[details]
	Accélère et stabilise l'entraînement des réseaux profonds en réduisant le déplacement de covariable interne.
	Calcule la moyenne et la variance sur l'axe 0. Le buffer de statistiques est mis à jour discrètement (détaché du graphe) via un EMA.

	Args:
		eps (float): Valeur minimale ajoutée à la variance pour la stabilité numérique.
		momentum (float): Constante pour l'ajustement exponentiel des statistiques.

	Returns:
		Tensor: [B, features, ...] Tenseur recalibré par gamma et beta.
	"""

	def __init__(self, eps: float = 1e-5, momentum: float = 0.1):
		super().__init__()
		self.eps = eps
		self.momentum = momentum
		self.gamma = None
		self.beta = None
		self.running_mean = None
		self.running_var = None

	def _build(self, input_shape):
		if input_shape is None or len(input_shape) < 2:
			raise ValueError("BatchNorm expects at least 2‑D input [B, features, ...]")
		features = input_shape[1]

		self.gamma = cvmml.Tensor.ones([1, features])
		self.gamma.set_requires_grad(True)
		self.beta = cvmml.Tensor.zeros([1, features])
		self.beta.set_requires_grad(True)

		self.running_mean = cvmml.Tensor.zeros([1, features])
		self.running_var = cvmml.Tensor.ones([1, features])

		return input_shape

	def forward(self, x):
		if self._training:
			mean = x.mean([0], True)
			var  = x.var([0], True)

			x_hat = (x - mean) / (var + self.eps).sqrt()

			m = self.momentum
			rm_np = np.asarray(self.running_mean)
			rv_np = np.asarray(self.running_var)
			rm_np[:] = (1.0 - m) * rm_np + m * np.asarray(mean.contiguous())
			rv_np[:] = (1.0 - m) * rv_np + m * np.asarray(var.contiguous())
		else:
			x_hat = (x - self.running_mean) / (self.running_var + self.eps).sqrt()

		return self.gamma * x_hat + self.beta

	def parameters(self):
		return [self.gamma, self.beta]


#* --- LayerNorm Layer Class --- *
class LayerNorm(Layer):
	"""
	[brief] Normalise individuellement chaque échantillon sur sa dernière dimension.

	[details]
	Pallie les limites de BatchNorm sur les mini-batch ou les architectures séquentielles type Transformers.
	Calcule la moyenne et la variance exclusivement sur le dernier axe.

	Args:
		eps (float): Scalaire de stabilité numérique.

	Returns:
		Tensor: [..., features] Entrées normalisées.
	"""

	def __init__(self, eps: float = 1e-5):
		super().__init__()
		self.eps = eps
		self.gamma = None
		self.beta = None

	def _build(self, input_shape):
		if input_shape is None or len(input_shape) < 1:
			raise ValueError("LayerNorm needs at least 1‑D input")
		features = input_shape[-1]

		self.gamma = cvmml.Tensor.ones([1, features])
		self.gamma.set_requires_grad(True)
		self.beta = cvmml.Tensor.zeros([1, features])
		self.beta.set_requires_grad(True)

		return input_shape

	def forward(self, x):
		rank = len(x.shape())
		last_axis = rank - 1
		mean = x.mean([last_axis], True)
		var  = x.var([last_axis], True)
		x_hat = (x - mean) / (var + self.eps).sqrt()
		return self.gamma * x_hat + self.beta

	def parameters(self):
		return [self.gamma, self.beta]


#* --- ResidualBlock --- *
class ResidualBlock(Layer):
	"""
	[brief] Implémente une architecture de connexion résiduelle (Skip Connection).

	[details]
	Prévient la disparition du gradient dans les réseaux profonds en ajoutant le signal d'origine à la sortie.
	Propage l'entrée dans une suite de 'sub_layers', puis additionne son résultat à l'entrée d'origine (y = x + F(x)).

	Args:
		*layers (Layer): Séquence d'objets Layer constituant la transformation F(x).

	Returns:
		Tensor: [Mêmes dimensions] Résultat final additionné.
	"""

	def __init__(self, *layers):
		super().__init__()
		self.sub_layers = list(layers)

	def _build(self, input_shape):
		current_shape = list(input_shape) if input_shape is not None else None
		for layer in self.sub_layers:
			if not layer.built:
				current_shape = layer.build(current_shape)
			else:
				current_shape = layer.output_shape
		if current_shape != list(input_shape):
			raise ValueError(
				f"ResidualBlock: sub‑layers changed shape from {input_shape} "
				f"to {current_shape}. Skip connection requires identical shapes."
			)
		return current_shape

	def forward(self, x):
		out = x
		for layer in self.sub_layers:
			out = layer(out)
		return x + out

	def parameters(self):
		params = []
		for layer in self.sub_layers:
			params.extend(layer.parameters())
		return params

	def train(self) -> "ResidualBlock":
		super().train()
		for layer in self.sub_layers:
			layer.train()
		return self

	def eval(self) -> "ResidualBlock":
		super().eval()
		for layer in self.sub_layers:
			layer.eval()
		return self
