# TinyMML roadmap

This roadmap distinguishes behavior validated on the reference CPU backend from experimental code that merely exists in the repository.

*   **Ops de Vision**
    *   [ ] Grayscale, Resizing, Normalisation en C++.
*   **Convolutional Networks**
    *   [ ] Entraînement sur CIFAR-10.
*   **Pipeline Génératif**
    *   [ ] Super-résolution ou modèles de diffusion simplifiés.


## Experimental

- [ ] Validate CUDA compilation and the existing elementwise, reduction, and matrix kernels on CUDA hardware.
- [ ] Implement and verify CUDA backward parity for views, contiguous tensors, `im2col`, and Conv2D.
- [ ] Add CUDA verification without weakening explicit errors in CPU-only builds.

## Next

- [ ] Profile CPU kernels before adding or expanding parallelism.
- [ ] Add model state serialization and restoration.
- [ ] Add a small data-loader abstraction without hiding NumPy interoperability.
- [ ] Extend convolution options only when supported by gradient checks.
- [ ] Add release packaging after the public API stabilizes.

## Explicit non-goals today

- No Metal backend.
- No claim of CUDA training validation.
- No claim that TinyMML should outperform optimized frameworks.
- No open-source licensing claim; the repository currently has no license.
