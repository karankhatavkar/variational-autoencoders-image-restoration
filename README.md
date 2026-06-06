# Image Restoration with Variational Autoencoders

Three notebooks, each using VAEs on a harder problem than the last.

The first notebook establishes the architecture on MNIST — straightforward generative modeling with latent space visualization. The second applies the same class of model to natural images (COCO), using random polygon masks to corrupt images and measuring reconstruction quality with PSNR and SSIM. The third does the same on face images (CelebA), where the structured nature of faces gives the latent space more to work with.

---

## Notebooks

### `01_vae_mnist_generative.ipynb` — [Kaggle](https://www.kaggle.com/code/devkarankh/variational-autoencoder)
**Dataset:** MNIST (60K training, 10K validation, 28×28 grayscale)

Builds a convolutional VAE with a 2D latent space. The low dimensionality is deliberate — it lets you walk the latent space and see continuous transitions between digit classes in a 20×20 manifold grid.

Architecture:
- Encoder: Conv2D(32) → Conv2D(64, stride=2) → Conv2D(64) → Dense(32) → z_mean, z_log_sigma
- Decoder: Dense(3136) → Reshape → Conv2DTranspose(32, stride=2) → Conv2DTranspose(1, sigmoid)
- Loss: Binary crossentropy + KL divergence (weight: -5e-4)
- Optimizer: RMSprop

Output: 20×20 latent space manifold showing smooth digit interpolations; PCA projection of the latent space colored by digit class.

---

### `02_inpainting_coco_conv_vae.ipynb` — [Kaggle](https://www.kaggle.com/code/devkarankh/vae-missing-part-of-image)
**Dataset:** COCO (128×128 RGB images; 70/30 train/val split)

Corruption method: random polygon masks covering 2–20% of each image, applied at training time. The model learns to reconstruct the full image from the corrupted input.

Architecture:
- Encoder: Conv2D(32, stride=2) → Conv2D(64, stride=2) → Conv2D(128, stride=2) → Conv2D(256, stride=2) → z_mean, z_log_var (latent_dim=64)
- Decoder: symmetric transposed convolution path → Conv2DTranspose(3, sigmoid)
- Beta-VAE: β=0.005 (reconstruction-first tradeoff — lower β means the model prioritizes filling the masked region over learning a tightly regularized latent space)
- Training: 60 epochs, batch size 32, Adam (lr=1e-4)

Evaluation per test image:
- MSE between reconstructed and original
- PSNR (dB)
- SSIM

Also includes a 5×5 local latent neighborhood grid — sample nearby points in latent space and decode them — to check whether the latent space is smooth around a given image.

---

### `03_facial_inpainting_celeba_vae.ipynb` — [Kaggle](https://www.kaggle.com/code/devkarankh/vae-missing-part-of-face-image)
**Dataset:** CelebA (128×128 RGB face images; 70/30 split)

Same architecture and corruption strategy as notebook 02, but on face images. Faces have stronger spatial structure than general objects — jaw lines, eye placement, skin tone distributions — so the model learns a more interpretable latent space.

Outputs:
- Side-by-side: corrupted → reconstructed → filled region → original
- PCA of latent space (2D projection of 64D codes)
- 5×5 latent neighborhood grid showing plausible face variations around a query image
- MSE / PSNR / SSIM on the test set

The neighborhood sampling is the most interesting part: small steps in the 64D latent space produce consistent changes in hair color, skin tone, and facial structure, which shows the model encoded something meaningful rather than memorized training images.

---

## Architecture Comparison

| Notebook | Dataset | Latent Dim | Corruption | Metrics |
|---|---|---|---|---|
| 01 | MNIST | 2 | None (generative) | Manifold visualization |
| 02 | COCO 128×128 | 64 | Random polygon masks (2–20%) | MSE, PSNR, SSIM |
| 03 | CelebA 128×128 | 64 | Random polygon masks (2–20%) | MSE, PSNR, SSIM |

---

## Setup

```bash
pip install tensorflow numpy matplotlib scikit-image scikit-learn pillow
```

Datasets:
- MNIST: loaded via `tensorflow.keras.datasets.mnist`
- [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) — use the validation subset for faster iteration
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

Place datasets in the paths referenced in each notebook's data loading cells.
