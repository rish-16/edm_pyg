# edm_pyg [WIP]
PyTorch Geometric wrapper of Equivariant Diffusion Models (EDM) for geometric graph generation.

## What is this?
This repo provides a ready-to-use wrapper over the Equivariant Diffusion Model (EDM) by Hoogeboom, Satorras, Vignac, and Welling (2022) ([`abs`](https://arxiv.org/abs/2203.17003), [`pdf`](https://arxiv.org/pdf/2203.17003.pdf)). The [original codebase](https://github.com/ehoogeboom/e3_diffusion_for_molecules) relies on pure PyTorch, requiring dense batch representations of the geometric graphs. From personal experience, this made it difficult to train on graphs with >50 nodes with very small batch sizes even without an over-the-top GPU. 

## Why?
HuggingFace `diffusers` definitely exists but there's no go-to DDPM framework for graphs, let alone geometric graphsk – at least none that works with PyTorch Geometric and sparse batches of geometric data. This repository seeks to address that gap.

1. Installation
2. Implementation Details
3. Usage
    1. Training
    2. Sampling
    3. Customisation
4. Contributing
5. Acknowledgements

## Installation

You can install this library via the Python package manager, `pip`:

```bash
$ pip install edm_pyg
```

Or feel free to install whatever is on the `requirements.txt` file in the main directory.

## Specifics
The library uses the following _forward SDE_ formulation for an initial data distribution $q(\mathbf{x}_0)$.

$$
d\mathbf{x} = f(t)\mathbf{x}dt + g(t)d\mathbf{w}_\mathbf{x}
$$

where $\mathbf{w}_\mathbf{x}$ is the standard Weiner process and $f(t)$ and $g(t)$ are scalar functions that control the diffusion process. The corresponding reverse SDE is,

$$
d\mathbf{x} = [f(t)\mathbf{x}-g(t)^2 \nabla_\mathbf{x}\log q_t(\mathbf{x})]dt \nonumber + g(t)d\mathbf{\tilde{w}}_\mathbf{x}
$$

where $`\mathbf{\tilde{w}}_\mathbf{x}`$ is the reverse-time standard Wiener process. The score function $`\nabla_\mathbf{x}\log q_t(\mathbf{x})`$ is parameterised by a denoising dynamics model $`\epsilon_\theta(\mathbf{x}_t, t)`$ which `edm_pyg` allows you to train using the standard MSE objective:

$$
\mathcal{L} = ||\epsilon_\theta(\mathbf{x}_t,t)-\epsilon_t||^2_2, \quad \epsilon_t \sim \mathcal{N}(0, \mathbf{I})
$$

## Usage
`edm_pyg` allows you to both train the dynamics model and sample from it as demonstrated by the original EDM framework. 

### Training

```python
from edm_pyg import EDMWrapper
from models import EGNNModel

dynamics = EGNNModel(...).to(device)
opt = torch.optim.Adam(dynamics.parameters())
edm = EDMWrapper(
  diffuser=dynamics,
  timesteps=500,
  scheduler="polynomial",
  condition_time=True,
  batch_size=64,
  grad_clip=False,
  ema=True, 
  ema_decay=0.995, # not necessary if ema=False
).to(device)

batch = next(iter(train_loader)) # should be a pyg.data.Batch(...) object

for epoch in range(100):
  epoch_loss, epoch_nll = edm.train_epoch(epoch, train_loader, opt)
```

Suppose you've finished training the model and wish to sample. As different geometric datasets have different protocols on how to save data (eg: PDB for proteins, xyz for QM9, etc), the library provides some flexibility in how you choose to deal with the sampled objects from the EDM. Internally, `edm_pyg.EDMWrapper` calls the custom saving function on the generated tensors that pop out as timestep $t$ goes from $T$ to $0$.

```python
def custom_save_sample(sampled_tensor):
  # deal with your tensors
  
test_loss, test_nll = edm.sample(test_loader, custom_save_sample)
```

> I suggest heading over the [Geometric GNN Dojo](https://github.com/chaitjo/geometric-gnn-dojo) by Joshi, Bodnar, Mathis, Cohen, and Lio (2022) in case you want to find off-the-shelf geometric GNNs that serve as denoisers. You can find different implementations in the `src/models.py` file. Depending on the model, you may need to copy over extra accessory files so do remember to cite them for their amazing compilation! 

### Customisation
You can also configure the EDM for your specific task based on what's being diffused over. 

#### Custom Noise Schedules
The repository offers a few pre-built schedules: `linear`, `cosine`, `polynomial`, `learned`. However, suppose you discovered a custom schedule (congrats!) that you wish to use. You can do so using a custom `torch.nn.Module` class with a required `forward` method:

```python
import torch

class MyAwesomeSchedule(torch.nn.Module):
  def __init__(self, total_timesteps):
    super().__init__()
    
    # pre-compute lookup arrays of values to access (like beta, gamma, alpha, alpha_bar, etc)
    
  def forward(self, t):
    # return epicly computed variance schedule at time t

edm = EDMWrapper(
  dynamics=my_model,
  ...,
  schedule=custom_schedule,
  timesteps=500,
  ...
)
```

> Note: this entire repository has minimal coupling between files and their various dependencies so that you can copy and tinker with the codebase. The methods also have extensive comments to help guide the process of making custom changes for your application. If you have any doubts here, feel free to reach out via an Issue or email! 

## Contributing

This library is based off the original implementation by Hoogeboom, Satorras, Garcia, and Welling. If you run into any issues or have any fixes/suggestions, feel free to drop an Issue or PR!

## Acknowledgements
This wrapper was inspired by the original "Equivariant Diffusion for Molecule Generation in 3D" paper by _Hoogeboom et al. (2022)_ and is based off of the original codebase.

```
@misc{hoogeboom2022equivariant,
      title={Equivariant Diffusion for Molecule Generation in 3D}, 
      author={Emiel Hoogeboom and Victor Garcia Satorras and Clément Vignac and Max Welling},
      year={2022},
      eprint={2203.17003},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
