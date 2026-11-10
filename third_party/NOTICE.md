# Third-party model notices

NovaTurbo does not vendor the four upstream model repositories. It imports pinned packages through thin adapters, except for the low-dimensional CVAE adaptation described below.

- PyTorch-VAE: copyright 2020 Anand Krishnamoorthy Subramanian, Apache-2.0. NovaTurbo retains the public ConditionalVAE reparameterization and ELBO organization as a reference, replacing the image CNN and discrete class condition with a 5D/2D MLP interface. The adapted file is `src/ai/generators/cvae.py`.
- nflows: copyright 2020 Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios, MIT. Imported as a pinned dependency without copying its transforms or probability code.
- FrEIA: copyright 2018 Lynton Ardizzone, MIT. Imported as a pinned dependency without copying its invertible blocks.
- TorchCFM: copyright 2023 Alexander Tong, MIT. Imported as a pinned dependency without copying its conditional path implementation.
- torchdiffeq: copyright 2018 Ricky Tian Qi Chen, MIT. Imported as the pinned ODE solver used by the CFM adapter.

Repository URLs, commits/releases, package versions, license hashes, papers, examples, core classes and allowed local changes are frozen in `config/third_party_models.lock.yaml`.
