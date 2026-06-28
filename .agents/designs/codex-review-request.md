# Codex review request

Paste the prompt below into the Codex IDE extension chat, with `.agents/designs/design.md` open (or referenced) in the workspace.

---

Please review `.agents/designs/design.md` in this repo. It is implementation guidance for a coding agent to add VICReg and Barlow Twins self-supervised baselines (branch `2-vicreg-barlow-twins-baselines`) and Colab notebooks, per `.agents/designs/idea.md` Section 2.1.

Before reviewing, read these files to ground your assessment in the actual codebase:
- `.agents/designs/idea.md` (the research plan the design serves)
- `models/__init__.py`, `models/simclr.py`, `models/simsiam.py`, `models/pseudo_supervised_net.py`
- `main.py` (esp. `build_train_loader`, `forward_batch`, `train_model`, `build_optimizer_and_scheduler`)
- `arguments.py`, `augmentations/__init__.py`, `datasets/__init__.py`, `tools/knn_monitor.py`
- `configs/meta_exps/meta_random_config.yaml`, `tests/test_negatives_ratio.py`
- `notebooks/random-meta-cifar10-ssl.ipynb`

Assess the design on:
1. **Correctness against the codebase** — do the named registration points, the model `forward(x1,x2)->{"loss"}` / `.backbone` contract, config→`Namespace` flow, `get_aug` keying on `model.name`, and notebook entry via `colab_utils.train_from_colab` actually match the code? Flag anything inaccurate.
2. **Loss-function fidelity** — are the VICReg (invariance/variance/covariance, coeffs 25/25/1) and Barlow Twins (cross-correlation, λ=0.0051) formulas and default hyperparameters correct per the original papers?
3. **Completeness/gaps** — is anything missing for a clean implementation (e.g. seeded N-subset handling, feature extraction from the penultimate layer vs projector, linear-eval on/off, optimizer parity with existing baselines)?
4. **Test plan adequacy** — are the proposed tests sufficient and genuinely TDD-ordered? What edge cases are missing?
5. **Scope discipline** — does the design stay additive (no edits to the training loop / existing models beyond registration hooks)? Call out scope creep.
6. **Risks** — anything likely to break, be flaky on CPU, or diverge from the reference notebook structure.

Give concrete, file-specific feedback and concrete edits where the design is wrong or underspecified.
