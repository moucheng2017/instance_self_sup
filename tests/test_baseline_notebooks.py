import ast
import json


NOTEBOOKS = {
    "pseudo_sup": ("notebooks/pseudo-sup-cifar10-ssl.ipynb", "configs/baselines/pseudo_sup_cifar10.yaml"),
    "vicreg": ("notebooks/vicreg-cifar10-ssl.ipynb", "configs/baselines/vicreg_cifar10.yaml"),
    "barlow_twins": ("notebooks/barlow-twins-cifar10-ssl.ipynb", "configs/baselines/barlow_twins_cifar10.yaml"),
    "simclr": ("notebooks/simclr-cifar10-ssl.ipynb", "configs/baselines/simclr_cifar10.yaml"),
    "simsiam": ("notebooks/simsiam-cifar10-ssl.ipynb", "configs/baselines/simsiam_cifar10.yaml"),
    "swav": ("notebooks/swav-cifar10-ssl.ipynb", "configs/baselines/swav_cifar10.yaml"),
}


def _source(path):
    with open(path, "r") as f:
        notebook = json.load(f)
    code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    return notebook, "\n".join(code_cells), code_cells


def test_baseline_notebook_code_cells_parse():
    for path, _ in NOTEBOOKS.values():
        _, _, code_cells = _source(path)
        for source in code_cells:
            ast.parse(source)


def test_baseline_notebooks_expose_required_training_and_diagnostic_controls():
    required = [
        "from colab_utils import train_from_colab",
        "N_SWEEP",
        "subset_n",
        "subset_seed",
        "INIT_CHECKPOINT",
        "INIT_LOAD_BACKBONE",
        "INIT_LOAD_PROJECTOR",
        "INIT_LOAD_PREDICTOR",
        "selected_subset_indices_path",
        "selected_indices_for_result",
        "monitor_accuracy",
        "batch_size_for_n",
        "build_diagnostics_loader",
        "extract_features",
        "spectral_diagnostics",
        "knn_eval",
    ]
    for _, (path, config_path) in NOTEBOOKS.items():
        _, source, _ = _source(path)
        assert f"config_file='{config_path}'" in source
        for needle in required:
            assert needle in source


def test_pseudo_sup_notebook_carries_single_episode_settings():
    _, source, _ = _source(NOTEBOOKS["pseudo_sup"][0])
    assert "pseudo_sup_cifar10.yaml" in source
    assert "COSINE_SOFTMAX = True" in source
    assert "L2_NORM_BACKBONE_FEATURES = True" in source
    assert "NEGATIVES_RATIO = 0.25" in source
    assert "NUM_ITERATIONS_PER_SAMPLE = 20" in source
    assert "samples_per_epoch_for_n" in source
    assert "'samples_per_epoch': samples_per_epoch" in source
