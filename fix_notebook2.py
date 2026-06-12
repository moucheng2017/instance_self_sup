import json
from pathlib import Path

cells = []

# Cell 1: Colab badge
cells.append({
    "cell_type": "markdown",
    "id": "c120fdec",
    "metadata": {},
    "source": [
        '<a href="https://colab.research.google.com/github/moucheng2017/my_simsiam/blob/53-recursive-self-labelling-sigmoid-loss/notebooks/hierarchical-balanced-vmf-cifar10-ssl.ipynb" target="_parent">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
    ]
})

# Cell 2: Description
cells.append({
    "cell_type": "markdown",
    "id": "c9fa8429",
    "metadata": {},
    "source": [
        "# Hierarchical Balanced vMF Self-Labeling on Google Colab\n",
        "\n",
        "This notebook follows the existing Colab flow and trains the current hierarchical balanced vMF self-labeling experiment.\n",
        "\n",
        "Images are encoded into L2-normalized embeddings, batch-local OT/vMF labels are fitted from detached features, and the model predicts the discovered hierarchical path with optional sigmoid image-index regularization.\n",
        "\n",
        "If the GitHub repository is private, create a GitHub fine-grained personal access token with repository read access and paste it when prompted in the setup cell."
    ]
})

# Cell 3: Drive mount
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "93e0cf35",
    "metadata": {},
    "outputs": [],
    "source": [
        "try:\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        "except ModuleNotFoundError:\n",
        "    print('google.colab is only available inside Google Colab; skipping Drive mount.')\n"
    ]
})

# Cell 4: Repo setup
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "2c5f17e1",
    "metadata": {},
    "outputs": [],
    "source": [
        "import getpass\n",
        "import os\n",
        "import subprocess\n",
        "from pathlib import Path\n",
        "from urllib.parse import quote\n",
        "\n",
        "PUBLIC_REPO_URL = 'https://github.com/moucheng2017/my_simsiam.git'\n",
        "BRANCH = '53-recursive-self-labelling-sigmoid-loss'\n",
        "REPO_DIR = Path('/content/my_simsiam')\n",
        "GITHUB_TOKEN = \"\"  # Set to a token string here if the repo is private.\n",
        "\n",
        "\n",
        "def run(cmd, cwd=None):\n",
        "    print('+', ' '.join(cmd))\n",
        "    subprocess.run(cmd, cwd=cwd, check=True)\n",
        "\n",
        "\n",
        "try:\n",
        "    from google.colab import userdata\n",
        "    GITHUB_TOKEN = GITHUB_TOKEN or userdata.get('GITHUB_TOKEN')\n",
        "except Exception:\n",
        "    pass\n",
        "\n",
        "repo_url = PUBLIC_REPO_URL\n",
        "if GITHUB_TOKEN:\n",
        "    repo_url = PUBLIC_REPO_URL.replace('https://', f'https://oauth2:{quote(GITHUB_TOKEN, safe=\"\")}@')\n",
        "\n",
        "\n",
        "def sync_repo():\n",
        "    if GITHUB_TOKEN:\n",
        "        run(['git', 'remote', 'set-url', 'origin', repo_url], cwd=REPO_DIR)\n",
        "    else:\n",
        "        print('No GitHub token found; keeping the existing origin URL for fetch/pull.')\n",
        "    run(['git', 'fetch', 'origin', BRANCH], cwd=REPO_DIR)\n",
        "    run(['git', 'checkout', BRANCH], cwd=REPO_DIR)\n",
        "    run(['git', 'pull', '--ff-only', 'origin', BRANCH], cwd=REPO_DIR)\n",
        "\n",
        "os.chdir('/content')\n",
        "\n",
        "if not REPO_DIR.exists():\n",
        "    try:\n",
        "        run(['git', 'clone', '--branch', BRANCH, repo_url, str(REPO_DIR)])\n",
        "    except subprocess.CalledProcessError as exc:\n",
        "        if not GITHUB_TOKEN:\n",
        "            print('\\nClone failed without a token. If this repository is private, paste a GitHub token with read access.')\n",
        "            token = getpass.getpass('GitHub token (leave blank to stop): ').strip()\n",
        "            if token:\n",
        "                repo_url = PUBLIC_REPO_URL.replace('https://', f'https://oauth2:{quote(token, safe=\"\")}@')\n",
        "                run(['git', 'clone', '--branch', BRANCH, repo_url, str(REPO_DIR)])\n",
        "            else:\n",
        "                raise RuntimeError('Repository clone was skipped because no token was provided.') from exc\n",
        "        else:\n",
        "            raise\n",
        "else:\n",
        "    print(f'Repo already exists at {REPO_DIR}; pulling latest changes from {BRANCH}.')\n",
        "    try:\n",
        "        sync_repo()\n",
        "    except subprocess.CalledProcessError as exc:\n",
        "        raise RuntimeError(\n",
        "            'Failed to fetch the latest repo changes. If the repository is private, make sure a GITHUB_TOKEN Colab secret is set or rerun after deleting /content/my_simsiam so the clone step can prompt for a token.'\n",
        "        ) from exc\n",
        "\n",
        "if not REPO_DIR.exists():\n",
        "    raise FileNotFoundError(f'Expected repository at {REPO_DIR}, but it was not created.')\n",
        "\n",
        "os.chdir(REPO_DIR)\n",
        "print('Working directory:', os.getcwd())\n",
        "run(['pip', 'install', '-r', 'requirements_colab.txt'], cwd=REPO_DIR)\n"
    ]
})

# Cell 5: Training
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "5f8519c6",
    "metadata": {},
    "outputs": [],
    "source": [
        "from colab_utils import train_from_colab\n",
        "\n",
        "hierarchical_balanced_vmf_result = train_from_colab(\n",
        "    config_file='configs/hierarchical_balanced_vmf_cifar_colab.yaml',\n",
        "    project_name='SSL_exps',\n",
        "    use_drive=True,\n",
        "    device='cuda',\n",
        "    download=True,\n",
        "    overrides={\n",
        "        'train': {\n",
        "            'batch_size': 512,\n",
        "            'source_pool_size': 50000,\n",
        "            'source_subset_seed': 0,\n",
        "            'num_epochs': 100,\n",
        "            'stop_at_epoch': 100\n",
        "        }\n",
        "    },\n",
        ")\n",
        "hierarchical_balanced_vmf_result\n"
    ]
})

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

path = str(Path("notebooks") / "hierarchical-balanced-vmf-cifar10-ssl-generated.ipynb")
with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

with open(path) as f:
    json.load(f)
print("SUCCESS: Valid JSON notebook written.")
