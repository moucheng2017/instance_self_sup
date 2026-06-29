try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from torch import Tensor
from collections import OrderedDict
import os
from .plotter import Plotter


class Logger(object):
    def __init__(
        self,
        log_dir,
        tensorboard=True,
        matplotlib=True,
        wandb=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_config=None,
    ):

        self.reset(
            log_dir,
            tensorboard,
            matplotlib,
            wandb,
            wandb_project,
            wandb_entity,
            wandb_run_name,
            wandb_config,
        )

    def reset(
        self,
        log_dir=None,
        tensorboard=True,
        matplotlib=True,
        wandb=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_config=None,
    ):

        if log_dir is not None: self.log_dir=log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir) if tensorboard else None
        self.plotter = Plotter() if matplotlib else None
        self.counter = OrderedDict()
        # Weights & Biases backend (optional). Holds the imported module when an
        # active run exists, otherwise None. Failures here never stop training.
        self._wandb = self._init_wandb(
            wandb, wandb_project, wandb_entity, wandb_run_name, wandb_config
        )

    def _init_wandb(self, enabled, project, entity, run_name, config):
        if not enabled:
            return None
        try:
            import wandb
        except ImportError:
            print(
                "[Logger] W&B logging requested but the 'wandb' package is not "
                "installed; continuing without it. Install with `pip install wandb`."
            )
            return None
        try:
            run = wandb.init(
                project=project or "instance_self_sup",
                entity=entity,
                name=run_name,
                config=config,
                reinit=True,
            )
            print(
                f"[Logger] W&B logging enabled "
                f"(project={project or 'instance_self_sup'}, run={getattr(run, 'name', run_name)})."
            )
            return wandb
        except Exception as exc:  # login/network/other issues must not stop training
            print(f"[Logger] Failed to initialize W&B ({exc}); continuing without it.")
            return None

    def update_scalers(self, ordered_dict):

        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            if self.counter.get(key) is None:
                self.counter[key] = 1
            else:
                self.counter[key] += 1

            if self.writer:
                self.writer.add_scalar(key, value, self.counter[key])

        if self._wandb is not None:
            try:
                self._wandb.log({key: value for key, value in ordered_dict.items()})
            except Exception as exc:
                print(f"[Logger] W&B log failed ({exc}); disabling W&B for this run.")
                self._wandb = None

        if self.plotter:
            self.plotter.update(ordered_dict)
            self.plotter.save(os.path.join(self.log_dir, 'plotter.svg'))

    def finish(self):
        """Flush/close backends. Safe to call multiple times."""
        if self.writer is not None:
            try:
                self.writer.flush()
            except Exception:
                pass
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
            self._wandb = None
