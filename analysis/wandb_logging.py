"""Optional Weights & Biases logging for the Section 2.1 diagnostics.

Core helper only: import and call this from your diagnostics cell/code. It degrades
gracefully (prints a message and returns None) when `wandb` is unavailable or a run
cannot be created, so it never breaks the analysis.

Typical use from a notebook diagnostics cell::

    from analysis.wandb_logging import log_section21_diagnostics

    diagnostics_table = run_section_21_diagnostics(CONFIG_FILE, results)
    log_section21_diagnostics(
        diagnostics_table,
        figures={"singular_values": "/path/to/section21_top20_singular_values_epoch100.png"},
        project="instance_self_sup",
        run_name="vicreg-section21-diagnostics",
    )
"""


def _import_wandb():
    try:
        import wandb

        return wandb
    except ImportError:
        print(
            "[wandb_logging] 'wandb' is not installed; skipping W&B diagnostics logging. "
            "Install with `pip install wandb`."
        )
        return None


def log_section21_diagnostics(
    diagnostics_table,
    figures=None,
    project="instance_self_sup",
    run_name="section21-diagnostics",
    entity=None,
    config=None,
    reuse_active_run=True,
):
    """Log the Section 2.1 diagnostics table (and optional figures) to W&B.

    Args:
        diagnostics_table: a pandas DataFrame, e.g. with columns
            ``N, epoch, effective_rank, knn_accuracy, ...`` (one row per checkpoint).
        figures: optional ``{name: figure}`` where each figure is a matplotlib
            Figure, a PIL image, or a path to a saved image file.
        project / run_name / entity / config: standard ``wandb.init`` arguments.
        reuse_active_run: if True and a W&B run is already active, log into it
            instead of starting a new one.

    Returns the active ``wandb`` run, or None if W&B was unavailable.
    """
    wandb = _import_wandb()
    if wandb is None:
        return None

    try:
        if reuse_active_run and getattr(wandb, "run", None) is not None:
            run = wandb.run
        else:
            run = wandb.init(
                project=project, entity=entity, name=run_name, config=config, reinit=True
            )
    except Exception as exc:
        print(f"[wandb_logging] could not start a W&B run ({exc}); skipping.")
        return None

    # Full table for interactive inspection in the W&B UI.
    try:
        wandb.log({"section21/diagnostics": wandb.Table(dataframe=diagnostics_table)})
    except Exception as exc:
        print(f"[wandb_logging] could not log diagnostics table ({exc}).")

    # Per-row scalars so effective rank / KNN are plottable vs N and epoch.
    try:
        for _, row in diagnostics_table.iterrows():
            payload = {}
            for column in ("N", "epoch", "effective_rank", "knn_accuracy", "monitor_accuracy"):
                if column in diagnostics_table.columns and row[column] is not None:
                    payload[f"section21/{column}"] = row[column]
            if payload:
                wandb.log(payload)
    except Exception as exc:
        print(f"[wandb_logging] could not log per-row scalars ({exc}).")

    if figures:
        for name, figure in figures.items():
            try:
                wandb.log({f"section21/{name}": wandb.Image(figure)})
            except Exception as exc:
                print(f"[wandb_logging] could not log figure '{name}' ({exc}).")

    return run
