import torch


_SUBMODULE_ALIASES = {
    "backbone": ("backbone",),
    "projector": ("projector", "proj"),
    "predictor": ("predictor", "pred"),
}


def _find_submodule(model, logical_name):
    for attr in _SUBMODULE_ALIASES[logical_name]:
        module = getattr(model, attr, None)
        if module is not None:
            return attr, module
    return None, None


def _extract_prefixed_state(state_dict, prefix):
    dotted = prefix + "."
    return {key[len(dotted):]: value for key, value in state_dict.items() if key.startswith(dotted)}


def _load_requested_submodule(model, state_dict, logical_name):
    attr, module = _find_submodule(model, logical_name)
    if module is None:
        raise ValueError(f"Requested {logical_name} init, but target model has no {logical_name} submodule.")

    candidate_states = []
    for source_attr in _SUBMODULE_ALIASES[logical_name]:
        sub_state = _extract_prefixed_state(state_dict, source_attr)
        if sub_state:
            candidate_states.append((source_attr, sub_state))
    if not candidate_states:
        raise ValueError(f"Checkpoint has no weights for requested {logical_name} submodule.")

    target_state = module.state_dict()
    mismatch_messages = []
    for source_attr, sub_state in candidate_states:
        target_keys = set(target_state)
        source_keys = set(sub_state)
        if target_keys != source_keys:
            missing = sorted(target_keys - source_keys)[:5]
            unexpected = sorted(source_keys - target_keys)[:5]
            mismatch_messages.append(
                f"{source_attr}: key mismatch missing={missing} unexpected={unexpected}"
            )
            continue
        bad_shapes = [
            key
            for key in target_keys
            if tuple(target_state[key].shape) != tuple(sub_state[key].shape)
        ]
        if bad_shapes:
            preview = ", ".join(bad_shapes[:5])
            mismatch_messages.append(f"{source_attr}: shape mismatch for {preview}")
            continue
        module.load_state_dict(sub_state, strict=True)
        return attr, source_attr

    detail = "; ".join(mismatch_messages)
    raise ValueError(f"Could not load requested {logical_name} weights: {detail}")


def load_init_weights(
    model,
    checkpoint_path,
    load_backbone=True,
    load_projector=False,
    load_predictor=False,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    loaded = []
    requested = {
        "backbone": load_backbone,
        "projector": load_projector,
        "predictor": load_predictor,
    }
    for logical_name, should_load in requested.items():
        if should_load:
            target_attr, source_attr = _load_requested_submodule(model, state_dict, logical_name)
            loaded.append(f"{logical_name} ({source_attr} -> {target_attr})")

    print(f"Loaded init weights from {checkpoint_path}: {', '.join(loaded) if loaded else 'none'}")
    return loaded
