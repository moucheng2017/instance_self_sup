import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_active_depth(epoch, depth, epochs_per_level, initial_depth=1):
    """Staircase depth-annealing schedule.

    Starting from ``initial_depth`` active tree levels, one extra level is
    activated every ``epochs_per_level`` epochs, capped at ``depth``.
    ``epochs_per_level=None`` disables annealing (all levels active).
    """
    if epochs_per_level is None:
        return int(depth)
    epochs_per_level = float(epochs_per_level)
    if epochs_per_level <= 0:
        raise ValueError("epochs_per_level must be positive or None.")
    initial_depth = int(initial_depth)
    if not 1 <= initial_depth <= int(depth):
        raise ValueError(f"initial_depth must be in [1, {int(depth)}], got {initial_depth}.")
    active = initial_depth + int(float(epoch) // epochs_per_level)
    return max(1, min(int(depth), active))


def compute_unbalanced_tau(epoch, tau_start, tau_final, anneal_epochs):
    """Cosine schedule for the unbalanced-OT marginal penalty strength.

    ``tau`` shares ``ot_epsilon``'s cost units: tau >> ot_epsilon behaves like
    hard balance, the balanced->natural transition happens around
    tau ~ ot_epsilon, and tau << ot_epsilon leaves component masses nearly
    free. Anneals from ``tau_start`` (epoch 0) to ``tau_final``
    (epoch >= anneal_epochs); ``anneal_epochs`` in (None, 0) returns
    ``tau_final`` immediately.
    """
    tau_final = float(tau_final)
    if tau_final <= 0:
        raise ValueError("tau_final must be positive.")
    if anneal_epochs is None or float(anneal_epochs) <= 0:
        return tau_final
    tau_start = float(tau_start)
    if tau_start <= 0:
        raise ValueError("tau_start must be positive.")
    progress = min(1.0, max(0.0, float(epoch) / float(anneal_epochs)))
    return tau_final + 0.5 * (tau_start - tau_final) * (1.0 + math.cos(math.pi * progress))


def select_reseed_indices(streaks, last_accs, patience, budget):
    """Pick the worst-accuracy nodes whose low-acc streak reached patience.

    Pure-Python selection so the budget logic is testable in isolation.
    Returns at most ``budget`` node indices, worst last-accuracy first.
    """
    candidates = [i for i, streak in enumerate(streaks) if streak >= patience]
    candidates.sort(key=lambda i: last_accs[i])
    return candidates[: max(int(budget), 0)]


def _subtree_node_ids(node_id, num_internal_nodes):
    """All internal-node ids in the subtree rooted at node_id (inclusive)."""
    ids, stack = [], [int(node_id)]
    while stack:
        nid = stack.pop()
        if nid >= int(num_internal_nodes):
            continue
        ids.append(nid)
        stack.extend((2 * nid + 1, 2 * nid + 2))
    return ids


def _balanced_initial_paths(num_classes, depth):
    num_leaves = 2 ** depth
    leaf_ids = (torch.arange(num_classes, dtype=torch.long) * num_leaves) // num_classes
    shifts = torch.arange(depth - 1, -1, -1, dtype=torch.long)
    return ((leaf_ids[:, None] >> shifts[None, :]) & 1).long()


def _path_node_ids(paths):
    num_classes, depth = paths.shape
    nodes = torch.zeros(num_classes, depth, dtype=torch.long)
    current = torch.zeros(num_classes, dtype=torch.long)
    for level in range(depth):
        nodes[:, level] = current
        current = current * 2 + 1 + paths[:, level]
    return nodes


class HierarchicalBalancedVMFSelfLabelingNet(nn.Module):
    """Hierarchical balanced vMF self-labeling on the unit sphere.

    A tree refresh pass extracts clean source-pool features, repeatedly solves
    local balanced 2-way entropy-regularized OT problems on the sphere, and
    stores the resulting binary path for each source image. Training then asks
    each augmented image to predict its current path by aligning to the vMF
    child means at every visited tree node.
    """

    def __init__(
        self,
        num_classes,
        backbone=None,
        embedding_dim=None,
        depth=8,
        kappa=20.0,
        ot_epsilon=0.05,
        sinkhorn_iters=3,
        em_iters=5,
        ot_unbalanced_tau=None,
        ot_unbalanced_min_split_fraction=0.05,
        ot_assignment_mode="hierarchical_vmf_ot",
        flat_vmf_num_components=None,
        batch_self_labeling=True,
        tree_warm_start=False,
        reseed_acc_threshold=None,
        reseed_patience=3,
        reseed_min_node_samples=64,
        reseed_enabled=False,
        reseed_budget_fraction=0.1,
        learnable_vmf_prototypes=False,
        prototype_ema_momentum=None,
        sigmoid_regularization_weight=0.0,
        sigmoid_init_temperature=1.0,
        sigmoid_init_bias=-10.0,
    ):
        super().__init__()
        if backbone is None:
            raise ValueError("backbone must be provided explicitly.")
        if num_classes is None or int(num_classes) < 2:
            raise ValueError("HierarchicalBalancedVMFSelfLabelingNet requires at least 2 pseudo classes.")

        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim or backbone.output_dim)
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("depth must be at least 1.")
        if self.depth > 16:
            raise ValueError("depth above 16 creates a very large binary tree; use a smaller depth.")

        # Depth annealing: only the first `active_depth` tree levels contribute
        # to the prediction loss. Defaults to full depth (annealing disabled).
        # Plain int (not a buffer) so checkpoints keep their existing keys; the
        # trainer re-derives it from the epoch schedule every epoch.
        self.active_depth = self.depth

        self.kappa = float(kappa)
        self.ot_epsilon = float(ot_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.em_iters = int(em_iters)
        self.ot_assignment_mode = str(ot_assignment_mode)
        self.flat_vmf_num_components = None if flat_vmf_num_components is None else int(flat_vmf_num_components)
        self.batch_self_labeling = bool(batch_self_labeling)
        # Unbalanced OT (semi-relaxed Sinkhorn): None keeps the exact balanced
        # transport (legacy). A float relaxes the component marginal with a KL
        # penalty of that strength; tau shares ot_epsilon's cost units and the
        # balanced->natural transition happens around tau ~ ot_epsilon.
        self.ot_unbalanced_tau = None if ot_unbalanced_tau is None else float(ot_unbalanced_tau)
        self.ot_unbalanced_min_split_fraction = float(ot_unbalanced_min_split_fraction)
        if self.ot_unbalanced_tau is not None and self.ot_unbalanced_tau <= 0:
            raise ValueError("ot_unbalanced_tau must be positive or null.")
        if not 0.0 <= self.ot_unbalanced_min_split_fraction < 0.5:
            raise ValueError("ot_unbalanced_min_split_fraction must be in [0, 0.5).")
        # Warm start: initialize each node's 2-way EM from the node's stored
        # prototypes (temporal continuity) instead of re-deriving seeds from
        # the data every refresh. Nodes never fitted before stay cold-seeded.
        self.tree_warm_start = bool(tree_warm_start)
        # Node-health bookkeeping (selective re-seeding, logging stage).
        # reseed_acc_threshold=None disables streak tracking entirely.
        self.reseed_acc_threshold = None if reseed_acc_threshold is None else float(reseed_acc_threshold)
        self.reseed_patience = int(reseed_patience)
        self.reseed_min_node_samples = int(reseed_min_node_samples)
        if self.reseed_acc_threshold is not None and not 0.0 < self.reseed_acc_threshold < 1.0:
            raise ValueError("reseed_acc_threshold must be in (0, 1) or null.")
        if self.reseed_patience < 1:
            raise ValueError("reseed_patience must be at least 1.")
        if self.reseed_min_node_samples < 1:
            raise ValueError("reseed_min_node_samples must be at least 1.")
        # Live trigger (stage 3): cold-restart flagged nodes at the next tree
        # build. Requires reseed_acc_threshold to be set.
        self.reseed_enabled = bool(reseed_enabled)
        self.reseed_budget_fraction = float(reseed_budget_fraction)
        if self.reseed_enabled and self.reseed_acc_threshold is None:
            raise ValueError("reseed_enabled=True requires reseed_acc_threshold to be set.")
        if self.reseed_enabled and not self.tree_warm_start:
            raise ValueError(
                "reseed_enabled=True requires tree_warm_start=True: without warm "
                "start every node is cold-restarted at every build anyway, so "
                "selective re-seeding is meaningless."
            )
        if not 0.0 < self.reseed_budget_fraction <= 1.0:
            raise ValueError("reseed_budget_fraction must be in (0, 1].")
        self.learnable_vmf_prototypes = bool(learnable_vmf_prototypes)
        self.prototype_ema_momentum = None if prototype_ema_momentum is None else float(prototype_ema_momentum)
        self.sigmoid_regularization_max_weight = float(sigmoid_regularization_weight)
        if self.ot_assignment_mode == "hierachical_vmf":
            self.ot_assignment_mode = "hierarchical_vmf"
        valid_modes = ("hierarchical_vmf_ot", "hierarchical_vmf", "recursive_ot", "flat_ot", "flat_vmf_ot", "vmf")
        if self.ot_assignment_mode not in valid_modes:
            raise ValueError(f"ot_assignment_mode must be one of {valid_modes}.")
        if self.flat_vmf_num_components is not None and self.flat_vmf_num_components < 2:
            raise ValueError("flat_vmf_num_components must be at least 2 when set.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.ot_epsilon <= 0:
            raise ValueError("ot_epsilon must be positive.")
        if self.sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be at least 1.")
        if self.em_iters < 1:
            raise ValueError("em_iters must be at least 1.")
        if self.prototype_ema_momentum is not None and not 0.0 <= self.prototype_ema_momentum < 1.0:
            raise ValueError("prototype_ema_momentum must be in [0, 1) or null.")
        if self.learnable_vmf_prototypes and self.prototype_ema_momentum is not None:
            raise ValueError(
                "learnable_vmf_prototypes and prototype_ema_momentum cannot both be set. "
                "With learnable prototypes the optimizer updates them via gradients; "
                "the EMA write would overwrite the gradient step every forward pass."
            )
        if (
            self.batch_self_labeling
            and self.tree_warm_start
            and not self.learnable_vmf_prototypes
            and self.prototype_ema_momentum is None
        ):
            raise ValueError(
                "batch_self_labeling=True with tree_warm_start=True requires either "
                "prototype_ema_momentum to persist fitted batch prototypes or "
                "learnable_vmf_prototypes=True. Otherwise warm start would reuse "
                "stale stored prototypes after the first batch."
            )
        if self.sigmoid_regularization_max_weight < 0:
            raise ValueError("sigmoid_regularization_weight must be non-negative.")
        if sigmoid_init_temperature <= 0:
            raise ValueError("sigmoid_init_temperature must be positive.")

        if self.embedding_dim == backbone.output_dim:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Linear(backbone.output_dim, self.embedding_dim)

        num_internal_nodes = 2 ** self.depth - 1
        num_flat_prototypes = (
            self.flat_vmf_num_components
            if self.ot_assignment_mode == "flat_vmf_ot" and self.flat_vmf_num_components is not None
            else 2 ** self.depth
        )
        self.num_flat_prototypes = num_flat_prototypes
        prototypes = torch.randn(num_internal_nodes, 2, self.embedding_dim)
        prototypes = F.normalize(prototypes, dim=-1)
        flat_prototypes = torch.randn(num_flat_prototypes, self.embedding_dim)
        flat_prototypes = F.normalize(flat_prototypes, dim=-1)
        paths = _balanced_initial_paths(self.num_classes, self.depth)
        path_masks = torch.ones(self.num_classes, self.depth, dtype=torch.bool)
        flat_labels = (torch.arange(self.num_classes, dtype=torch.long) * num_flat_prototypes) // self.num_classes
        flat_probs = F.one_hot(flat_labels, num_classes=num_flat_prototypes).float()

        if self.learnable_vmf_prototypes:
            self.node_prototypes = nn.Parameter(prototypes)
            self.flat_prototypes = nn.Parameter(flat_prototypes)
        else:
            self.register_buffer("node_prototypes", prototypes)
            self.register_buffer("flat_prototypes", flat_prototypes)
        # True once a node's 2-way split has been fitted at least once; only
        # fitted nodes are eligible for warm starting.
        self.register_buffer("node_fitted", torch.zeros(num_internal_nodes, dtype=torch.bool))
        # Per-node branch-accuracy bookkeeping, accumulated between calls to
        # finalize_node_stats(). node_levels maps node id -> tree level.
        self.register_buffer("node_correct_sum", torch.zeros(num_internal_nodes))
        self.register_buffer("node_count", torch.zeros(num_internal_nodes))
        self.register_buffer("node_low_acc_streak", torch.zeros(num_internal_nodes, dtype=torch.long))
        self.register_buffer("node_last_acc", torch.full((num_internal_nodes,), -1.0))
        self.register_buffer(
            "node_levels",
            (torch.arange(num_internal_nodes, dtype=torch.double) + 1).log2().floor().long(),
        )
        self.register_buffer("assignment_paths", paths)
        self.register_buffer("assignment_node_ids", _path_node_ids(paths))
        self.register_buffer("assignment_masks", path_masks)
        self.register_buffer("flat_assignment_labels", flat_labels)
        self.register_buffer("flat_assignment_probs", flat_probs)
        self.register_buffer("sigmoid_regularization_current_weight", torch.zeros([]))

        self.sigmoid_label_embeddings = nn.Embedding(self.num_classes, self.embedding_dim)
        nn.init.normal_(self.sigmoid_label_embeddings.weight, std=0.02)
        self.sigmoid_logit_scale = nn.Parameter(torch.ones([]) * math.log(float(sigmoid_init_temperature)))
        self.sigmoid_logit_bias = nn.Parameter(torch.ones([]) * float(sigmoid_init_bias))

    def _uses_spherical_geometry(self):
        return self.ot_assignment_mode in ("hierarchical_vmf_ot", "hierarchical_vmf", "flat_vmf_ot", "vmf")

    def _uses_flat_assignments(self):
        return self.ot_assignment_mode in ("flat_ot", "flat_vmf_ot", "vmf")

    def _prepare_ot_prototypes(self, prototypes):
        if self._uses_spherical_geometry():
            return F.normalize(prototypes, dim=-1)
        return prototypes

    def _prepared_node_prototypes(self):
        return self._prepare_ot_prototypes(self.node_prototypes)

    def _prepared_flat_prototypes(self):
        return self._prepare_ot_prototypes(self.flat_prototypes)

    @torch.no_grad()
    def _set_node_prototypes(self, prototypes):
        prototypes = self._prepare_ot_prototypes(prototypes.to(self.node_prototypes.device))
        self.node_prototypes.copy_(prototypes)

    @torch.no_grad()
    def _set_flat_prototypes(self, prototypes):
        prototypes = self._prepare_ot_prototypes(prototypes.to(self.flat_prototypes.device))
        self.flat_prototypes.copy_(prototypes)

    @torch.no_grad()
    def _ema_update_node_prototypes(self, batch_prototypes):
        if self.prototype_ema_momentum is None:
            return
        momentum = self.prototype_ema_momentum
        current = self._prepared_node_prototypes().detach()
        updated = momentum * current + (1.0 - momentum) * batch_prototypes.to(self.node_prototypes.device)
        self._set_node_prototypes(updated)

    @torch.no_grad()
    def _ema_update_flat_prototypes(self, batch_prototypes):
        if self.prototype_ema_momentum is None:
            return
        momentum = self.prototype_ema_momentum
        current = self._prepared_flat_prototypes().detach()
        updated = momentum * current + (1.0 - momentum) * batch_prototypes.to(self.flat_prototypes.device)
        self._set_flat_prototypes(updated)

    def set_ot_unbalanced_tau(self, tau):
        """Set the unbalanced-OT marginal penalty strength (None = balanced).

        Called once per epoch by the trainer when the tau schedule is
        configured; tau shares ot_epsilon's cost units.
        """
        if tau is not None:
            tau = float(tau)
            if tau <= 0:
                raise ValueError("ot_unbalanced_tau must be positive or None.")
        self.ot_unbalanced_tau = tau

    def set_active_depth(self, active_depth):
        """Set how many tree levels (from the root) the prediction loss uses.

        Levels >= active_depth are not supervised: their splits exist in the
        stored/batch-built tree but produce no gradient, so the backbone is
        never trained to conform to fine-scale structure before the coarser
        levels have consolidated. Only meaningful for hierarchical modes; flat
        modes ignore it.
        """
        active_depth = int(active_depth)
        if not 1 <= active_depth <= self.depth:
            raise ValueError(f"active_depth must be in [1, {self.depth}], got {active_depth}.")
        self.active_depth = active_depth

    @torch.no_grad()
    def finalize_node_stats(self):
        """Fold the accumulated per-node branch accuracies into health stats.

        Call once per epoch (trainer). Returns per-level and overall node
        accuracy stats, updates low-accuracy streaks (only when
        reseed_acc_threshold is set), and resets the accumulators. Nodes with
        fewer than reseed_min_node_samples visits are ignored this round.
        Logging-only: nothing here mutates the tree.
        """
        if self._uses_flat_assignments():
            return {}

        counts = self.node_count
        accs = self.node_correct_sum / counts.clamp_min(1.0)
        visited = counts >= float(self.reseed_min_node_samples)
        self.node_last_acc[visited] = accs[visited]

        if self.reseed_acc_threshold is not None:
            low = visited & (accs < self.reseed_acc_threshold)
            self.node_low_acc_streak[low] += 1
            self.node_low_acc_streak[visited & ~low] = 0

        stats = {"tree_nodes_visited": float(visited.sum().item())}
        if visited.any():
            stats["tree_node_acc_overall"] = float(accs[visited].mean().item())
        for level in range(self.depth):
            level_visited = visited & (self.node_levels == level)
            if level_visited.any():
                stats[f"tree_node_acc_level{level}"] = float(accs[level_visited].mean().item())
        if self.reseed_acc_threshold is not None:
            stats["tree_reseed_candidates"] = float(
                (self.node_low_acc_streak >= self.reseed_patience).sum().item()
            )

        self.node_correct_sum.zero_()
        self.node_count.zero_()
        return stats

    @torch.no_grad()
    def select_reseed_nodes(self):
        """Cold-restart the worst persistently-unlearnable nodes (live trigger).

        Call once per epoch, after finalize_node_stats(). Nodes whose low-acc
        streak reached reseed_patience are selected worst-accuracy-first, up
        to reseed_budget_fraction of the measured nodes per call. A selected
        node and its whole subtree get node_fitted=False (next build re-derives
        them from data) and their stats/streaks reset — their history refers to
        a partition that no longer exists. Returns the number of re-seeded
        subtree roots.
        """
        if not self.reseed_enabled or self._uses_flat_assignments():
            return 0

        measured = int((self.node_last_acc >= 0.0).sum().item())
        if measured == 0:
            return 0
        budget = max(1, int(self.reseed_budget_fraction * measured))
        selected = select_reseed_indices(
            self.node_low_acc_streak.tolist(),
            self.node_last_acc.tolist(),
            self.reseed_patience,
            budget,
        )
        num_internal_nodes = self.node_fitted.shape[0]
        for node_id in selected:
            for nid in _subtree_node_ids(node_id, num_internal_nodes):
                self.node_fitted[nid] = False
                self.node_low_acc_streak[nid] = 0
                self.node_last_acc[nid] = -1.0
                self.node_correct_sum[nid] = 0.0
                self.node_count[nid] = 0.0
        return len(selected)

    def set_sigmoid_regularization_progress(self, current_step, rampup_steps):
        if self.sigmoid_regularization_max_weight <= 0:
            weight = 0.0
        elif rampup_steps <= 0:
            weight = self.sigmoid_regularization_max_weight
        else:
            progress = min(1.0, float(current_step) / float(rampup_steps))
            weight = self.sigmoid_regularization_max_weight * progress
        self.sigmoid_regularization_current_weight.fill_(weight)

    def encode(self, images):
        embeddings = self.projector(self.backbone(images))
        return F.normalize(embeddings, dim=1)

    # Minimum number of Sinkhorn iterations used for flat K-way OT.  The binary
    # splits used by the hierarchical modes converge in just a few steps, but a
    # K-way problem (K = 2 ** depth, e.g. 64) needs many more iterations to
    # reach balanced marginals.
    _FLAT_SINKHORN_MIN_ITERS: int = 50

    def _sinkhorn(self, scores, num_iters=None):
        iters = self.sinkhorn_iters if num_iters is None else int(num_iters)
        q = torch.exp(scores - scores.max(dim=1, keepdim=True).values).t()
        q = q / q.sum().clamp_min(1e-12)
        num_components, batch_size = q.shape

        for _ in range(iters):
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
            q = q / num_components
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            q = q / batch_size

        return (q * batch_size).t()

    def _unbalanced_sinkhorn(self, scores, tau, num_iters=None, target=None):
        """Semi-relaxed unbalanced Sinkhorn (scaling form, Chizat et al. 2018).

        The sample marginal stays hard (every returned row sums to 1); the
        component marginal is only KL-penalized with strength ``tau``, applied
        as a partial correction with exponent phi = tau / (tau + ot_epsilon)
        (scores are already cost / ot_epsilon, so tau shares ot_epsilon's
        units). tau -> inf recovers `_sinkhorn` exactly; tau << ot_epsilon
        lets component masses follow the data's natural ratio.
        """
        iters = self.sinkhorn_iters if num_iters is None else int(num_iters)
        phi = float(tau) / (float(tau) + max(self.ot_epsilon, 1e-6))
        q = torch.exp(scores - scores.max(dim=1, keepdim=True).values).t()
        q = q / q.sum().clamp_min(1e-12)
        num_components, batch_size = q.shape
        if target is None:
            target = q.new_full((num_components, 1), 1.0 / num_components)

        for _ in range(iters):
            mass = q.sum(dim=1, keepdim=True).clamp_min(1e-12)
            q = q * (target / mass).pow(phi)
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            q = q / batch_size

        return (q * batch_size).t()

    def _sinkhorn_2way(self, scores):
        if self.ot_unbalanced_tau is not None:
            return self._unbalanced_sinkhorn(scores, self.ot_unbalanced_tau)
        return self._sinkhorn(scores)

    def _ot_scores(self, z, prototypes):
        if self._uses_spherical_geometry():
            return (F.normalize(z, dim=1) @ F.normalize(prototypes, dim=1).t()) / max(self.ot_epsilon, 1e-6)
        distances = torch.cdist(z, prototypes, p=2).pow(2)
        return -distances / max(self.ot_epsilon, 1e-6)

    def _prediction_logits(self, z, prototypes):
        if self._uses_spherical_geometry():
            return self.kappa * (F.normalize(z, dim=1) @ F.normalize(prototypes, dim=1).t())
        distances = torch.cdist(z, prototypes, p=2).pow(2)
        return -self.kappa * distances

    def _prediction_logits_per_sample(self, z, prototypes):
        if self._uses_spherical_geometry():
            return self.kappa * torch.einsum(
                "bd,bkd->bk",
                F.normalize(z, dim=1),
                F.normalize(prototypes, dim=-1),
            )
        distances = (z[:, None, :] - prototypes).pow(2).sum(dim=-1)
        return -self.kappa * distances

    def _update_prototypes_from_assignments(self, z, q):
        weighted = q.t() @ z
        if self._uses_spherical_geometry():
            return F.normalize(weighted, dim=1)
        return weighted / q.sum(dim=0, keepdim=True).t().clamp_min(1e-12)

    def _is_degenerate_pair(self, mu):
        if self._uses_spherical_geometry():
            return (mu[0] * mu[1]).sum().item() > 0.99
        return torch.norm(mu[0] - mu[1]).item() < 1e-6

    def _cold_seed_pair(self, z):
        """Data-derived farthest-point seeding (the original cold init)."""
        mean = F.normalize(z.mean(dim=0, keepdim=True), dim=1)
        first = torch.argmin((z @ mean.t()).squeeze(1))
        second = torch.argmin(z @ z[first : first + 1].t())
        return F.normalize(torch.stack([z[first], z[second]], dim=0), dim=1)

    def _fit_binary_ot(self, z, fallback_mu=None, init_mu=None):
        num_samples = z.shape[0]
        if init_mu is not None:
            mu = self._prepare_ot_prototypes(init_mu.to(z.device))
            if self._is_degenerate_pair(mu):
                mu = self._cold_seed_pair(z)
        else:
            mu = self._cold_seed_pair(z)
        if fallback_mu is not None:
            if self._uses_spherical_geometry() and (mu[0] * mu[1]).sum().item() > 0.99:
                mu = self._prepare_ot_prototypes(fallback_mu.to(z.device))
            elif not self._uses_spherical_geometry() and torch.norm(mu[0] - mu[1]).item() < 1e-6:
                mu = self._prepare_ot_prototypes(fallback_mu.to(z.device))

        q = None
        for _ in range(self.em_iters):
            scores = self._ot_scores(z, mu)
            q = self._sinkhorn_2way(scores)
            mu = self._update_prototypes_from_assignments(z, q)
            if fallback_mu is not None and self._uses_spherical_geometry() and (mu[0] * mu[1]).sum().item() > 0.99:
                mu = self._prepare_ot_prototypes(fallback_mu.to(z.device))
                scores = self._ot_scores(z, mu)
                q = self._sinkhorn_2way(scores)
                break

        hard = torch.zeros(num_samples, dtype=torch.long, device=z.device)
        if self.ot_assignment_mode == "hierarchical_vmf":
            hard = self._prediction_logits(z, mu).argmax(dim=1)
        elif self.ot_unbalanced_tau is not None:
            # Unbalanced mode: keep the transport's natural ratio instead of
            # re-imposing balance through a median split — otherwise relaxing
            # the Sinkhorn marginal would be silently undone here. Guarded:
            # if a child would starve below ot_unbalanced_min_split_fraction,
            # fall back to the balanced median split for this node.
            hard = q.argmax(dim=1)
            counts = torch.bincount(hard, minlength=2)
            min_count = max(1, int(math.floor(self.ot_unbalanced_min_split_fraction * num_samples)))
            if int(counts.min().item()) < min_count:
                hard = torch.zeros(num_samples, dtype=torch.long, device=z.device)
                order = torch.argsort(q[:, 1])
                hard[order[num_samples // 2 :]] = 1
        else:
            order = torch.argsort(q[:, 1])
            hard[order[num_samples // 2 :]] = 1
        return mu, hard

    def _vmf_scores(self, z, prototypes):
        return self.kappa * (F.normalize(z, dim=1) @ F.normalize(prototypes, dim=1).t())

    def _vmf_responsibilities(self, z, prototypes):
        return F.softmax(self._vmf_scores(z, prototypes), dim=1)

    @torch.no_grad()
    def _build_tree_from_embeddings(self, embeddings, fallback_prototypes=None):
        device = embeddings.device
        num_items = embeddings.shape[0]
        z = F.normalize(embeddings, dim=1) if self._uses_spherical_geometry() else embeddings
        paths = torch.zeros(num_items, self.depth, dtype=torch.long, device=device)
        node_ids = torch.zeros(num_items, self.depth, dtype=torch.long, device=device)
        masks = torch.zeros(num_items, self.depth, dtype=torch.bool, device=device)
        if fallback_prototypes is None:
            prototypes = torch.randn(2 ** self.depth - 1, 2, self.embedding_dim, device=device)
            prototypes = F.normalize(prototypes, dim=-1)
        else:
            prototypes = fallback_prototypes.to(device).clone()

        split_fractions = []
        level_nodes = [(0, torch.arange(num_items, device=device))]
        for level in range(self.depth):
            next_level_nodes = []
            for node_id, indices in level_nodes:
                node_ids[indices, level] = node_id
                if indices.numel() < 2:
                    continue

                use_warm = self.tree_warm_start and bool(self.node_fitted[node_id].item())
                mu, hard = self._fit_binary_ot(
                    z[indices],
                    fallback_mu=prototypes[node_id],
                    init_mu=prototypes[node_id] if use_warm else None,
                )
                self.node_fitted[node_id] = True
                prototypes[node_id] = mu
                paths[indices, level] = hard
                masks[indices, level] = True

                # Observability for unbalanced OT: minority-child fraction per
                # node (0.5 = perfectly balanced; natural ratios drift lower).
                counts = torch.bincount(hard, minlength=2).float()
                split_fractions.append((counts.min() / counts.sum().clamp_min(1.0)).item())

                left = indices[hard == 0]
                right = indices[hard == 1]
                if level + 1 < self.depth:
                    next_level_nodes.append((2 * node_id + 1, left))
                    next_level_nodes.append((2 * node_id + 2, right))
            level_nodes = next_level_nodes

        leaf_ids = torch.zeros(num_items, dtype=torch.long, device=device)
        for level in range(self.depth):
            leaf_ids = leaf_ids * 2 + paths[:, level]
        leaf_counts = torch.bincount(leaf_ids, minlength=2 ** self.depth).float()
        nonempty = leaf_counts[leaf_counts > 0]
        stats = {
            "tree_nonempty_leaves": int(nonempty.numel()),
            "tree_min_leaf_count": int(nonempty.min().item()) if nonempty.numel() else 0,
            "tree_max_leaf_count": int(nonempty.max().item()) if nonempty.numel() else 0,
            "tree_min_split_fraction": round(min(split_fractions), 4) if split_fractions else 0.0,
            "tree_mean_split_fraction": round(sum(split_fractions) / len(split_fractions), 4) if split_fractions else 0.0,
        }
        return prototypes, paths, node_ids, masks, stats

    @torch.no_grad()
    def _build_flat_from_embeddings(self, embeddings, fallback_prototypes=None):
        device = embeddings.device
        z = F.normalize(embeddings, dim=1) if self._uses_spherical_geometry() else embeddings
        if fallback_prototypes is None:
            prototypes = torch.randn(self.num_flat_prototypes, self.embedding_dim, device=device)
            prototypes = self._prepare_ot_prototypes(prototypes)
        else:
            prototypes = self._prepare_ot_prototypes(fallback_prototypes.to(device).clone())

        flat_sinkhorn_iters = max(self.sinkhorn_iters, self._FLAT_SINKHORN_MIN_ITERS)
        q = None
        for _ in range(self.em_iters):
            scores = self._ot_scores(z, prototypes)
            q = self._sinkhorn(scores, num_iters=flat_sinkhorn_iters)
            prototypes = self._update_prototypes_from_assignments(z, q)

        labels = q.argmax(dim=1)
        transport_mass = q.sum(dim=0)
        nonempty = transport_mass[transport_mass > 0]
        stats = {
            "flat_nonempty_prototypes": int(nonempty.numel()),
            "flat_min_transport_mass": float(nonempty.min().item()) if nonempty.numel() else 0.0,
            "flat_max_transport_mass": float(nonempty.max().item()) if nonempty.numel() else 0.0,
        }
        return prototypes, q, labels, stats

    @torch.no_grad()
    def _build_vmf_from_embeddings(self, embeddings, fallback_prototypes=None):
        device = embeddings.device
        z = F.normalize(embeddings, dim=1)
        if fallback_prototypes is None:
            prototypes = torch.randn(self.num_flat_prototypes, self.embedding_dim, device=device)
            prototypes = F.normalize(prototypes, dim=-1)
        else:
            prototypes = F.normalize(fallback_prototypes.to(device).clone(), dim=-1)

        for _ in range(self.em_iters):
            q = self._vmf_responsibilities(z, prototypes)
            weighted = q.t() @ z
            updated = F.normalize(weighted, dim=1)
            prototypes = torch.where(weighted.norm(dim=1, keepdim=True) > 1e-12, updated, prototypes)

        scores = self._vmf_scores(z, prototypes)
        labels = scores.argmax(dim=1)
        target_probs = F.one_hot(labels, num_classes=prototypes.shape[0]).float()
        assignment_counts = torch.bincount(labels, minlength=prototypes.shape[0]).float()
        nonempty = assignment_counts[assignment_counts > 0]
        stats = {
            "flat_nonempty_prototypes": int(nonempty.numel()),
            "flat_min_assignment_count": int(nonempty.min().item()) if nonempty.numel() else 0,
            "flat_max_assignment_count": int(nonempty.max().item()) if nonempty.numel() else 0,
        }
        return prototypes, target_probs, labels, stats

    @torch.no_grad()
    def refresh_assignments(self, embeddings):
        if embeddings.shape[0] != self.num_classes:
            raise ValueError(
                "Expected one embedding per pseudo class, got "
                f"{embeddings.shape[0]} for {self.num_classes} classes."
            )

        if self._uses_flat_assignments():
            if self.ot_assignment_mode in ("flat_ot", "flat_vmf_ot"):
                build_flat = self._build_flat_from_embeddings
            else:
                build_flat = self._build_vmf_from_embeddings
            prototypes, target_probs, labels, stats = build_flat(
                embeddings.to(self.flat_prototypes.device),
                fallback_prototypes=self._prepared_flat_prototypes(),
            )
            self._set_flat_prototypes(prototypes)
            self.flat_assignment_labels.copy_(labels)
            self.flat_assignment_probs.copy_(target_probs)
            return stats
        else:
            prototypes, paths, node_ids, masks, stats = self._build_tree_from_embeddings(
                embeddings.to(self.node_prototypes.device),
                fallback_prototypes=self._prepared_node_prototypes(),
            )
            self._set_node_prototypes(prototypes)
            self.assignment_paths.copy_(paths)
            self.assignment_node_ids.copy_(node_ids)
            self.assignment_masks.copy_(masks)
            return stats

    def _forward_embeddings(self, z, pseudo_labels=None, paths=None, node_ids=None, masks=None, prototypes=None):
        if paths is None:
            # Non-batch-local path: look up stored OT tree assignments via the
            # image-index pseudo-labels supplied by the dataloader.
            if pseudo_labels is None:
                raise ValueError(
                    "_forward_embeddings: pseudo_labels (image-index labels from the "
                    "source pool) must be provided when paths/node_ids/masks are not "
                    "given explicitly (non-batch-local mode). Do not pass image-index "
                    "labels as OT tree paths — they are different label spaces."
                )
            labels = pseudo_labels.long()
            paths = self.assignment_paths[labels]
            node_ids = self.assignment_node_ids[labels]
            masks = self.assignment_masks[labels]
            prototypes = self._prepared_node_prototypes()
        else:
            prototypes = prototypes if prototypes is not None else self._prepared_node_prototypes()

        total_loss = z.new_tensor(0.0)
        branch_correct = z.new_tensor(0.0)
        branch_count = z.new_tensor(0.0)
        active_levels = 0
        path_correct = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)

        # Depth annealing: levels >= active_depth are built but not supervised.
        for level in range(min(self.depth, self.active_depth)):
            valid = masks[:, level]
            if valid.sum().item() == 0:
                continue
            active_levels += 1

            node_prototypes = prototypes[node_ids[:, level]]
            logits = self._prediction_logits_per_sample(z, node_prototypes)
            targets = paths[:, level]
            loss = F.cross_entropy(logits, targets, reduction="none")

            valid_float = valid.float()
            total_loss = total_loss + (loss * valid_float).sum() / valid_float.sum().clamp_min(1.0)

            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(targets)
            branch_correct = branch_correct + (correct.float() * valid_float).sum()
            branch_count = branch_count + valid_float.sum()
            path_correct = path_correct & (correct | ~valid)

            # Per-node health bookkeeping (consumed by finalize_node_stats).
            with torch.no_grad():
                ids = node_ids[:, level][valid]
                if ids.numel():
                    self.node_correct_sum.index_add_(0, ids, correct[valid].float())
                    self.node_count.index_add_(0, ids, torch.ones_like(ids, dtype=self.node_count.dtype))

        loss = total_loss / max(active_levels, 1)

        return {
            "loss": loss,
            "acc": path_correct.float().mean(),
            "acc_branch": branch_correct / branch_count.clamp_min(1.0),
            "active_depth": float(self.active_depth),
        }

    def _forward_flat_embeddings(self, z, pseudo_labels=None, labels=None, target_probs=None, prototypes=None):
        if labels is None or target_probs is None:
            # Non-batch-local path: look up stored OT flat assignments via the
            # image-index pseudo-labels supplied by the dataloader.
            if pseudo_labels is None:
                raise ValueError(
                    "_forward_flat_embeddings: pseudo_labels (image-index labels from "
                    "the source pool) must be provided when labels or target_probs are "
                    "not given explicitly (non-batch-local mode). Do not pass image-index "
                    "labels as OT cluster labels — they are different label spaces."
                )
            if labels is None:
                labels = self.flat_assignment_labels[pseudo_labels.long()]
            if target_probs is None:
                target_probs = self.flat_assignment_probs[pseudo_labels.long()]
        prototypes = prototypes if prototypes is not None else self._prepared_flat_prototypes()
        logits = self._prediction_logits(z, prototypes)
        loss = -(target_probs.to(logits.device) * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        acc = logits.argmax(dim=1).eq(labels).float().mean()
        return {
            "loss": loss,
            "acc": acc,
            "acc_branch": acc,
        }

    def _sigmoid_regularization(self, z, pseudo_labels):
        labels = pseudo_labels.long()

        label_embeddings = F.normalize(self.sigmoid_label_embeddings(labels), dim=1)
        logits = z @ label_embeddings.t()
        logits = logits * self.sigmoid_logit_scale.exp() + self.sigmoid_logit_bias

        positive_mask = labels[:, None].eq(labels[None, :])
        signed_targets = torch.where(
            positive_mask,
            torch.ones_like(logits),
            -torch.ones_like(logits),
        )
        # Average over all B² pairs so the loss magnitude is independent of
        # batch size.  The old .sum() / B scaled as O(B); .mean() = .sum() / B²
        # gives a per-pair average that stays constant when batch_size changes.
        loss = -F.logsigmoid(signed_targets * logits).mean()

        predictions = labels[logits.argmax(dim=1)]
        acc = predictions.eq(labels).float().mean()
        positive_logits = logits[positive_mask]
        negative_logits = logits[~positive_mask]
        acc_pos = positive_logits.gt(0).float().mean()
        acc_neg = negative_logits.lt(0).float().mean() if negative_logits.numel() else logits.new_tensor(1.0)

        return {
            "loss_sigmoid_regularization": loss,
            "acc_sigmoid": acc,
            "acc_sigmoid_pos": acc_pos,
            "acc_sigmoid_neg": acc_neg,
            "sigmoid_logit_scale": self.sigmoid_logit_scale.exp().detach(),
            "sigmoid_logit_bias": self.sigmoid_logit_bias.detach(),
        }

    def forward(self, images, pseudo_labels):
        if pseudo_labels.ndim != 1:
            raise ValueError("pseudo_labels must have shape [batch_size].")
        if images.ndim != 4:
            raise ValueError("images must have shape [batch_size, channels, height, width].")

        z = self.encode(images)

        if self._uses_flat_assignments():
            if self.batch_self_labeling:
                current_prototypes = self._prepared_flat_prototypes()
                if self.ot_assignment_mode in ("flat_ot", "flat_vmf_ot"):
                    build_flat = self._build_flat_from_embeddings
                else:
                    build_flat = self._build_vmf_from_embeddings
                prototypes, target_probs, labels, _ = build_flat(
                    z.detach(),
                    fallback_prototypes=current_prototypes.detach(),
                )
                self._ema_update_flat_prototypes(prototypes)
                loss_prototypes = self._prepared_flat_prototypes() if self.learnable_vmf_prototypes else prototypes
                # pseudo_labels (image-index) intentionally not passed:
                # assignment labels are provided directly from z.detach() and
                # must not be confused with image-index labels.
                data_dict = self._forward_flat_embeddings(
                    z,
                    labels=labels,
                    target_probs=target_probs,
                    prototypes=loss_prototypes,
                )
            else:
                data_dict = self._forward_flat_embeddings(z, pseudo_labels)
        elif self.batch_self_labeling:
            current_prototypes = self._prepared_node_prototypes()
            prototypes, paths, node_ids, masks, _ = self._build_tree_from_embeddings(
                z.detach(),
                fallback_prototypes=current_prototypes.detach(),
            )
            self._ema_update_node_prototypes(prototypes)
            loss_prototypes = self._prepared_node_prototypes() if self.learnable_vmf_prototypes else prototypes
            # pseudo_labels (image-index) intentionally not passed: tree paths
            # are provided directly from z.detach() and must not be confused
            # with image-index labels.
            data_dict = self._forward_embeddings(
                z,
                paths=paths,
                node_ids=node_ids,
                masks=masks,
                prototypes=loss_prototypes,
            )
        else:
            data_dict = self._forward_embeddings(z, pseudo_labels)

        if self.sigmoid_regularization_max_weight > 0:
            regularization = self._sigmoid_regularization(z, pseudo_labels)
            weight = self.sigmoid_regularization_current_weight
            data_dict["loss_hierarchical"] = data_dict["loss"]
            data_dict["loss"] = data_dict["loss"] + weight * regularization["loss_sigmoid_regularization"]
            data_dict.update(regularization)
            data_dict["sigmoid_regularization_weight"] = weight.detach()

        return data_dict
