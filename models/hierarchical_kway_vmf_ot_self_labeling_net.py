"""Hierarchical K-way vMF/OT self-labeling on the unit sphere.

Combines the HiRef rank-annealing *prior* (per-level branching factor chosen by
the dynamic program of Halmos et al., ICML 2025) with vMF fitting in spherical
feature space. Each tree node refines its points into ``r_t`` children via a
balanced (optionally unbalanced) spherical Sinkhorn / vMF EM step; the HiRef
``argmax`` Assign rule produces hard child labels. Only the first
``supervised_depth`` (early, stable) levels emit pseudo-labels for the backbone.

Kept isolated from ``HierarchicalBalancedVMFSelfLabelingNet`` (the user-approved
"new mode, additive" path); a few short numerical helpers are re-implemented
locally so the existing binary model and its tests are untouched.

See ``.agents/designs/hierarchical_kway_vmf_ot_design.md``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.rank_annealing import (
    child_global_id,
    leaf_index,
    level_offsets,
    level_sizes,
    num_internal_nodes,
    optimal_rank_schedule,
)


def resolve_rank_schedule(
    rank_schedule=None,
    num_leaf_clusters=None,
    depth=None,
    max_rank=None,
    base_rank=1,
):
    """Return a concrete rank schedule, honoring an explicit override.

    If ``rank_schedule`` is given it is validated and returned. Otherwise the
    HiRef dynamic program derives the optimal schedule of length ``depth`` whose
    ranks multiply to ``num_leaf_clusters / base_rank``.
    """
    if rank_schedule is not None:
        schedule = [int(r) for r in rank_schedule]
        if len(schedule) < 1:
            raise ValueError("rank_schedule must have at least one level.")
        if any(r < 2 for r in schedule):
            raise ValueError("every rank in rank_schedule must be at least 2.")
        return schedule
    if num_leaf_clusters is None or depth is None or max_rank is None:
        raise ValueError(
            "Provide rank_schedule explicitly, or num_leaf_clusters + depth + "
            "rank_schedule_max_rank to derive it via the HiRef DP."
        )
    return optimal_rank_schedule(
        n=int(num_leaf_clusters),
        depth=int(depth),
        max_rank=int(max_rank),
        base_rank=int(base_rank),
    )


class HierarchicalKWayVMFOTSelfLabelingNet(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone=None,
        embedding_dim=None,
        rank_schedule=None,
        num_leaf_clusters=256,
        rank_schedule_depth=None,
        rank_schedule_max_rank=16,
        rank_schedule_base_rank=1,
        supervised_depth=None,
        kappa=20.0,
        ot_epsilon=0.05,
        sinkhorn_iters=10,
        em_iters=5,
        ot_unbalanced_tau=None,
        ot_unbalanced_min_split_fraction=0.05,
        batch_self_labeling=True,
        swapped_view_assignment=False,
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
            raise ValueError("HierarchicalKWayVMFOTSelfLabelingNet requires at least 2 pseudo classes.")

        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim or backbone.output_dim)

        if rank_schedule is not None:
            schedule = resolve_rank_schedule(rank_schedule=rank_schedule)
        else:
            # HiRef treats hierarchy depth as a user budget (number of LROT
            # calls / memory), then optimizes ranks for it. Default to the
            # minimum feasible depth only when none is given.
            depth = rank_schedule_depth or _required_depth(
                num_leaf_clusters, rank_schedule_max_rank, rank_schedule_base_rank
            )
            schedule = optimal_rank_schedule(
                n=int(num_leaf_clusters),
                depth=int(depth),
                max_rank=int(rank_schedule_max_rank),
                base_rank=int(rank_schedule_base_rank),
            )
        self.rank_schedule = list(schedule)
        self.depth = len(self.rank_schedule)
        if self.depth < 1:
            raise ValueError("rank schedule produced depth < 1.")
        if self.depth > 24:
            raise ValueError("rank schedule depth above 24 is unsupported.")

        self.supervised_depth = self.depth if supervised_depth is None else int(supervised_depth)
        if not 1 <= self.supervised_depth <= self.depth:
            raise ValueError(f"supervised_depth must be in [1, {self.depth}], got {self.supervised_depth}.")
        # Depth annealing clamps active_depth into [1, supervised_depth].
        self.active_depth = self.supervised_depth

        self.kappa = float(kappa)
        self.ot_epsilon = float(ot_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.em_iters = int(em_iters)
        self.batch_self_labeling = bool(batch_self_labeling)
        self.swapped_view_assignment = bool(swapped_view_assignment)
        if self.swapped_view_assignment and not self.batch_self_labeling:
            raise ValueError(
                "swapped_view_assignment requires batch_self_labeling=True: the "
                "tree must be (re)fit on the current batch so each view can "
                "predict the other view's freshly assigned path."
            )
        self.ot_unbalanced_tau = None if ot_unbalanced_tau is None else float(ot_unbalanced_tau)
        self.ot_unbalanced_min_split_fraction = float(ot_unbalanced_min_split_fraction)
        self.learnable_vmf_prototypes = bool(learnable_vmf_prototypes)
        self.prototype_ema_momentum = None if prototype_ema_momentum is None else float(prototype_ema_momentum)
        self.sigmoid_regularization_max_weight = float(sigmoid_regularization_weight)

        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.ot_epsilon <= 0:
            raise ValueError("ot_epsilon must be positive.")
        if self.sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be at least 1.")
        if self.em_iters < 1:
            raise ValueError("em_iters must be at least 1.")
        if self.ot_unbalanced_tau is not None and self.ot_unbalanced_tau <= 0:
            raise ValueError("ot_unbalanced_tau must be positive or null.")
        if not 0.0 <= self.ot_unbalanced_min_split_fraction < 0.5:
            raise ValueError("ot_unbalanced_min_split_fraction must be in [0, 0.5).")
        if self.prototype_ema_momentum is not None and not 0.0 <= self.prototype_ema_momentum < 1.0:
            raise ValueError("prototype_ema_momentum must be in [0, 1) or null.")
        if self.learnable_vmf_prototypes and self.prototype_ema_momentum is not None:
            raise ValueError("learnable_vmf_prototypes and prototype_ema_momentum cannot both be set.")
        if self.sigmoid_regularization_max_weight < 0:
            raise ValueError("sigmoid_regularization_weight must be non-negative.")
        if sigmoid_init_temperature <= 0:
            raise ValueError("sigmoid_init_temperature must be positive.")

        if self.embedding_dim == backbone.output_dim:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Linear(backbone.output_dim, self.embedding_dim)

        self.max_rank = max(self.rank_schedule)
        self.n_internal = num_internal_nodes(self.rank_schedule)
        self._level_sizes = level_sizes(self.rank_schedule)
        self._level_offsets = level_offsets(self.rank_schedule)
        # node id -> level, as a buffer for gather-free lookups.
        node_levels = torch.empty(self.n_internal, dtype=torch.long)
        for t, (off, size) in enumerate(zip(self._level_offsets, self._level_sizes)):
            node_levels[off : off + size] = t
        self.register_buffer("node_levels", node_levels)

        # Padded prototype store: node at level t uses slots [:r_t].
        prototypes = F.normalize(
            torch.randn(self.n_internal, self.max_rank, self.embedding_dim), dim=-1
        )
        # Coarse pseudo-labels for the non-batch-local path: one path per class.
        paths = self._balanced_initial_paths()
        node_ids, masks = self._paths_to_node_ids(paths)

        if self.learnable_vmf_prototypes:
            self.node_prototypes = nn.Parameter(prototypes)
        else:
            self.register_buffer("node_prototypes", prototypes)
        self.register_buffer("node_fitted", torch.zeros(self.n_internal, dtype=torch.bool))
        self.register_buffer("node_correct_sum", torch.zeros(self.n_internal))
        self.register_buffer("node_count", torch.zeros(self.n_internal))
        self.register_buffer("assignment_paths", paths)
        self.register_buffer("assignment_node_ids", node_ids)
        self.register_buffer("assignment_masks", masks)
        self.register_buffer("sigmoid_regularization_current_weight", torch.zeros([]))

        self.sigmoid_label_embeddings = nn.Embedding(self.num_classes, self.embedding_dim)
        nn.init.normal_(self.sigmoid_label_embeddings.weight, std=0.02)
        self.sigmoid_logit_scale = nn.Parameter(torch.ones([]) * math.log(float(sigmoid_init_temperature)))
        self.sigmoid_logit_bias = nn.Parameter(torch.ones([]) * float(sigmoid_init_bias))

    # ----- initial coarse assignments (non-batch-local fallback) -----------
    def _balanced_initial_paths(self):
        num_leaves = math.prod(self.rank_schedule)
        leaf_ids = (torch.arange(self.num_classes, dtype=torch.long) * num_leaves) // self.num_classes
        paths = torch.zeros(self.num_classes, self.depth, dtype=torch.long)
        for i in range(self.num_classes):
            remainder = int(leaf_ids[i].item())
            digits = []
            for r in reversed(self.rank_schedule):
                digits.append(remainder % r)
                remainder //= r
            paths[i] = torch.tensor(list(reversed(digits)), dtype=torch.long)
        return paths

    def _paths_to_node_ids(self, paths):
        n = paths.shape[0]
        node_ids = torch.zeros(n, self.depth, dtype=torch.long)
        masks = torch.ones(n, self.depth, dtype=torch.bool)
        local = torch.zeros(n, dtype=torch.long)
        for t in range(self.depth):
            node_ids[:, t] = self._level_offsets[t] + local
            local = local * self.rank_schedule[t] + paths[:, t]
        return node_ids, masks

    # ----- public hooks used by main.py ------------------------------------
    def set_active_depth(self, active_depth):
        active_depth = int(active_depth)
        # Never supervise beyond the configured early-levels cutoff.
        active_depth = min(active_depth, self.supervised_depth)
        if not 1 <= active_depth <= self.depth:
            raise ValueError(f"active_depth must be in [1, {self.depth}], got {active_depth}.")
        self.active_depth = active_depth

    def set_ot_unbalanced_tau(self, tau):
        if tau is not None:
            tau = float(tau)
            if tau <= 0:
                raise ValueError("ot_unbalanced_tau must be positive or None.")
        self.ot_unbalanced_tau = tau

    def set_sigmoid_regularization_progress(self, current_step, rampup_steps):
        if self.sigmoid_regularization_max_weight <= 0:
            weight = 0.0
        elif rampup_steps <= 0:
            weight = self.sigmoid_regularization_max_weight
        else:
            progress = min(1.0, float(current_step) / float(rampup_steps))
            weight = self.sigmoid_regularization_max_weight * progress
        self.sigmoid_regularization_current_weight.fill_(weight)

    @torch.no_grad()
    def finalize_node_stats(self):
        counts = self.node_count
        accs = self.node_correct_sum / counts.clamp_min(1.0)
        visited = counts > 0
        stats = {"tree_nodes_visited": float(visited.sum().item())}
        if visited.any():
            stats["tree_node_acc_overall"] = float(accs[visited].mean().item())
        for level in range(self.depth):
            level_visited = visited & (self.node_levels == level)
            if level_visited.any():
                stats[f"tree_node_acc_level{level}"] = float(accs[level_visited].mean().item())
        self.node_correct_sum.zero_()
        self.node_count.zero_()
        return stats

    # ----- geometry / numerics --------------------------------------------
    def encode(self, images):
        embeddings = self.projector(self.backbone(images))
        return F.normalize(embeddings, dim=1)

    def _prepared_prototypes(self):
        return F.normalize(self.node_prototypes, dim=-1)

    @torch.no_grad()
    def _set_prototypes(self, prototypes):
        self.node_prototypes.copy_(F.normalize(prototypes.to(self.node_prototypes.device), dim=-1))

    @torch.no_grad()
    def _ema_update_prototypes(self, batch_prototypes):
        if self.prototype_ema_momentum is None:
            return
        m = self.prototype_ema_momentum
        current = self._prepared_prototypes().detach()
        self._set_prototypes(m * current + (1.0 - m) * batch_prototypes.to(current.device))

    def _cosine_scores(self, z, mu):
        return F.normalize(z, dim=1) @ F.normalize(mu, dim=1).t()

    def _sinkhorn(self, scores, num_iters):
        """Balanced Sinkhorn-Knopp (uniform component marginal 1/K)."""
        q = torch.exp(scores - scores.max(dim=1, keepdim=True).values).t()
        q = q / q.sum().clamp_min(1e-12)
        k, b = q.shape
        for _ in range(int(num_iters)):
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
            q = q / k
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            q = q / b
        return (q * b).t()

    def _unbalanced_sinkhorn(self, scores, tau, num_iters):
        """Semi-relaxed unbalanced Sinkhorn (Chizat et al. 2018, scaling form)."""
        phi = float(tau) / (float(tau) + max(self.ot_epsilon, 1e-6))
        q = torch.exp(scores - scores.max(dim=1, keepdim=True).values).t()
        q = q / q.sum().clamp_min(1e-12)
        k, b = q.shape
        target = q.new_full((k, 1), 1.0 / k)
        for _ in range(int(num_iters)):
            mass = q.sum(dim=1, keepdim=True).clamp_min(1e-12)
            q = q * (target / mass).pow(phi)
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            q = q / b
        return (q * b).t()

    def _ot_assign(self, scores, num_samples, num_components):
        if self.ot_unbalanced_tau is not None:
            return self._unbalanced_sinkhorn(scores, self.ot_unbalanced_tau, self.sinkhorn_iters)
        return self._sinkhorn(scores, self.sinkhorn_iters)

    def _cold_seed(self, z, r):
        """Spherical farthest-point (k-means++ flavour) seeding of r directions."""
        n = z.shape[0]
        mean = F.normalize(z.mean(dim=0, keepdim=True), dim=1)
        first = int(torch.argmin((z @ mean.t()).squeeze(1)).item())
        chosen = [first]
        max_sim = (z @ z[first : first + 1].t()).squeeze(1)
        for _ in range(1, r):
            nxt = int(torch.argmin(max_sim).item())
            chosen.append(nxt)
            sim_new = (z @ z[nxt : nxt + 1].t()).squeeze(1)
            max_sim = torch.maximum(max_sim, sim_new)
        return F.normalize(z[torch.tensor(chosen, device=z.device)], dim=1)

    def _fit_kway_vmf_ot(self, z, r, init_mu=None):
        """Fit r vMF directions and return (mu[r,d], hard[N] in [0,r))."""
        num_samples = z.shape[0]
        if init_mu is not None:
            mu = F.normalize(init_mu.to(z.device), dim=1)
            # Degenerate (near-duplicate) seeds -> cold restart.
            gram = mu @ mu.t()
            off = gram - torch.eye(r, device=z.device)
            if off.max().item() > 0.99:
                mu = self._cold_seed(z, r)
        else:
            mu = self._cold_seed(z, r)

        q = None
        for _ in range(self.em_iters):
            scores = self._cosine_scores(z, mu) / max(self.ot_epsilon, 1e-6)
            q = self._ot_assign(scores, num_samples, r)
            weighted = q.t() @ z
            mu = F.normalize(weighted, dim=1)

        hard = q.argmax(dim=1)
        if self.ot_unbalanced_tau is not None:
            counts = torch.bincount(hard, minlength=r)
            min_count = max(1, int(math.floor(self.ot_unbalanced_min_split_fraction * num_samples)))
            if int(counts.min().item()) < min_count:
                # Collapse guard: refit this node balanced.
                q_bal = self._sinkhorn(self._cosine_scores(z, mu) / max(self.ot_epsilon, 1e-6), self.sinkhorn_iters)
                hard = q_bal.argmax(dim=1)
        return mu, hard

    @torch.no_grad()
    def _build_kway_tree(self, embeddings, fallback_prototypes=None, warm_start=True):
        device = embeddings.device
        n = embeddings.shape[0]
        z = F.normalize(embeddings, dim=1)
        paths = torch.zeros(n, self.depth, dtype=torch.long, device=device)
        node_ids = torch.zeros(n, self.depth, dtype=torch.long, device=device)
        masks = torch.zeros(n, self.depth, dtype=torch.bool, device=device)
        if fallback_prototypes is None:
            prototypes = F.normalize(
                torch.randn(self.n_internal, self.max_rank, self.embedding_dim, device=device), dim=-1
            )
        else:
            prototypes = fallback_prototypes.to(device).clone()

        split_fractions = []
        # Each entry: (global_id, local_index, indices).
        level_nodes = [(self._level_offsets[0], 0, torch.arange(n, device=device))]
        for t in range(self.depth):
            r = self.rank_schedule[t]
            next_level_nodes = []
            for node_gid, local_index, indices in level_nodes:
                node_ids[indices, t] = node_gid
                if indices.numel() < r:
                    continue  # too few points to populate r children; mask stays False
                use_warm = warm_start and bool(self.node_fitted[node_gid].item())
                mu, hard = self._fit_kway_vmf_ot(
                    z[indices],
                    r,
                    init_mu=prototypes[node_gid, :r] if use_warm else None,
                )
                self.node_fitted[node_gid] = True
                prototypes[node_gid, :r] = mu
                paths[indices, t] = hard
                masks[indices, t] = True

                counts = torch.bincount(hard, minlength=r).float()
                split_fractions.append((counts.min() / counts.sum().clamp_min(1.0)).item())

                if t + 1 < self.depth:
                    for child in range(r):
                        child_idx = indices[hard == child]
                        child_gid = child_global_id(self.rank_schedule, t, local_index, child)
                        next_level_nodes.append((child_gid, local_index * r + child, child_idx))
            level_nodes = next_level_nodes

        # Leaf ids (mixed radix) for occupancy stats.
        leaf_ids = torch.zeros(n, dtype=torch.long, device=device)
        for t in range(self.depth):
            leaf_ids = leaf_ids * self.rank_schedule[t] + paths[:, t]
        num_leaves = math.prod(self.rank_schedule)
        leaf_counts = torch.bincount(leaf_ids, minlength=num_leaves).float()
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
    def predict_paths(self, embeddings):
        """Read-only root-to-leaf descent through the stored prototypes.

        Returns mixed-radix paths [N, depth] (digit t in [0, r_t)). Used by
        the purity/NMI diagnostics in main.py: it scores the partition the
        current prototype hierarchy *predicts*, mutating no model state
        (unlike the tree builders, which fit and mark nodes).
        """
        z = F.normalize(embeddings.to(self.node_prototypes.device), dim=1)
        prototypes = self._prepared_prototypes()
        n = z.shape[0]
        paths = torch.zeros(n, self.depth, dtype=torch.long, device=z.device)
        local = torch.zeros(n, dtype=torch.long, device=z.device)
        for t in range(self.depth):
            r = self.rank_schedule[t]
            mu = prototypes[self._level_offsets[t] + local, :r, :]
            digit = torch.einsum("nd,nrd->nr", z, mu).argmax(dim=1)
            paths[:, t] = digit
            local = local * r + digit
        return paths

    @torch.no_grad()
    def refresh_assignments(self, embeddings):
        if embeddings.shape[0] != self.num_classes:
            raise ValueError(
                f"Expected one embedding per pseudo class, got {embeddings.shape[0]} "
                f"for {self.num_classes} classes."
            )
        prototypes, paths, node_ids, masks, stats = self._build_kway_tree(
            embeddings.to(self.node_prototypes.device),
            fallback_prototypes=self._prepared_prototypes(),
            warm_start=True,
        )
        self._set_prototypes(prototypes)
        self.assignment_paths.copy_(paths)
        self.assignment_node_ids.copy_(node_ids)
        self.assignment_masks.copy_(masks)
        return stats

    # ----- loss ------------------------------------------------------------
    def _branch_logits(self, z, node_prototypes, r):
        # node_prototypes: [B, max_rank, d]; use only the first r children.
        mu = F.normalize(node_prototypes[:, :r, :], dim=-1)
        return self.kappa * torch.einsum("bd,brd->br", F.normalize(z, dim=1), mu)

    def _forward_embeddings(self, z, paths, node_ids, masks, prototypes):
        total_loss = z.new_tensor(0.0)
        branch_correct = z.new_tensor(0.0)
        branch_count = z.new_tensor(0.0)
        active_levels = 0
        path_correct = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)

        for level in range(min(self.depth, self.active_depth)):
            valid = masks[:, level]
            if valid.sum().item() == 0:
                continue
            active_levels += 1
            r = self.rank_schedule[level]
            logits = self._branch_logits(z, prototypes[node_ids[:, level]], r)
            targets = paths[:, level]
            loss = F.cross_entropy(logits, targets, reduction="none")
            valid_f = valid.float()
            total_loss = total_loss + (loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)

            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(targets)
            branch_correct = branch_correct + (correct.float() * valid_f).sum()
            branch_count = branch_count + valid_f.sum()
            path_correct = path_correct & (correct | ~valid)

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

    def _sigmoid_regularization(self, z, pseudo_labels):
        labels = pseudo_labels.long()
        label_embeddings = F.normalize(self.sigmoid_label_embeddings(labels), dim=1)
        logits = z @ label_embeddings.t()
        logits = logits * self.sigmoid_logit_scale.exp() + self.sigmoid_logit_bias
        positive_mask = labels[:, None].eq(labels[None, :])
        signed_targets = torch.where(positive_mask, torch.ones_like(logits), -torch.ones_like(logits))
        loss = -F.logsigmoid(signed_targets * logits).mean()
        return {"loss_sigmoid_regularization": loss}

    def _forward_swapped_views(self, z_flat, batch_size):
        """Symmetric swapped-view prediction (SwAV-style), batch-local tree.

        ``z_flat`` is [2B, d], sample-major (the two views of image ``i`` sit at
        rows ``2i`` and ``2i + 1``). One tree is fit on all 2B detached
        embeddings (shared prototypes), then each view is trained to predict
        the *other* view's assigned path. Unlike single-view batch
        self-labeling -- where any stable partition of the batch is a perfect,
        self-confirming solution -- the swapped loss is only low when the two
        augmentations of an image route identically, i.e. when the partition
        is augmentation-invariant.
        """
        current = self._prepared_prototypes()
        prototypes, paths, node_ids, masks, _ = self._build_kway_tree(
            z_flat.detach(), fallback_prototypes=current.detach(), warm_start=True
        )
        self._ema_update_prototypes(prototypes)
        loss_prototypes = self._prepared_prototypes() if self.learnable_vmf_prototypes else prototypes

        z = z_flat.view(batch_size, 2, -1)
        paths = paths.view(batch_size, 2, self.depth)
        node_ids = node_ids.view(batch_size, 2, self.depth)
        masks = masks.view(batch_size, 2, self.depth)

        # Swapped prediction: view 1 predicts view 0's path and vice versa.
        first = self._forward_embeddings(z[:, 1], paths[:, 0], node_ids[:, 0], masks[:, 0], loss_prototypes)
        second = self._forward_embeddings(z[:, 0], paths[:, 1], node_ids[:, 1], masks[:, 1], loss_prototypes)

        data_dict = {
            "loss": 0.5 * (first["loss"] + second["loss"]),
            "acc": 0.5 * (first["acc"] + second["acc"]),
            "acc_branch": 0.5 * (first["acc_branch"] + second["acc_branch"]),
            "active_depth": first["active_depth"],
        }
        with torch.no_grad():
            # Diagnostic: do the two views of an image get the same OT
            # assignment? This is the invariance the swapped loss optimizes.
            sup = min(self.depth, self.active_depth)
            both = masks[:, 0, :sup] & masks[:, 1, :sup]
            agree = (paths[:, 0, :sup] == paths[:, 1, :sup]) & both
            data_dict["tree_view_agreement"] = agree.float().sum() / both.float().sum().clamp_min(1.0)
        return data_dict

    def forward(self, images, pseudo_labels):
        if pseudo_labels.ndim != 1:
            raise ValueError("pseudo_labels must have shape [batch_size].")

        if self.swapped_view_assignment:
            if images.ndim != 5 or images.shape[1] != 2:
                raise ValueError(
                    "swapped_view_assignment expects images of shape "
                    "[batch_size, 2, channels, height, width]; set train.num_views: 2."
                )
            z = self.encode(images.flatten(0, 1))
            data_dict = self._forward_swapped_views(z, images.shape[0])
            reg_z = z
            reg_labels = pseudo_labels.long().repeat_interleave(2)
        else:
            if images.ndim != 4:
                raise ValueError("images must have shape [batch_size, channels, height, width].")

            z = self.encode(images)

            if self.batch_self_labeling:
                current = self._prepared_prototypes()
                prototypes, paths, node_ids, masks, _ = self._build_kway_tree(
                    z.detach(), fallback_prototypes=current.detach(), warm_start=True
                )
                self._ema_update_prototypes(prototypes)
                loss_prototypes = self._prepared_prototypes() if self.learnable_vmf_prototypes else prototypes
                data_dict = self._forward_embeddings(z, paths, node_ids, masks, loss_prototypes)
            else:
                labels = pseudo_labels.long()
                data_dict = self._forward_embeddings(
                    z,
                    self.assignment_paths[labels],
                    self.assignment_node_ids[labels],
                    self.assignment_masks[labels],
                    self._prepared_prototypes(),
                )
            reg_z = z
            reg_labels = pseudo_labels

        if self.sigmoid_regularization_max_weight > 0:
            reg = self._sigmoid_regularization(reg_z, reg_labels)
            weight = self.sigmoid_regularization_current_weight
            data_dict["loss_hierarchical"] = data_dict["loss"]
            data_dict["loss"] = data_dict["loss"] + weight * reg["loss_sigmoid_regularization"]
            data_dict["loss_sigmoid_regularization"] = reg["loss_sigmoid_regularization"]
            data_dict["sigmoid_regularization_weight"] = weight.detach()

        return data_dict


def _required_depth(num_leaf_clusters, max_rank, base_rank):
    """Smallest depth whose max_rank^depth can reach num_leaf_clusters/base_rank."""
    target = int(num_leaf_clusters) // int(base_rank)
    depth = 1
    while max_rank ** depth < target:
        depth += 1
    return depth


