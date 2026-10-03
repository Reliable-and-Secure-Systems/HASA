"""Federated training and evaluation.

run_policy: the original FedAvg pipeline (full-tensor aggregation) used for the main
            results, the per-client table and the direction ablation.
run_method: selective (mask-aware) aggregation used for the matched-budget comparison
            with HeteroFL-, FedDropout- and FedRolex-style baselines.
Evaluation follows the paper: every client's test accuracy is measured with the
aggregated global model at that client's own subnet width.
"""
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .method import allocate_subnet_ratios
from .model import HeterogeneousSubnetLSTM, forward_subnet_with_indices
from .utils import set_all_seeds

EMBED_DIM, HIDDEN_DIM, NUM_LAYERS = 128, 256, 1
LR, FL_ROUNDS, LOCAL_EPOCHS, EVAL_EVERY = 1e-3, 50, 1, 5
BUDGET = 0.5

criterion = nn.CrossEntropyLoss()


def client_weights_by_size(client_sizes):
    """FedAvg weights n_i / N."""
    sizes = np.array(client_sizes, dtype=float)
    return (sizes / sizes.sum()).tolist()


def new_model(bench, ratio, device):
    return HeterogeneousSubnetLSTM(len(bench.features_vocab), len(bench.target_vocab), EMBED_DIM,
                                   HIDDEN_DIM, NUM_LAYERS, subnet_ratio=ratio).to(device)


def load_into_model(model, state_dict):
    current = model.state_dict()
    for k in state_dict:
        if k in current and "subnet_mask" not in k:
            current[k].copy_(state_dict[k])


def state_without_mask(model):
    return {k: v for k, v in model.state_dict().items() if "subnet_mask" not in k}


def train_client(model, loader, optimizer, device, epochs=1):
    model.train()
    total, n = 0.0, 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, use_subnet=True), y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n += 1
    return total / max(n, 1)


@torch.no_grad()
def accuracy(model, loader, device, use_subnet):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += int((model(x, use_subnet=use_subnet).argmax(dim=1) == y).sum().item())
        total += int(y.size(0))
    return correct / max(total, 1)


@torch.no_grad()
def perplexity(model, loader, device, use_subnet):
    model.eval()
    total_loss, total = 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total_loss += float(ce(model(x, use_subnet=use_subnet), y).item())
        total += int(y.size(0))
    return float(np.exp(total_loss / max(total, 1)))


@torch.no_grad()
def evaluate_global_on_clients(bench, global_model, ratios, device, split="test"):
    """Accuracy of the global model on each client's split, at that client's width."""
    global_state = state_without_mask(global_model)
    accs = []
    for cid, r in enumerate(ratios):
        temp = new_model(bench, r, device)
        load_into_model(temp, global_state)
        accs.append(accuracy(temp, bench.clients[cid].loaders[split], device, use_subnet=True))
    return accs


def fedavg_aggregate(client_models, client_weights):
    agg = {k: v.clone().float() * 0 for k, v in state_without_mask(client_models[0]).items()}
    for model, w in zip(client_models, client_weights):
        state = model.state_dict()
        for k in agg:
            agg[k] += state[k].float() * w
    return agg


def summarize(accs):
    return float(np.mean(accs)), float(np.min(accs)), float(np.percentile(accs, 10))


def run_policy(bench, scores, policy, device, seed=42, invert_difficulty=False, alpha=0.5,
               rounds=FL_ROUNDS, eval_every=EVAL_EVERY, log=print):
    """One federated run with full-tensor FedAvg. policy: difficulty_only (HASA), uniform_50,
    size_only, mixed or fedavg_full. Returns a history dict; the paper reports the last entry."""
    set_all_seeds(seed)
    sizes = bench.client_data_sizes
    client_weights = client_weights_by_size(sizes)
    ratios = allocate_subnet_ratios(sizes, scores, policy=policy, alpha=alpha,
                                    invert_difficulty=invert_difficulty, enforce_budget=True,
                                    target_weighted_mean=BUDGET)

    global_model = new_model(bench, 1.0, device)
    base_state = {k: v.clone().detach() for k, v in state_without_mask(global_model).items()}
    client_models = []
    for r in ratios:
        m = new_model(bench, r, device)
        load_into_model(m, base_state)
        client_models.append(m)
    load_into_model(global_model, base_state)

    hist = {"policy": policy, "seed": seed, "invert_difficulty": invert_difficulty, "ratios": ratios,
            "weighted_mean_ratio": float(np.sum(np.array(ratios) * np.array(sizes)) / sum(sizes)),
            "round": [], "train_loss": [], "test_mean": [], "test_worst": [], "test_p10": [],
            "test_ppl": [], "per_client_test": [], "val_mean": [], "global_infed": [], "global_ood": []}
    tag = f"{policy}{'-inverse' if invert_difficulty else ''}"
    for rnd in range(1, rounds + 1):
        global_state = state_without_mask(global_model)
        losses = []
        for cid in range(bench.num_clients):
            load_into_model(client_models[cid], global_state)
            opt = optim.Adam(client_models[cid].parameters(), lr=LR)
            losses.append(train_client(client_models[cid], bench.clients[cid].loaders["train"], opt, device, LOCAL_EPOCHS))
        load_into_model(global_model, fedavg_aggregate(client_models, client_weights))

        if rnd % eval_every == 0 or rnd == 1 or rnd == rounds:
            val = evaluate_global_on_clients(bench, global_model, ratios, device, "val")
            test = evaluate_global_on_clients(bench, global_model, ratios, device, "test")
            # The original pipeline also scores the local models here; kept so that runs match it exactly.
            for cid in range(bench.num_clients):
                accuracy(client_models[cid], bench.clients[cid].loaders["test"], device, use_subnet=True)
            g_ood = accuracy(global_model, bench.global_ood_loader, device, use_subnet=False)
            g_infed = accuracy(global_model, bench.global_infed_loader, device, use_subnet=False)
            perplexity(global_model, bench.global_ood_loader, device, use_subnet=False)
            perplexity(global_model, bench.global_infed_loader, device, use_subnet=False)
            ppl = float(np.mean([perplexity(client_models[cid], bench.clients[cid].loaders["test"], device, True)
                                 for cid in range(bench.num_clients)]))
            mean, worst, p10 = summarize(test)
            hist["round"].append(rnd)
            hist["train_loss"].append(float(np.mean(losses)))
            hist["test_mean"].append(mean)
            hist["test_worst"].append(worst)
            hist["test_p10"].append(p10)
            hist["test_ppl"].append(ppl)
            hist["per_client_test"].append(test)
            hist["val_mean"].append(float(np.mean(val)))
            hist["global_infed"].append(g_infed)
            hist["global_ood"].append(g_ood)
            if log:
                log(f"[{tag}] round {rnd:2d} | loss {np.mean(losses):.4f} | test mean {mean*100:.2f}% "
                    f"worst {worst*100:.2f}% P10 {p10*100:.2f}% | ppl {ppl:.1f}")
    return hist


# ---------------------------------------------------------------- matched-budget baselines
def select_indices(mode, a, H, rnd, rng):
    """Which hidden units a client trains this round."""
    if mode == "full":
        return list(range(H))
    if mode == "static":            # HASA, Uniform-Static, HeteroFL: fixed prefix
        return list(range(a))
    if mode == "random":            # Federated Dropout: random subset every round
        return sorted(rng.sample(range(H), a))
    if mode == "rolling":           # FedRolex: rolling window with wrap-around
        return [(rnd + i) % H for i in range(a)]
    raise ValueError(mode)


def selective_aggregate(global_state, client_states, client_active, client_weights, H):
    """Average only the entries each client actually trained; untouched entries keep
    the previous global value."""
    new_state = {k: v.clone().float() for k, v in global_state.items() if "subnet_mask" not in k}
    accum = {k: torch.zeros_like(v) for k, v in new_state.items()}
    counts = {k: torch.zeros_like(v) for k, v in new_state.items()}
    for cstate, active, w in zip(client_states, client_active, client_weights):
        act = torch.tensor(active, dtype=torch.long)
        gate = torch.cat([act, act + H, act + 2 * H, act + 3 * H])
        for k in new_state:
            p = cstate[k].float()
            if k in ("embedding.weight", "fc.bias"):
                accum[k] += p * w
                counts[k] += w
            elif k == "lstm.weight_hh_l0":
                rows, cols = gate.unsqueeze(1), act.unsqueeze(0)
                accum[k][rows, cols] += p[rows, cols] * w
                counts[k][rows, cols] += w
            elif k == "lstm.weight_ih_l0":
                accum[k][gate, :] += p[gate, :] * w
                counts[k][gate, :] += w
            elif k in ("lstm.bias_ih_l0", "lstm.bias_hh_l0"):
                accum[k][gate] += p[gate] * w
                counts[k][gate] += w
            elif k == "fc.weight":
                accum[k][:, act] += p[:, act] * w
                counts[k][:, act] += w
            else:
                accum[k] += p * w
                counts[k] += w
    for k in new_state:
        mask = counts[k] > 0
        new_state[k][mask] = accum[k][mask] / counts[k][mask]
    return new_state


def active_units(ratio, H=HIDDEN_DIM, min_active=16):
    return max(min_active, min(int(H * ratio), H))


def overhead(a, V, E, H, O, T):
    """Active parameters, forward MACs per sequence and fp32 traffic for a units."""
    full_params = V * E + (4 * H * E + 4 * H * H + 8 * H) + (O * H + O)
    active_params = V * E + (4 * a * E + 4 * a * a + 8 * a) + (O * a + O)
    return {"param_ratio": active_params / full_params,
            "macs": T * (4 * a * E + 4 * a * a) + O * a,
            "bytes": active_params * 4}


def run_method(bench, scores, assignment, extraction, aggregation, device, seed=42,
               rounds=FL_ROUNDS, eval_every=EVAL_EVERY, log=print, label=None):
    """One run of the matched-budget comparison (paper Sec. VI-B)."""
    rng = random.Random(seed)
    set_all_seeds(seed)
    V, E, H, O, T = len(bench.features_vocab), EMBED_DIM, HIDDEN_DIM, len(bench.target_vocab), bench.max_seq_length
    sizes = bench.client_data_sizes
    weights = client_weights_by_size(sizes)
    ratios = allocate_subnet_ratios(sizes, scores, policy=assignment, enforce_budget=True, target_weighted_mean=BUDGET)
    units = [active_units(r) for r in ratios]
    oh = [overhead(a, V, E, H, O, T) for a in units]
    w = np.array(weights)
    full_macs = T * (4 * H * E + 4 * H * H) + O * H

    global_model = new_model(bench, 1.0, device)
    base_state = {k: v.clone().detach() for k, v in state_without_mask(global_model).items()}
    client_models = []
    for r in ratios:
        m = new_model(bench, r, device)
        load_into_model(m, base_state)
        client_models.append(m)
    load_into_model(global_model, base_state)

    label = label or f"{assignment}/{extraction}/{aggregation}"
    hist = {"method": label, "seed": seed, "ratios": ratios, "active_units": units,
            "wtd_macs": float(np.sum(np.array([o["macs"] for o in oh]) * w)) / full_macs,
            "wtd_params": float(np.sum(np.array([o["param_ratio"] for o in oh]) * w)),
            "sim_ul_mb": float(np.sum(np.array([o["bytes"] for o in oh]) * w)) / 1e6,
            "round": [], "test_mean": [], "test_worst": [], "test_p10": [], "test_ppl": [],
            "per_client_test": [], "global_infed": [], "global_ood": [], "epoch_s": []}
    for rnd in range(1, rounds + 1):
        global_state = {k: v.clone() for k, v in state_without_mask(global_model).items()}
        active = []
        for cid in range(bench.num_clients):
            load_into_model(client_models[cid], global_state)
            active.append(select_indices(extraction, units[cid], H, rnd, rng))
        for cid in range(bench.num_clients):
            m = client_models[cid]
            opt = optim.Adam(m.parameters(), lr=LR)
            m.train()
            t0 = time.time()
            for x, y in bench.clients[cid].loaders["train"]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = criterion(forward_subnet_with_indices(m, x, active[cid], H), y)
                loss.backward()
                opt.step()
            hist["epoch_s"].append(time.time() - t0)

        if aggregation == "selective":
            states = [{k: v.clone() for k, v in cm.state_dict().items()} for cm in client_models]
            agg = selective_aggregate(global_state, states, active, weights, H)
        else:
            agg = fedavg_aggregate(client_models, weights)
        load_into_model(global_model, agg)

        if rnd % eval_every == 0 or rnd == 1 or rnd == rounds:
            accs, ppls = [], []
            ce = nn.CrossEntropyLoss(reduction="sum")
            for cid in range(bench.num_clients):
                temp = new_model(bench, ratios[cid], device)
                load_into_model(temp, agg)
                temp.eval()
                correct, n, loss_sum = 0, 0, 0.0
                with torch.no_grad():
                    for xb, yb in bench.clients[cid].loaders["test"]:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = forward_subnet_with_indices(temp, xb, active[cid], H)
                        correct += int((logits.argmax(1) == yb).sum().item())
                        n += int(yb.size(0))
                        loss_sum += float(ce(logits, yb).item())
                accs.append(correct / max(n, 1))
                ppls.append(float(np.exp(loss_sum / max(n, 1))))
            g_ood = accuracy(global_model, bench.global_ood_loader, device, use_subnet=False)
            g_infed = accuracy(global_model, bench.global_infed_loader, device, use_subnet=False)
            mean, worst, p10 = summarize(accs)
            hist["round"].append(rnd)
            hist["test_mean"].append(mean)
            hist["test_worst"].append(worst)
            hist["test_p10"].append(p10)
            hist["test_ppl"].append(float(np.mean(ppls)))
            hist["per_client_test"].append(accs)
            hist["global_infed"].append(g_infed)
            hist["global_ood"].append(g_ood)
            if log:
                log(f"[{label}] round {rnd:2d} | test mean {mean*100:.2f}% worst {worst*100:.2f}% P10 {p10*100:.2f}%")
    return hist
