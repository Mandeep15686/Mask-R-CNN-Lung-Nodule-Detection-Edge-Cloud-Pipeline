"""
CN3 Research: Federated Learning across hospital edge nodes.

Simulates LUNA16's 10 subsets as independent hospital 'silos'.
Uses the Flower (flwr) framework for FedAvg / FedProx.

Usage:
  # Terminal 1 — start FL server:
  python training/federated.py --mode server --rounds 50

  # Terminals 2-N — start clients (one per simulated hospital):
  python training/federated.py --mode client --client-id 0 --data ./data/luna16
  python training/federated.py --mode client --client-id 1 --data ./data/luna16
  ...

  # Simulation mode (all clients in one process):
  python training/federated.py --mode simulate --num-clients 5 --rounds 20
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from collections import OrderedDict

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flower client
# ---------------------------------------------------------------------------
def _flower_client(client_id: int, data_root: str, config: dict):
    try:
        import flwr as fl
    except ImportError:
        raise ImportError("Install Flower: pip install flwr")

    from .model import LungNoduleMaskRCNN
    from .dataset import build_dataloaders
    from .losses import CompositeMaskLoss
    from .metrics import evaluate_batch, MetricTracker
    import torch.optim as optim

    class LungNoduleFlowerClient(fl.client.NumPyClient):
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model  = LungNoduleMaskRCNN(
                num_classes=2,
                mask_loss_cfg=config.get("mask_loss", {"alpha": 0.3, "beta": 0.5, "gamma": 0.5}),
            ).to(self.device)
            self.criterion = CompositeMaskLoss.from_config(
                config.get("mask_loss", {"alpha": 0.3, "beta": 0.5, "gamma": 0.5})
            )
            self.train_loader, self.val_loader = build_dataloaders(
                data_root=data_root,
                fold=client_id % 10,
                batch_size=config.get("batch_size", 4),
                num_workers=0,
                patch_size=config.get("patch_size", 64),
            )
            logger.info(f"Client {client_id}: {len(self.train_loader)} train batches")

        def get_parameters(self, config):
            state = self.model.state_dict()
            self._parameter_keys = sorted(state)
            return [state[k].detach().cpu().numpy() for k in self._parameter_keys]

        def set_parameters(self, parameters):
            state = self.model.state_dict()
            keys = sorted(state)
            if len(parameters) != len(keys):
                raise ValueError(f"Expected {len(keys)} federated tensors, got {len(parameters)}")
            state_dict = OrderedDict(
                (k, torch.as_tensor(v, dtype=state[k].dtype, device=state[k].device))
                for k, v in zip(keys, parameters)
            )
            self.model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            self.model.train()
            lr       = config.get("lr", 0.005)
            n_epochs = config.get("local_epochs", 1)
            optimizer = optim.SGD(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=lr, momentum=0.9, weight_decay=1e-4,
            )
            tracker = MetricTracker()

            for _epoch in range(n_epochs):
                for batch in self.train_loader:
                    images  = batch["image"].to(self.device)
                    masks   = batch["mask"].to(self.device)
                    labels  = batch["label"].to(self.device)

                    targets = _build_targets(images, masks, labels, self.device)
                    optimizer.zero_grad()
                    loss_dict = self.model(list(images), targets)

                    if isinstance(loss_dict, dict):
                        total = sum(loss_dict.values())
                        tracker.update({k: v.item() for k, v in loss_dict.items()})
                        if total.requires_grad:
                            total.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                            optimizer.step()

            metrics = tracker.averages()
            return self.get_parameters(config={}), len(self.train_loader.dataset), metrics

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            self.model.eval()
            all_preds, all_targets = [], []

            with torch.no_grad():
                for batch in self.val_loader:
                    images  = batch["image"].to(self.device)
                    masks   = batch["mask"]
                    outputs = self.model(list(images))
                    for i, out in enumerate(outputs):
                        h, w = masks[i].shape
                        pred = np.zeros((h, w), dtype=np.float32)
                        if len(out.get("masks", [])) > 0 and len(out["scores"]) > 0:
                            best = int(out["scores"].argmax())
                            pred = out["masks"][best, 0].cpu().numpy()
                        all_preds.append(pred)
                        all_targets.append(masks[i].numpy())

            from .metrics import evaluate_batch
            if all_preds:
                seg = evaluate_batch(np.stack(all_preds), np.stack(all_targets))
                dice = seg.get("dice")
                loss = 1.0 - (float(dice) if dice is not None and np.isfinite(dice) else 0.0)
                return float(loss), len(self.val_loader.dataset), seg
            return 1.0, 0, {}

    return LungNoduleFlowerClient()


# ---------------------------------------------------------------------------
# Flower server strategy
# ---------------------------------------------------------------------------
def _build_strategy(strategy_name: str, num_clients: int):
    try:
        import flwr as fl
    except ImportError:
        raise ImportError("Install Flower: pip install flwr")

    # Weighted-average aggregation with evaluation metrics
    def evaluate_metrics_aggregation(eval_metrics):
        """Aggregate per-client evaluation metrics."""
        if not eval_metrics:
            return {}
        total_examples = sum(n for n, _ in eval_metrics)
        aggregated = {}
        for n, metrics in eval_metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    aggregated[k] = aggregated.get(k, 0.0) + v * (n / total_examples)
        return aggregated

    if strategy_name == "fedavg":
        return fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            on_fit_config_fn=lambda rnd: {"lr": 0.005, "local_epochs": 1, "round": rnd},
        )
    elif strategy_name == "fedprox":
        return fl.server.strategy.FedProx(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            proximal_mu=0.1,   # FedProx regularisation strength
            on_fit_config_fn=lambda rnd: {"lr": 0.005, "local_epochs": 1},
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Use 'fedavg' or 'fedprox'")


# ---------------------------------------------------------------------------
# Simulation mode: all clients in one process (for research/no-network setup)
# ---------------------------------------------------------------------------
def run_simulation(
    data_root: str,
    num_clients: int = 5,
    num_rounds: int = 20,
    strategy_name: str = "fedavg",
    config: Optional[dict] = None,
    output_dir: str = "./experiments/federated",
):
    """
    Run federated simulation with num_clients in-process.
    Equivalent to running server + all clients on one machine.
    """
    try:
        import flwr as fl
    except ImportError:
        raise ImportError("Install Flower: pip install flwr")

    import json
    from datetime import datetime

    if config is None:
        config = {
            "mask_loss":  {"alpha": 0.3, "beta": 0.5, "gamma": 0.5},
            "batch_size": 4,
            "patch_size": 64,
        }

    logger.info(f"=== Federated Simulation: {num_clients} clients | {num_rounds} rounds | {strategy_name} ===")

    history_log = []

    def client_fn(cid: str):
        return _flower_client(int(cid), data_root, config)

    strategy = _build_strategy(strategy_name, num_clients)

    # Monkey-patch strategy to log each round
    original_aggregate_evaluate = strategy.aggregate_evaluate

    def logged_aggregate(rnd, results, failures):
        agg = original_aggregate_evaluate(rnd, results, failures)
        log_entry = {
            "round": rnd,
            "strategy": strategy_name,
            "num_clients": num_clients,
        }
        if agg and len(agg) == 2:
            loss, metrics = agg
            log_entry["loss"] = float(loss) if loss else None
            log_entry.update({k: float(v) for k, v in (metrics or {}).items()})
        history_log.append(log_entry)
        logger.info(f"Round {rnd}/{num_rounds}: {log_entry}")
        # Save progress
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "fl_history.json"), "w") as f:
            json.dump(history_log, f, indent=2)
        return agg

    strategy.aggregate_evaluate = logged_aggregate

    # Run simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # Save final results
    final = {
        "strategy":    strategy_name,
        "num_clients": num_clients,
        "num_rounds":  num_rounds,
        "config":      config,
        "history":     history_log,
        "best_dice":   max((h.get("dice", 0) for h in history_log), default=0),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(output_dir, "fl_summary.json"), "w") as f:
        json.dump(final, f, indent=2)

    logger.info(f"Federated simulation complete. Results in {output_dir}")
    return final


# ---------------------------------------------------------------------------
# Target builder helper
# ---------------------------------------------------------------------------
def _build_targets(images, masks, labels, device):
    targets = []
    for i in range(images.shape[0]):
        t = {}
        if labels[i].item() == 1 and masks[i].sum() > 0:
            m = masks[i].cpu().numpy().astype(bool)
            rows = np.any(m, axis=1); cols = np.any(m, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            t["boxes"]  = torch.tensor([[cmin, rmin, cmax+1, rmax+1]], dtype=torch.float32, device=device)
            t["labels"] = torch.tensor([1], dtype=torch.int64, device=device)
            t["masks"]  = masks[i].unsqueeze(0).to(torch.uint8).to(device)
        else:
            H, W = masks[i].shape
            t["boxes"]  = torch.zeros((0, 4), dtype=torch.float32, device=device)
            t["labels"] = torch.zeros(0, dtype=torch.int64, device=device)
            t["masks"]  = torch.zeros((0, H, W), dtype=torch.uint8, device=device)
        targets.append(t)
    return targets


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [FL] %(message)s")
    parser = argparse.ArgumentParser(description="Federated Learning for Lung Nodule Detection")
    parser.add_argument("--mode",        choices=["server","client","simulate"], default="simulate")
    parser.add_argument("--data",        default="./data/luna16")
    parser.add_argument("--out",         default="./experiments/federated")
    parser.add_argument("--client-id",   type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds",      type=int, default=20)
    parser.add_argument("--strategy",    choices=["fedavg","fedprox"], default="fedavg")
    parser.add_argument("--server-addr", default="[::]:8080")
    args = parser.parse_args()

    if args.mode == "simulate":
        run_simulation(
            data_root=args.data,
            num_clients=args.num_clients,
            num_rounds=args.rounds,
            strategy_name=args.strategy,
            output_dir=args.out,
        )
    elif args.mode == "server":
        import flwr as fl
        strategy = _build_strategy(args.strategy, args.num_clients)
        fl.server.start_server(
            server_address=args.server_addr,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
    elif args.mode == "client":
        import flwr as fl
        client = _flower_client(args.client_id, args.data, {})
        fl.client.start_numpy_client(server_address=args.server_addr, client=client)
