"""Adapters reuse the repository's model topology without importing ROOT trainers."""

import copy
import json
from pathlib import Path
import numpy as np
from .config import resolve_path
from .data import fit_normalization


def lower_broadcasts(model, implementation="indexed", relu_implementation="native"):
    """Clone the encoder with indexed broadcasting, or the legacy replicated form.

    Learned weights, activation boundaries and pooling are preserved.
    """
    import tensorflow as tf

    if implementation not in ("indexed", "replicated"):
        raise ValueError("Unknown broadcast implementation")
    from .indexed_broadcast import IndexedBroadcast, broadcast_shape

    if relu_implementation not in ("native", "controlled"):
        raise ValueError("Unknown ReLU implementation")
    layers = tf.keras.layers
    mapping = {}
    rewrites = []
    for tensor in model.inputs:
        mapping[tensor.ref()] = layers.Input(
            tuple(tensor.shape[1:]), dtype=tensor.dtype, name=tensor.name.split(":")[0]
        )
    for layer in model.layers:
        if isinstance(layer, layers.InputLayer):
            continue
        if len(layer._inbound_nodes) != 1:
            raise ValueError(
                "Broadcast lowering supports single-use layers in the supported builder"
            )
        node = layer._inbound_nodes[0]
        old_inputs = tf.nest.flatten(node.input_tensors)
        inputs = [mapping[x.ref()] for x in old_inputs]
        if isinstance(layer, (layers.Multiply, layers.Add)):
            shapes = [tuple(x.shape[1:]) for x in inputs]
            if any(s != shapes[0] for s in shapes):
                if any(len(s) != 2 or None in s for s in shapes):
                    raise ValueError(f"Unsupported broadcast ranks: {shapes}")
                target = tuple(max(s[d] for s in shapes) for d in range(2))
                if implementation == "indexed":
                    broadcast_shape(shapes)
                    output = IndexedBroadcast(
                        "multiply" if isinstance(layer, layers.Multiply) else "add",
                        name=layer.name,
                    )(inputs)
                    rewrites.append(
                        {
                            "layer": layer.name,
                            "original_shapes": shapes,
                            "expanded_shape": target,
                            "implementation": "indexed",
                        }
                    )
                    for old, new in zip(
                        tf.nest.flatten(node.output_tensors), tf.nest.flatten(output)
                    ):
                        mapping[old.ref()] = new
                    continue
                expanded = []
                for i, (x, shape) in enumerate(zip(inputs, shapes)):
                    prefix = f"{layer.name}_replica_{i}"
                    if any(a not in (1, b) for a, b in zip(shape, target)):
                        raise ValueError(f"Illegal broadcast: {shapes}")
                    if shape[1] != target[1]:
                        x = layers.Dense(
                            target[1],
                            use_bias=False,
                            trainable=False,
                            kernel_initializer="ones",
                            name=prefix + "_features",
                        )(x)
                    if shape[0] != target[0]:
                        x = layers.Permute((2, 1), name=prefix + "_transpose_in")(x)
                        x = layers.Dense(
                            target[0],
                            use_bias=False,
                            trainable=False,
                            kernel_initializer="ones",
                            name=prefix + "_tracks",
                        )(x)
                        x = layers.Permute((2, 1), name=prefix + "_transpose_out")(x)
                    expanded.append(x)
                inputs = expanded
                rewrites.append(
                    {
                        "layer": layer.name,
                        "original_shapes": shapes,
                        "expanded_shape": target,
                    }
                )
        if isinstance(layer, layers.ReLU) and relu_implementation == "controlled":
            from .controlled_relu import ControlledReLU

            c = layer.get_config()
            if (
                c.get("max_value") is not None
                or c.get("negative_slope", 0) != 0
                or c.get("threshold", 0) != 0
            ):
                raise ValueError("Controlled conversion supports standard ReLU only")
            cloned = ControlledReLU(
                name=layer.name, trainable=layer.trainable, dtype=layer.dtype
            )
        else:
            cloned = layer.__class__.from_config(layer.get_config())
        output = cloned(
            inputs
            if isinstance(layer, (layers.Multiply, layers.Add, layers.Concatenate))
            else inputs[0]
        )
        cloned.set_weights(layer.get_weights())
        for old, new in zip(
            tf.nest.flatten(node.output_tensors), tf.nest.flatten(output)
        ):
            mapping[old.ref()] = new
    out = [mapping[t.ref()] for t in model.outputs]
    return tf.keras.Model(
        [mapping[t.ref()] for t in model.inputs],
        out[0] if len(out) == 1 else out,
        name="deepset_lowered",
    ), rewrites


class DeepSetAdapter:
    kind = "deepset_film"

    def __init__(self, cfg, data):
        import tensorflow as tf
        from AutoEncoder import build_deepset_film, build_set_decoder

        tf.keras.backend.clear_session()
        self.cfg = copy.deepcopy(cfg)
        self.norm = fit_normalization(data["train"])
        m = cfg["model"]
        self.history = []
        self.diagnostics = {}
        self.diagnostic_arrays = {}
        tf.keras.utils.set_random_seed(cfg["run"]["seed"])
        self.model = build_deepset_film(
            m["max_tracks"],
            m["n_features"],
            m["n_context"],
            phi_dim=m["phi_dim"],
            attention_hidden_dim=m["attention_hidden_dim"],
            rho_dim=m["rho_dim"],
            n_hidden_layers_tracks=m["track_layers"],
            trk_shift=self.norm["track_shift"],
            trk_scale=self.norm["track_scale"],
            event_shift=self.norm["context_shift"],
            event_scale=self.norm["context_scale"],
            vae_output=False,
        )
        self.decoder = build_set_decoder(
            m["rho_dim"],
            m["max_tracks"],
            m["n_features"],
            hidden_dim=cfg["training"]["decoder_hidden"],
        )

    def inputs(self, split):
        return [
            np.ascontiguousarray(split[k], dtype=np.float32)
            for k in ("tracks", "mask", "context")
        ]

    def fit(self, data, settings):
        import tensorflow as tf
        from VICReg_utility import chamfer_loss

        optimizer = tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"])
        rng = np.random.default_rng(self.cfg["run"]["seed"])
        shift = tf.constant(self.norm["track_shift"])
        scale = tf.constant(self.norm["track_scale"])

        def loss(s, training):
            rho = self.model(self.inputs(s), training=training)
            prediction = self.decoder(rho, training=training)
            return chamfer_loss((s["tracks"] - shift) / scale, prediction, s["mask"])

        from .diagnostics import (
            array_statistics,
            chamfer_per_example,
            normalized_targets,
        )

        initial_encoder = [v.numpy().copy() for v in self.model.trainable_weights]
        initial_val = float(loss(data["validation"], False))
        best_loss = initial_val
        best_epoch = 0
        steps = 0
        best_weights = (self.model.get_weights(), self.decoder.get_weights())
        for epoch in range(settings["epochs"]):
            order = rng.permutation(len(data["train"]["tracks"]))
            total = 0.0
            count = 0
            for start in range(0, len(order), settings["batch_size"]):
                ids = order[start : start + settings["batch_size"]]
                split = {k: v[ids] for k, v in data["train"].items()}
                with tf.GradientTape() as tape:
                    value = loss(split, True)
                variables = (
                    self.model.trainable_weights + self.decoder.trainable_weights
                )
                gradients = tape.gradient(value, variables)
                if not np.isfinite(float(value)):
                    raise ValueError("Nonfinite training loss")
                for gradient in gradients:
                    if gradient is not None:
                        tf.debugging.assert_all_finite(gradient, "Nonfinite gradient")
                optimizer.apply_gradients(
                    (g, v) for g, v in zip(gradients, variables) if g is not None
                )
                steps += 1
                total += float(value) * len(ids)
                count += len(ids)
            val = float(loss(data["validation"], False))
            if not np.isfinite(val):
                raise ValueError("Nonfinite validation loss")
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total / count,
                    "validation_loss": val,
                }
            )
            if val < best_loss:
                best_loss = val
                best_epoch = epoch + 1
                best_weights = (self.model.get_weights(), self.decoder.get_weights())

        def delta():
            return float(
                np.sqrt(
                    sum(
                        np.sum((v.numpy().astype(np.float64) - a) ** 2)
                        for v, a in zip(self.model.trainable_weights, initial_encoder)
                    )
                )
            )

        last_delta = delta()
        self.model.set_weights(best_weights[0])
        self.decoder.set_weights(best_weights[1])
        validation = self.predict(data["validation"])
        physical = normalized_targets(self, data["train"])[
            data["train"]["mask"][..., 0] > 0
        ]
        center = physical.mean(0)
        constant = np.broadcast_to(center, data["validation"]["tracks"].shape)
        baseline = chamfer_per_example(
            normalized_targets(self, data["validation"]),
            constant,
            data["validation"]["mask"],
        )
        gate_model = tf.keras.Model(
            self.model.inputs, self.model.get_layer("attention_mask").output
        )
        gate = gate_model(self.inputs(data["validation"]), training=False).numpy()
        learned = np.concatenate(
            [v.numpy().ravel() for v in self.model.trainable_weights]
        )
        decoder = np.concatenate(
            [v.numpy().ravel() for v in self.decoder.trainable_weights]
        )
        self.diagnostic_arrays = {
            "encoder_weights": learned,
            "host_decoder_weights": decoder,
            "rho_validation": validation["rho"],
            "gate_validation": gate,
            "constant_track": center,
        }
        rho_stats = array_statistics(validation["rho"])
        rho_std = np.std(validation["rho"], axis=0)
        self.diagnostics = {
            "optimizer_steps": steps,
            "initial_validation_chamfer": initial_val,
            "best_validation_chamfer": best_loss,
            "best_epoch": best_epoch,
            "constant_validation_chamfer": float(baseline.mean()),
            "last_encoder_delta_l2": last_delta,
            "selected_encoder_delta_l2": delta(),
            "encoder_weights": array_statistics(learned),
            "host_decoder_weights": array_statistics(decoder),
            "rho_validation": rho_stats,
            "rho_std_per_coordinate": rho_std.tolist(),
            "gate_active_fraction_valid_tracks": float(
                np.mean(gate[data["validation"]["mask"] > 0] > 0)
            ),
            "warnings": [],
        }
        if not np.any(rho_std > 1e-8):
            self.diagnostics["warnings"].append("Collapsed validation embeddings")
        if best_epoch == 0:
            self.diagnostics["warnings"].append(
                "Validation selected the initial weights; training did not improve validation"
            )
        if best_loss >= baseline.mean():
            self.diagnostics["warnings"].append(
                "Validation reconstruction does not beat the constant-mean baseline"
            )
        return self.history

    def decode(self, rho):
        chunks = []
        batch = self.cfg["training"]["batch_size"]
        for start in range(0, len(rho), batch):
            chunks.append(
                self.decoder(rho[start : start + batch], training=False).numpy()
            )
        return np.concatenate(chunks)

    def predict(self, split):
        chunks = []
        batch = self.cfg["training"]["batch_size"]
        inputs = self.inputs(split)
        for start in range(0, len(inputs[0]), batch):
            chunks.append(
                self.model(
                    [x[start : start + batch] for x in inputs], training=False
                ).numpy()
            )
        rho = np.concatenate(chunks)
        return {"rho": rho, "reconstruction": self.decode(rho)}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(path / "encoder.weights.h5")
        self.decoder.save_weights(path / "decoder.weights.h5")
        np.savez(path / "normalization.npz", **self.norm)
        (path / "training_diagnostics.json").write_text(
            json.dumps(self.diagnostics, indent=2, allow_nan=False)
        )
        np.savez_compressed(path / "diagnostics.npz", **self.diagnostic_arrays)
        (path / "model.json").write_text(
            json.dumps(
                {
                    "kind": self.kind,
                    "model": self.cfg["model"],
                    "training": self.cfg["training"],
                    "history": self.history,
                    "format_version": 1,
                    "optimizer_state_saved": False,
                },
                indent=2,
            )
        )

    def restore(self, path):
        self.model.load_weights(Path(path) / "encoder.weights.h5")
        self.decoder.load_weights(Path(path) / "decoder.weights.h5")
        with np.load(Path(path) / "normalization.npz", allow_pickle=False) as z:
            self.norm = {k: z[k] for k in z.files}
        metadata = json.loads((Path(path) / "model.json").read_text())
        self.history = metadata.get("history", [])
        diag = Path(path) / "training_diagnostics.json"
        if diag.exists():
            self.diagnostics = json.loads(diag.read_text())
        arrays = Path(path) / "diagnostics.npz"
        if arrays.exists():
            with np.load(arrays, allow_pickle=False) as z:
                self.diagnostic_arrays = {k: z[k] for k in z.files}


def graph_arrays(split):
    """One graph per example; sort by Euclidean feature norm, not a physics pT claim."""
    nodes = []
    edges = []
    senders = []
    receivers = []
    counts = []
    ecounts = []
    for x, mask in zip(split["tracks"], split["mask"]):
        x = x[mask[:, 0] > 0]
        x = x[np.argsort(-np.linalg.norm(x, axis=1), kind="stable")]
        n = len(x)
        if n < 2:
            raise ValueError(
                "GraphNet synthetic profile requires >=2 real tracks per example"
            )
        r, s = np.nonzero(~np.eye(n, dtype=bool))
        nodes.append(x)
        edges.append(x[s] - x[r])
        senders.extend(s)
        receivers.extend(r)
        counts.append(n)
        ecounts.append(len(s))
    f = split["tracks"].shape[-1]
    c = split["context"].shape[-1]
    n = len(counts)
    return {
        "node_features": np.concatenate(nodes),
        "edge_features": np.concatenate(edges),
        "senders": np.asarray(senders, np.int64),
        "receivers": np.asarray(receivers, np.int64),
        "global_features": split["context"],
        "n_nodes": np.array(counts, np.int32),
        "n_edges": np.array(ecounts, np.int32),
        "event_index": np.arange(n, dtype=np.int32),
        "jet_index": np.zeros(n, np.int32),
        "n_events": np.array(n),
        "node_feature_names": np.array([f"random_{i}" for i in range(f)]),
        "edge_feature_names": np.array([f"d_random_{i}" for i in range(f)]),
        "global_feature_names": np.array([f"context_{i}" for i in range(c)]),
    }


class GraphAdapter:
    kind = "graphnet"

    def __init__(self, cfg, data):
        import torch
        from GraphNet import GraphAutoEncoder

        self.cfg = copy.deepcopy(cfg)
        m = cfg["model"]
        self.history = []
        torch.manual_seed(cfg["run"]["seed"])
        g = graph_arrays(data["train"])

        def stats(a):
            low = a.min(0)
            high = a.max(0)
            return (low + high) / 2, np.where(high > low, (high - low) / 2, 1).astype(
                np.float32
            )

        self.norm = (
            *stats(g["node_features"]),
            *stats(g["edge_features"]),
            *stats(g["global_features"]),
        )
        self.graph_config = {
            "n_node": m["n_features"],
            "n_edge": m["n_features"],
            "n_glob": m["n_context"],
            "hidden": m["hidden"],
            "latent_dim": m["latent_dim"],
            "encoder_blocks": m["encoder_blocks"],
            "decoder_blocks": m["decoder_blocks"],
            "mlp_layers": m["mlp_layers"],
            "max_nodes": m["max_tracks"],
        }
        self.model = GraphAutoEncoder(self.graph_config)

    def fit(self, data, settings):
        import torch
        from GraphNet import GraphData, run_epoch

        train = GraphData(graph_arrays(data["train"]), self.norm)
        val = GraphData(graph_arrays(data["validation"]), self.norm)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=settings["learning_rate"]
        )
        for epoch in range(settings["epochs"]):
            a = run_epoch(self.model, train, settings["batch_size"], "cpu", optimizer)
            b = run_epoch(self.model, val, settings["batch_size"], "cpu")
            if not np.isfinite([a, b]).all():
                raise ValueError("Nonfinite GraphNet loss")
            self.history.append(
                {"epoch": epoch + 1, "train_loss": a, "validation_loss": b}
            )
        return self.history

    def predict(self, split):
        import torch
        from GraphNet import GraphData, per_graph_error

        data = GraphData(graph_arrays(split), self.norm)
        recon = []
        latent = []
        score = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(data), self.cfg["training"]["batch_size"]):
                batch = data.batch(
                    torch.arange(
                        start,
                        min(start + self.cfg["training"]["batch_size"], len(data)),
                    )
                )
                x, z = self.model(batch)
                recon.append(x.numpy())
                latent.append(z.numpy())
                score.append(per_graph_error(x, batch).numpy())
        return {
            "reconstruction": np.concatenate(recon),
            "latent": np.concatenate(latent),
            "score": np.concatenate(score),
        }

    def save(self, path):
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        np.savez(
            path / "normalization.npz", **{str(i): v for i, v in enumerate(self.norm)}
        )
        (path / "model.json").write_text(
            json.dumps(
                {
                    "kind": self.kind,
                    "model": self.cfg["model"],
                    "training": self.cfg["training"],
                    "history": self.history,
                    "format_version": 1,
                    "optimizer_state_saved": False,
                },
                indent=2,
            )
        )

    def restore(self, path):
        import torch

        self.model.load_state_dict(
            torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=True)
        )
        with np.load(Path(path) / "normalization.npz", allow_pickle=False) as z:
            self.norm = tuple(z[str(i)] for i in range(6))


def create_adapter(config, data):
    cfg = copy.deepcopy(config)
    checkpoint = cfg["model"]["checkpoint"]
    if checkpoint:
        path = resolve_path(cfg, checkpoint)
        saved = json.loads((path / "model.json").read_text())
        if saved["format_version"] != 1 or saved["kind"] != cfg["model"]["kind"]:
            raise ValueError("Checkpoint kind/format mismatch")
        for key in ("max_tracks", "n_features", "n_context"):
            if saved["model"][key] != cfg["model"][key]:
                raise ValueError(f"Checkpoint shape mismatch: {key}")
        # Saved architecture wins: loading never silently builds a different width.
        cfg["model"] = saved["model"]
        cfg["training"]["decoder_hidden"] = saved["training"]["decoder_hidden"]
    cls = DeepSetAdapter if cfg["model"]["kind"] == "deepset_film" else GraphAdapter
    adapter = cls(cfg, data)
    if checkpoint:
        adapter.restore(path)
    return adapter
