"""Adapters reuse the repository's model topology without importing ROOT trainers."""

import copy
import json
from pathlib import Path
import numpy as np
from .config import resolve_path
from .data import fit_normalization


def lower_broadcasts(model):
    """Replay a standard Keras graph, explicitly replicating broadcast operands.

    Fixed all-ones Dense maps copy values (one input scalar per output). The
    trained Dense weights and every activation/pooling operation are preserved.
    """
    import tensorflow as tf

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

        for epoch in range(settings["epochs"]):
            order = rng.permutation(len(data["train"]["tracks"]))
            total = 0.0
            count = 0
            for start in range(0, len(order), settings["batch_size"]):
                ids = order[start : start + settings["batch_size"]]
                s = {k: v[ids] for k, v in data["train"].items()}
                with tf.GradientTape() as tape:
                    value = loss(s, True)
                variables = (
                    self.model.trainable_weights + self.decoder.trainable_weights
                )
                gradients = tape.gradient(value, variables)
                if not np.isfinite(float(value)):
                    raise ValueError("Nonfinite training loss")
                optimizer.apply_gradients(
                    (g, v) for g, v in zip(gradients, variables) if g is not None
                )
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
        return self.history

    def predict(self, split):
        return {"rho": self.model(self.inputs(split), training=False).numpy()}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(path / "encoder.weights.h5")
        self.decoder.save_weights(path / "decoder.weights.h5")
        np.savez(path / "normalization.npz", **self.norm)
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
