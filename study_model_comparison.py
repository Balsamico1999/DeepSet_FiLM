"""Compare ONNX anomaly scores on nominal and LLP-injected validation jets.

The evaluation uses the intersection of DeepSet's held-out fold and GraphNet's
held-out event fraction so both models are compared on the same events.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
import torch
import yaml

from DataProcessor import DataProcessor
from GraphNet import GraphData
from LLPMaker import LIGHT_SPEED, LLPMaker


@dataclass
class JetSamples:
    tracks: np.ndarray
    events: np.ndarray
    jets: np.ndarray

    def __len__(self):
        return len(self.tracks)

    @property
    def event_context(self):
        return np.concatenate((self.events, self.jets), axis=1).astype(
            np.float32, copy=False
        )


@dataclass
class EvaluationData:
    reference: JetSamples
    background: JetSamples
    signal: JetSamples


class ModelAdapter(Protocol):
    label: str

    @classmethod
    def from_config(
        cls, settings: dict, training: dict, processor: DataProcessor
    ) -> "ModelAdapter":
        """Build this adapter from comparison settings and the data processor."""
        ...

    def score(
        self, data: EvaluationData
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return anomaly scores for nominal validation jets and LLP jets."""


MODEL_ADAPTERS: dict[str, type[ModelAdapter]] = {}


def register_adapter(name: str):
    def register(factory):
        if name in MODEL_ADAPTERS:
            raise ValueError(f"Model adapter '{name}' is already registered")
        MODEL_ADAPTERS[name] = factory
        return factory

    return register


def encode_phi(tracks, track_columns):
    if "trk_phi" not in track_columns:
        return tracks

    phi_index = track_columns.index("trk_phi")
    valid = np.any(tracks != 0, axis=-1)
    phi = tracks[..., phi_index]
    encoded = np.concatenate(
        (
            tracks[..., :phi_index],
            np.sin(phi)[..., np.newaxis],
            np.cos(phi)[..., np.newaxis],
            tracks[..., phi_index + 1:],
        ),
        axis=-1,
    )
    return np.where(valid[..., np.newaxis], encoded, 0)


def onnx_input_shapes(session):
    return {
        item.name: tuple(item.shape)
        for item in session.get_inputs()
    }


def compute_roc(background_scores, signal_scores):
    """Build a descending-score ROC curve and its trapezoidal area."""
    if len(background_scores) == 0 or len(signal_scores) == 0:
        raise ValueError("ROC calculation needs both background and signal scores")
    labels = np.concatenate(
        (
            np.zeros(len(background_scores), dtype=np.int8),
            np.ones(len(signal_scores), dtype=np.int8),
        )
    )
    scores = np.concatenate((background_scores, signal_scores))
    if not np.all(np.isfinite(scores)):
        raise ValueError("Anomaly scores must all be finite to calculate a ROC curve")

    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    threshold_indices = np.flatnonzero(np.diff(sorted_scores))
    threshold_indices = np.concatenate((threshold_indices, [len(sorted_scores) - 1]))
    true_positives = np.cumsum(sorted_labels, dtype=np.int64)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    true_positives = np.concatenate(([0], true_positives))
    false_positives = np.concatenate(([0], false_positives))
    thresholds = np.concatenate(([np.inf], sorted_scores[threshold_indices]))
    tpr = true_positives / len(signal_scores)
    fpr = false_positives / len(background_scores)
    area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    return fpr, tpr, thresholds, float(area)


@register_adapter("deepset_film")
class DeepSetFiLMAdapter:
    label = "DeepSet + FiLM"

    @classmethod
    def from_config(cls, settings, training, processor):
        return cls(
            Path("deepset_film.onnx"),
            processor,
            settings["BATCH_SIZE"],
        )

    def __init__(self, model_path, processor, batch_size):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.batch_size = batch_size
        self.shapes = onnx_input_shapes(self.session)
        required = {"tracks", "mask", "event"}
        if set(self.shapes) != required:
            raise ValueError(
                f"DeepSet ONNX inputs must be {sorted(required)}; "
                f"got {sorted(self.shapes)}"
            )
        self.max_tracks = self.shapes["tracks"][1]
        self.n_track_features = self.shapes["tracks"][2]
        self.n_event_features = self.shapes["event"][1]
        self.track_columns = processor.trk_columns
        expected_context_features = len(processor.event_columns) + len(
            processor.jets_columns
        )
        if self.n_event_features != expected_context_features:
            raise ValueError(
                "DeepSet event input width does not match configured event + jet features"
            )

    def encode(self, samples):
        tracks = encode_phi(samples.tracks, self.track_columns)
        if tracks.shape[1] < self.max_tracks:
            tracks = np.pad(
                tracks,
                ((0, 0), (0, self.max_tracks - tracks.shape[1]), (0, 0)),
            )
        tracks = np.ascontiguousarray(tracks[:, :self.max_tracks], dtype=np.float32)
        events = np.ascontiguousarray(samples.event_context, dtype=np.float32)
        if tracks.shape[2] != self.n_track_features:
            raise ValueError(
                f"DeepSet expects {self.n_track_features} track features, "
                f"got {tracks.shape[2]}"
            )
        if events.shape[1] != self.n_event_features:
            raise ValueError(
                f"DeepSet expects {self.n_event_features} event-context features, "
                f"got {events.shape[1]}"
            )

        mask = np.ascontiguousarray(
            np.any(tracks != 0, axis=-1, keepdims=True), dtype=np.float32
        )
        latent = []
        output_name = self.session.get_outputs()[0].name
        for start in range(0, len(samples), self.batch_size):
            stop = min(start + self.batch_size, len(samples))
            output = self.session.run(
                [output_name],
                {
                    "tracks": tracks[start:stop],
                    "mask": mask[start:stop],
                    "event": events[start:stop],
                },
            )[0]
            latent.append(output)
        return np.concatenate(latent, axis=0)

    @staticmethod
    def mahalanobis_scores(latent, mean, covariance_inverse):
        delta = latent - mean
        return np.einsum("...i,ij,...j->...", delta, covariance_inverse, delta)

    def score(self, data):
        reference_latent = self.encode(data.reference)
        if len(reference_latent) < 2:
            raise ValueError("At least two common training events are needed to calibrate DeepSet")
        mean = reference_latent.mean(axis=0)
        covariance_inverse = np.linalg.pinv(
            np.cov(reference_latent, rowvar=False), hermitian=True
        )
        background = self.encode(data.background)
        signal = self.encode(data.signal)
        return (
            self.mahalanobis_scores(background, mean, covariance_inverse),
            self.mahalanobis_scores(signal, mean, covariance_inverse),
        )


@register_adapter("graphnet")
class GraphNetAdapter:
    label = "GraphNet"
    input_names = (
        "node", "edge", "senders", "receivers", "node_graph", "edge_graph",
        "node_rank", "glob", "n_nodes", "dec_senders", "dec_receivers",
        "dec_edge_graph",
    )

    @classmethod
    def from_config(cls, settings, training, processor):
        return cls(
            Path(settings["OUTPUT_DIR"]) / "graphnet_ae.onnx",
            processor,
            settings["MIN_TRACKS"],
            settings["K_NEIGHBORS"],
            settings["PHI_ENCODING"],
            training["BATCH_SIZE"],
        )

    def __init__(
        self,
        model_path,
        graph_builder,
        min_tracks,
        k_neighbors,
        phi_encoding,
        batch_size,
    ):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.graph_builder = graph_builder
        self.min_tracks = min_tracks
        self.k_neighbors = k_neighbors
        self.phi_encoding = phi_encoding
        self.batch_size = batch_size

        metadata = self.session.get_modelmeta().custom_metadata_map
        try:
            self.config = json.loads(metadata["config"])
            self.normalization = json.loads(metadata["normalization"])
            self.feature_names = json.loads(metadata["feature_names"])
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(
                "GraphNet ONNX must include config, normalization, and feature_names metadata"
            ) from error
        expected_widths = {
            "node_feature_names": "n_node",
            "edge_feature_names": "n_edge",
            "global_feature_names": "n_glob",
        }
        for feature_key, config_key in expected_widths.items():
            if len(self.feature_names[feature_key]) != self.config[config_key]:
                raise ValueError(
                    f"GraphNet metadata has {len(self.feature_names[feature_key])} "
                    f"{feature_key} but config expects {self.config[config_key]}"
                )

        actual_inputs = {item.name for item in self.session.get_inputs()}
        if actual_inputs != set(self.input_names):
            raise ValueError(
                f"Unexpected GraphNet ONNX inputs: {sorted(actual_inputs)}"
            )
        self.norm = tuple(
            np.asarray(self.normalization[key], dtype=np.float32)
            for key in (
                "node_shift", "node_scale", "edge_shift", "edge_scale",
                "glob_shift", "glob_scale",
            )
        )

    def make_graph_data(self, samples):
        tracks = np.asarray(samples.tracks, dtype=np.float32)
        counts = np.any(tracks != 0, axis=-1).sum(axis=1)
        graphs = self.graph_builder.jet_matched_arrays_to_graphs(
            tracks[:, np.newaxis, :, :],
            counts[:, np.newaxis],
            samples.events,
            samples.jets[:, np.newaxis, :],
            min_tracks=self.min_tracks,
            k_neighbors=self.k_neighbors,
            sort_by="trk_px",
            phi_encoding=self.phi_encoding,
        )

        for key in ("node_feature_names", "edge_feature_names", "global_feature_names"):
            actual = graphs[key].tolist()
            expected = self.feature_names[key]
            if actual != expected:
                raise ValueError(
                    f"GraphNet {key} mismatch: model expects {expected}, got {actual}. "
                    "Use the same graph feature options used during training."
                )
        return GraphData(graphs, self.norm)

    def score_samples(self, samples):
        scores = []
        for start in range(0, len(samples), self.batch_size):
            stop = min(start + self.batch_size, len(samples))
            graph_data = self.make_graph_data(
                JetSamples(
                    samples.tracks[start:stop],
                    samples.events[start:stop],
                    samples.jets[start:stop],
                )
            )
            if len(graph_data) != stop - start:
                raise ValueError(
                    "GraphNet preprocessing dropped input jets; check the minimum "
                    "track count and feature configuration."
                )
            for batch_start in range(0, len(graph_data), self.batch_size):
                ids = torch.arange(
                    batch_start, min(batch_start + self.batch_size, len(graph_data))
                )
                batch = graph_data.batch(ids)
                inputs = {
                    name: getattr(batch, name).numpy()
                    for name in self.input_names
                }
                reconstruction = self.session.run(
                    ["reconstruction"], inputs
                )[0]
                reconstruction_tensor = torch.tensor(
                    reconstruction.tolist(), dtype=batch.node.dtype
                )
                squared_error = (
                    reconstruction_tensor - batch.node
                ).square().mean(dim=1)
                graph_error = torch.zeros(
                    batch.num_graphs, dtype=squared_error.dtype
                ).index_add_(
                    0, batch.node_graph, squared_error
                )
                graph_scores = (
                    graph_error
                    / batch.n_nodes.clamp(min=1).to(squared_error.dtype)
                )
                scores.append(graph_scores.tolist())
        return np.concatenate(scores)

    def score(self, data):
        return (
            self.score_samples(data.background),
            self.score_samples(data.signal),
        )


def load_evaluation_data(settings, training, llp_settings, config):
    import ROOT

    root_path = Path(settings["ROOT_FILE"])
    if not root_path.is_file():
        raise FileNotFoundError(f"ROOT input file does not exist: {root_path}")
    root_file = ROOT.TFile.Open(str(root_path))
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open ROOT input file: {root_path}")
    tree = root_file.Get(settings["TREE"])
    if not tree:
        raise ValueError(f"ROOT tree '{settings['TREE']}' was not found in {root_path}")

    inputs = config["Inputs"]
    max_events = settings["MAX_EVENTS"]
    processor = DataProcessor(
        tree,
        inputs["trk_columns"],
        inputs["event_columns"],
        jets_columns=inputs["jet_columns"],
        variables_to_define=inputs["variables_to_define"],
        max_tracks=settings["MAX_TRACKS"],
        max_events=max_events,
        filepath=(
            inputs["JET_MATCHED_ARRAYS_FILEPATH"]
            if inputs["LOAD_ARRAYS"]
            else None
        ),
    )
    tracks, events, jets = processor.get_jet_matched_npy_arrays(
        cut=settings["CUT"], max_jets=1
    )
    tracks = tracks[:max_events]
    events = events[:max_events]
    jets = jets[:max_events]
    counts = processor.jet_matched_track_counts[:max_events, 0]
    valid = (
        (counts >= settings["MIN_TRACKS"])
        & np.any(jets[:, 0] != 0, axis=-1)
    )

    event_indices = np.arange(len(tracks))
    n_graphnet_val = int(
        np.round(settings["VAL_FRACTION"] * len(tracks))
    )
    deepset_val = event_indices % training["K_FOLDS"] == 0
    graphnet_val = event_indices < n_graphnet_val
    common_val = valid & deepset_val & graphnet_val
    common_train = valid & ~deepset_val & ~graphnet_val
    if not np.any(common_val):
        raise ValueError(
            "The models' validation splits do not overlap after selecting valid jets; "
            "check GraphNet.MAX_EVENTS, Training.K_FOLDS, and GraphNet.VAL_FRACTION."
        )
    if np.count_nonzero(common_train) < 2:
        raise ValueError(
            "Fewer than two events are held out from both models for DeepSet "
            "score calibration; increase Training.MAX_EVENTS or GraphNet.MAX_EVENTS."
        )

    def select(mask):
        return JetSamples(
            tracks[mask, 0].astype(np.float32, copy=False),
            events[mask].astype(np.float32, copy=False),
            jets[mask, 0].astype(np.float32, copy=False),
        )

    reference = select(common_train)
    background = select(common_val)
    np.random.seed(settings["SEED"])
    maker = LLPMaker(background.tracks, background.jets[:, np.newaxis, :], inputs["trk_columns"])
    maker.rng = np.random.default_rng(settings["SEED"])
    injected_tracks, generated_tracks = maker.sample_LLP_tracks(
        mass=llp_settings["MASS_GEV"],
        lifetime=llp_settings["PROPER_LIFETIME_MM"] / LIGHT_SPEED,
        alpha=llp_settings["ALPHA"],
        beta=llp_settings["BETA"],
        sig_0=llp_settings["SIG_0"],
        a=llp_settings["OPENING_POWER"],
        opposite_charge_fermions=True,
        different_fermion_flavors=False,
        n_llps_mean=llp_settings["N_LLPS_MEAN"],
        n_llps_std=llp_settings["N_LLPS_STD"],
    )
    has_injected_tracks = np.any(generated_tracks != 0, axis=(1, 2))
    injected_track_counts = np.any(injected_tracks != 0, axis=-1).sum(axis=1)
    graphnet_compatible = injected_track_counts >= settings["MIN_TRACKS"]
    accepted_signal = has_injected_tracks & graphnet_compatible
    skipped_signal = np.count_nonzero(has_injected_tracks & ~graphnet_compatible)
    if not np.any(accepted_signal):
        raise ValueError(
            "LLPMaker produced no LLP-injected jets with enough tracks for GraphNet; "
            "adjust the injection settings or GraphNet.MIN_TRACKS."
        )
    signal = JetSamples(
        injected_tracks[accepted_signal].astype(np.float32, copy=False),
        background.events[accepted_signal],
        background.jets[accepted_signal],
    )
    if skipped_signal:
        print(
            f"Skipped {skipped_signal} LLP-injected jets with fewer than "
            f"{settings['MIN_TRACKS']} tracks after injection."
        )
    print(
        f"Loaded {len(tracks)} events ({np.count_nonzero(valid)} valid leading jets); "
        f"{len(reference)} common training-reference jets, "
        f"{len(background)} common validation jets, "
        f"{len(signal)} with accepted LLP tracks."
    )
    return EvaluationData(reference, background, signal), root_file, processor


def validate_config(config):
    settings = config["GraphNet"]
    training = config["Training"]
    llp = config["LLP"]

    if training["K_FOLDS"] <= 0 or training["BATCH_SIZE"] <= 0:
        raise ValueError("Training.K_FOLDS and Training.BATCH_SIZE must be positive")
    if settings["MAX_EVENTS"] <= 0:
        raise ValueError("GraphNet.MAX_EVENTS must be positive")
    if settings["BATCH_SIZE"] <= 0 or settings["MAX_TRACKS"] <= 0:
        raise ValueError("GraphNet.BATCH_SIZE and GraphNet.MAX_TRACKS must be positive")
    if not 0 < settings["VAL_FRACTION"] < 1:
        raise ValueError("GraphNet.VAL_FRACTION must be between 0 and 1")
    if settings["MIN_TRACKS"] < 1:
        raise ValueError("GraphNet.MIN_TRACKS must be positive")
    if settings["K_NEIGHBORS"] is not None and settings["K_NEIGHBORS"] < 1:
        raise ValueError("GraphNet.K_NEIGHBORS must be positive when set")
    if settings["PHI_ENCODING"] not in ("raw", "sincos"):
        raise ValueError("GraphNet.PHI_ENCODING must be 'raw' or 'sincos'")
    if settings["MAX_JETS"] < 1:
        raise ValueError("GraphNet.MAX_JETS must be positive")
    if llp["MASS_GEV"] <= 0.211316:
        raise ValueError("LLP.MASS_GEV must exceed 0.211316 GeV")
    if (
        llp["PROPER_LIFETIME_MM"] <= 0
        or llp["ALPHA"] <= 0
        or llp["BETA"] <= 0
    ):
        raise ValueError("LLP lifetime and beta-distribution shapes must be positive")
    if (
        llp["SIG_0"] < 0
        or llp["N_LLPS_MEAN"] < 0
        or llp["N_LLPS_STD"] < 0
    ):
        raise ValueError("LLP widths and multiplicities must be non-negative")


def evaluate_models(settings, training, data, processor):
    if not MODEL_ADAPTERS:
        raise ValueError("No model adapters are registered")
    results = {}
    for name, adapter_type in MODEL_ADAPTERS.items():
        adapter = adapter_type.from_config(settings, training, processor)
        background_scores, signal_scores = adapter.score(data)
        if len(background_scores) != len(data.background):
            raise ValueError(f"{name} returned the wrong number of background scores")
        if len(signal_scores) != len(data.signal):
            raise ValueError(f"{name} returned the wrong number of signal scores")
        fpr, tpr, thresholds, model_auc = compute_roc(
            background_scores, signal_scores
        )
        results[name] = {
            "label": adapter.label,
            "background_scores": background_scores,
            "signal_scores": signal_scores,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": float(model_auc),
        }
        print(f"{adapter.label}: ROC AUC = {model_auc:.4f}")
    return results


def save_results(settings, training, llp_settings, data, results):
    import matplotlib.pyplot as plt

    outdir = Path(settings["OUTPUT_DIR"]) / "model_comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    for result in results.values():
        plt.plot(
            result["fpr"],
            result["tpr"],
            label=f'{result["label"]} (AUC = {result["auc"]:.3f})',
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Nominal vs LLP-injected leading jets")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "roc_comparison.png", dpi=160)
    plt.close()

    arrays = {}
    metrics = {
        "background_jets": len(data.background),
        "injected_jets": len(data.signal),
        "training_reference_jets": len(data.reference),
        "validation": {
            "deepset_folds": training["K_FOLDS"],
            "graphnet_val_fraction": settings["VAL_FRACTION"],
        },
        "llp_injection": {
            "mass_gev": llp_settings["MASS_GEV"],
            "proper_lifetime_mm": llp_settings["PROPER_LIFETIME_MM"],
            "alpha": llp_settings["ALPHA"],
            "beta": llp_settings["BETA"],
            "sig_0": llp_settings["SIG_0"],
            "opening_power": llp_settings["OPENING_POWER"],
            "n_llps_mean": llp_settings["N_LLPS_MEAN"],
            "n_llps_std": llp_settings["N_LLPS_STD"],
        },
        "seed": settings["SEED"],
    }
    arrays["labels"] = np.concatenate(
        (np.zeros(len(data.background)), np.ones(len(data.signal)))
    )
    for name, result in results.items():
        metrics[name] = {"auc": result["auc"]}
        for key in ("background_scores", "signal_scores", "fpr", "tpr", "thresholds"):
            arrays[f"{name}_{key}"] = result[key]
    np.savez_compressed(outdir / "roc_comparison.npz", **arrays)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def main():
    with open("config.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    settings = config["GraphNet"]
    training = config["Training"]
    llp_settings = config["LLP"]
    validate_config(config)

    data, root_file, processor = load_evaluation_data(
        settings, training, llp_settings, config
    )
    try:
        results = evaluate_models(settings, training, data, processor)
        save_results(settings, training, llp_settings, data, results)
    finally:
        root_file.Close()
    output_dir = Path(settings["OUTPUT_DIR"]) / "model_comparison"
    print(f"Saved ROC plot, scores, and metrics to {output_dir}")


if __name__ == "__main__":
    main()
