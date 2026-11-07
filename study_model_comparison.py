"""Compare ONNX anomaly scores on nominal and LLP-injected validation jets.

The evaluation uses the intersection of DeepSet's held-out fold and GraphNet's
held-out event fraction so both models are compared on the same events.
"""

from dataclasses import dataclass
from itertools import combinations
import json
from math import prod
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


@dataclass
class ModelOutputs:
    background_scores: np.ndarray
    signal_scores: np.ndarray
    background_representation: np.ndarray | None = None
    signal_representation: np.ndarray | None = None


class ModelAdapter(Protocol):
    label: str

    @classmethod
    def from_config(
        cls, settings: dict, training: dict, processor: DataProcessor
    ) -> "ModelAdapter":
        """Build this adapter from comparison settings and the data processor."""
        ...

    def evaluate(self, data: EvaluationData) -> ModelOutputs:
        """Return scores and optional multidimensional representations."""


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


def count_onnx_parameters(model_path):
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    initializers = model.graph.initializer
    parameter_initializers = [
        tensor
        for tensor in initializers
        if "ReadVariableOp" in tensor.name
        or tensor.name.endswith((".weight", ".bias", ".edge_init"))
    ]
    if not parameter_initializers:
        raise ValueError(f"No parameter tensors found in ONNX model: {model_path}")
    return sum(prod(tensor.dims) for tensor in parameter_initializers)


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
        self.parameter_count = count_onnx_parameters(model_path)
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

    def evaluate(self, data):
        reference_latent = self.encode(data.reference)
        if len(reference_latent) < 2:
            raise ValueError("At least two common training events are needed to calibrate DeepSet")
        mean = reference_latent.mean(axis=0)
        covariance_inverse = np.linalg.pinv(
            np.cov(reference_latent, rowvar=False), hermitian=True
        )
        background = self.encode(data.background)
        signal = self.encode(data.signal)
        return ModelOutputs(
            background_scores=self.mahalanobis_scores(
                background, mean, covariance_inverse
            ),
            signal_scores=self.mahalanobis_scores(signal, mean, covariance_inverse),
            background_representation=background,
            signal_representation=signal,
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
        self.parameter_count = count_onnx_parameters(model_path)
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
        representations = []
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
                reconstruction, latent = self.session.run(
                    ["reconstruction", "latent"], inputs
                )
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
                scores.append(graph_scores.numpy())
                representations.append(latent)
        return np.concatenate(scores), np.concatenate(representations, axis=0)

    def evaluate(self, data):
        background_scores, background_representation = self.score_samples(
            data.background
        )
        signal_scores, signal_representation = self.score_samples(data.signal)
        return ModelOutputs(
            background_scores=background_scores,
            signal_scores=signal_scores,
            background_representation=background_representation,
            signal_representation=signal_representation,
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
        print(f"{adapter.label}: {adapter.parameter_count:,} parameters")
        outputs = adapter.evaluate(data)
        if len(outputs.background_scores) != len(data.background):
            raise ValueError(f"{name} returned the wrong number of background scores")
        if len(outputs.signal_scores) != len(data.signal):
            raise ValueError(f"{name} returned the wrong number of signal scores")
        fpr, tpr, thresholds, model_auc = compute_roc(
            outputs.background_scores, outputs.signal_scores
        )
        for representation, expected_length, sample_name in (
            (outputs.background_representation, len(data.background), "background"),
            (outputs.signal_representation, len(data.signal), "signal"),
        ):
            if representation is not None and len(representation) != expected_length:
                raise ValueError(
                    f"{name} returned the wrong number of {sample_name} representations"
                )
        results[name] = {
            "label": adapter.label,
            "background_scores": outputs.background_scores,
            "signal_scores": outputs.signal_scores,
            "background_representation": outputs.background_representation,
            "signal_representation": outputs.signal_representation,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": float(model_auc),
        }
        print(f"{adapter.label}: ROC AUC = {model_auc:.4f}")
    return results


def pairwise_overlap_curves(results, n_events, n_points=100):
    """Compare top-score event sets at matched background threshold percentiles."""
    if n_events < 1:
        raise ValueError("Pairwise overlap needs at least one background event")
    threshold_percentiles = np.linspace(0.0, 0.99, n_points)
    ranked_events = {}
    for name, result in results.items():
        scores = np.asarray(result["background_scores"])
        if len(scores) != n_events:
            raise ValueError(
                f"{name} has {len(scores)} background scores for {n_events} events"
            )
        ranked_events[name] = np.argsort(scores, kind="stable")[::-1]

    curves = {}
    for left, right in combinations(results, 2):
        intersection_fraction = []
        union_fraction = []
        jaccard = []
        left_ranking = ranked_events[left]
        right_ranking = ranked_events[right]
        for percentile in threshold_percentiles:
            selected_fraction = 1.0 - percentile
            selected_count = min(
                n_events, max(1, int(np.ceil(selected_fraction * n_events)))
            )
            left_set = set(left_ranking[:selected_count])
            right_set = set(right_ranking[:selected_count])
            intersection_count = len(left_set & right_set)
            union_count = len(left_set | right_set)
            intersection_fraction.append(intersection_count / n_events)
            union_fraction.append(union_count / n_events)
            jaccard.append(intersection_count / union_count)
        curves[(left, right)] = {
            "intersection_fraction": np.asarray(intersection_fraction),
            "union_fraction": np.asarray(union_fraction),
            "jaccard": np.asarray(jaccard),
        }
    return threshold_percentiles, curves


def plot_pairwise_overlap(results, n_events, outdir):
    import matplotlib.pyplot as plt

    threshold_percentiles, curves = pairwise_overlap_curves(results, n_events)
    if not curves:
        print("Skipping pairwise overlap plot: at least two model adapters are required.")
        return {"threshold_percentile": threshold_percentiles, "curves": {}}

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    metrics = (
        ("intersection_fraction", "Common selected events / all events"),
        ("union_fraction", "Union of selected events / all events"),
        ("jaccard", "Jaccard overlap: intersection / union"),
    )
    for axis, (key, ylabel) in zip(axes, metrics):
        for (left, right), values in curves.items():
            label = f"{results[left]['label']} vs {results[right]['label']}"
            axis.plot(threshold_percentiles, values[key], label=label)
        axis.set_xlabel("Model-specific background threshold percentile (increasing)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[-1].legend()
    figure.suptitle("Pairwise anomaly-score event overlap")
    figure.tight_layout()
    figure.savefig(outdir / "pairwise_score_overlap.png", dpi=160)
    plt.close(figure)
    return {"threshold_percentile": threshold_percentiles, "curves": curves}


def plot_score_distributions(results, outdir):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(results), figsize=(6 * len(results), 5), squeeze=False
    )
    for axis, (_, result) in zip(axes[0], results.items()):
        background = np.asarray(result["background_scores"])
        signal = np.asarray(result["signal_scores"])
        bins = np.histogram_bin_edges(
            np.concatenate((background, signal)), bins="auto"
        )
        axis.hist(
            background,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="Nominal validation",
        )
        axis.hist(
            signal,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="LLP-injected",
        )
        axis.set_xlabel("Anomaly score")
        axis.set_ylabel("Normalized density")
        axis.set_title(f"{result['label']} (AUC = {result['auc']:.3f})")
        axis.grid(alpha=0.25)
        axis.legend()
        axis.set_xscale("log")
        axis.set_yscale("log")
    figure.suptitle("Anomaly-score distributions")
    figure.tight_layout()
    figure.savefig(outdir / "anomaly_score_distributions.png", dpi=160)
    plt.close(figure)


def plot_umap_representations(results, outdir, seed):
    import matplotlib.pyplot as plt

    multidimensional = {}
    for name, result in results.items():
        background = result["background_representation"]
        signal = result["signal_representation"]
        if background is None or signal is None:
            continue
        background = np.asarray(background)
        signal = np.asarray(signal)
        if background.ndim < 2 or signal.ndim < 2:
            continue
        background = background.reshape(len(background), -1)
        signal = signal.reshape(len(signal), -1)
        if background.shape[1] > 1 and signal.shape[1] == background.shape[1]:
            multidimensional[name] = (background, signal)

    if not multidimensional:
        print("Skipping UMAP plots: no adapter returned multidimensional representations.")
        return {}

    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "UMAP plots require umap-learn; install it with "
            "`python -m pip install umap-learn`."
        ) from error

    embeddings = {}
    for name, (background, signal) in multidimensional.items():
        representation = np.concatenate((background, signal), axis=0)
        if len(representation) < 3:
            raise ValueError(f"UMAP for {name} requires at least three samples")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(representation) - 1),
            min_dist=0.1,
            n_jobs=1,
            random_state=seed,
            transform_seed=seed,
        )
        coordinates = reducer.fit_transform(representation)
        embeddings[name] = coordinates

        figure, axis = plt.subplots(figsize=(7, 6))
        background_count = len(background)
        axis.scatter(
            coordinates[:background_count, 0],
            coordinates[:background_count, 1],
            s=10,
            alpha=0.55,
            label="Nominal validation",
        )
        axis.scatter(
            coordinates[background_count:, 0],
            coordinates[background_count:, 1],
            s=10,
            alpha=0.55,
            label="LLP-injected",
        )
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_title(f"{results[name]['label']} representation")
        axis.legend(markerscale=1.5)
        figure.tight_layout()
        figure.savefig(outdir / f"umap_{name}.png", dpi=160)
        plt.close(figure)
    return embeddings


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

    plot_score_distributions(results, outdir)
    pairwise_overlap = plot_pairwise_overlap(results, len(data.background), outdir)
    umap_embeddings = plot_umap_representations(
        results, outdir, settings["SEED"]
    )

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
    arrays.update(
        {
            f"pairwise_overlap_{left}__{right}_{metric}": values[metric]
            for (left, right), values in pairwise_overlap["curves"].items()
            for metric in ("intersection_fraction", "union_fraction", "jaccard")
        }
    )
    if len(pairwise_overlap["threshold_percentile"]):
        arrays["pairwise_overlap_threshold_percentile"] = pairwise_overlap[
            "threshold_percentile"
        ]
    for name, result in results.items():
        metrics[name] = {"auc": result["auc"]}
        for key in ("background_scores", "signal_scores", "fpr", "tpr", "thresholds"):
            arrays[f"{name}_{key}"] = result[key]
        for key in ("background_representation", "signal_representation"):
            representation = result[key]
            if representation is not None:
                arrays[f"{name}_{key}"] = representation
        if name in umap_embeddings:
            arrays[f"{name}_umap_coordinates"] = umap_embeddings[name]
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
