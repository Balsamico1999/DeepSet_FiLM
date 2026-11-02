import argparse
import inspect
import json
import math
import time
from pathlib import Path

from DataProcessor import DataProcessor
from GraphNet import (
    GraphAutoEncoder,
    GraphAutoEncoderONNX,
    GraphData,
    norm_to_tensors,
    norm_from_tensors,
    run_epoch,
    score_graphs,
    save_scores,
)

import numpy as np
import torch

import yaml

with open("config.yaml","r") as config_file:
    config = yaml.safe_load(config_file)

VARIABLES_TO_DEFINE = config["Inputs"]["variables_to_define"]
trk_columns = config["Inputs"]["trk_columns"]
event_columns = config["Inputs"]["event_columns"]
jet_columns = config["Inputs"]["jet_columns"]

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("data")
    src.add_argument("--root-file", help="ROOT file; graphs are built with DataProcessor")
    src.add_argument("--tree", default="analyzer/Events")
    src.add_argument("--graphs", help="cached graph .npz (from --save-graphs)")
    src.add_argument("--save-graphs", help="where to cache the graphs built from --root-file")
    src.add_argument("--cut", default="1")
    src.add_argument("--max-events", type=int, default=None)
    src.add_argument("--max-tracks", type=int, default=64)
    src.add_argument("--max-jets", type=int, default=6)
    src.add_argument("--min-tracks", type=int, default=2, help="drop jets with fewer tracks")
    src.add_argument("--k-neighbors", type=int, default=None,
                     help="connect each track to its k nearest (eta, phi) tracks; default: fully connected")
    src.add_argument("--phi-encoding", choices=["raw", "sincos"], default="sincos")
    src.add_argument("--val-fraction", type=float, default=0.2)

    mod = p.add_argument_group("model")
    mod.add_argument("--hidden", type=int, default=64)
    mod.add_argument("--latent-dim", type=int, default=8)
    mod.add_argument("--encoder-blocks", type=int, default=3)
    mod.add_argument("--decoder-blocks", type=int, default=2)
    mod.add_argument("--mlp-layers", type=int, default=2)

    opt = p.add_argument_group("training")
    opt.add_argument("--epochs", type=int, default=50)
    opt.add_argument("--batch-size", type=int, default=256)
    opt.add_argument("--lr", type=float, default=1e-3)
    opt.add_argument("--weight-decay", type=float, default=0.0)
    opt.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    opt.add_argument("--seed", type=int, default=0)
    opt.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    opt.add_argument("--outdir", default="graphnet_run")
    opt.add_argument("--score-graphs", nargs="*", default=[],
                     help="other graph .npz files to score with the trained model")
    args = p.parse_args()
    if (args.root_file is None) == (args.graphs is None):
        p.error("give exactly one of --root-file or --graphs")
    if not 0 < args.val_fraction < 1:
        p.error("--val-fraction must be in (0, 1)")
    return args


def load_graphs(args):
    if args.graphs is not None:
        return DataProcessor.load_jet_graph_arrays(args.graphs)

    import ROOT  # only needed when building from a ROOT file

    root_file = ROOT.TFile.Open(args.root_file)
    tree = root_file.Get(args.tree)
    processor = DataProcessor(
        tree,
        trk_columns,
        event_columns,
        jets_columns=jet_columns,
        variables_to_define=VARIABLES_TO_DEFINE,
        max_tracks=args.max_tracks,
        max_events=args.max_events,
    )
    graphs = processor.get_jet_graph_arrays(
        cut=args.cut,
        max_jets=args.max_jets,
        min_tracks=args.min_tracks,
        k_neighbors=args.k_neighbors,
        phi_encoding=args.phi_encoding,
        sort_by="trk_px",
    )
    if args.save_graphs:
        processor.save_jet_graph_arrays(args.save_graphs)
    return graphs


def save_onnx_model(model, batch, path, cfg, norm, feature_names):
    import onnx
    import onnxruntime as ort

    input_names = (
        "node", "edge", "senders", "receivers", "node_graph", "edge_graph",
        "node_rank", "glob", "n_nodes", "dec_senders", "dec_receivers", "dec_edge_graph",
    )
    tensors = tuple(getattr(batch, name) for name in input_names)
    dynamic_axes = {
        "node": {0: "nodes"},
        "edge": {0: "edges"},
        "senders": {0: "edges"},
        "receivers": {0: "edges"},
        "node_graph": {0: "nodes"},
        "edge_graph": {0: "edges"},
        "node_rank": {0: "nodes"},
        "glob": {0: "graphs"},
        "n_nodes": {0: "graphs"},
        "dec_senders": {0: "decoder_edges"},
        "dec_receivers": {0: "decoder_edges"},
        "dec_edge_graph": {0: "decoder_edges"},
        "reconstruction": {0: "nodes"},
        "latent": {0: "graphs"},
    }
    adapter = GraphAutoEncoderONNX(model).eval()
    export_options = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    torch.onnx.export(
        adapter,
        tensors,
        str(path),
        input_names=input_names,
        output_names=("reconstruction", "latent"),
        dynamic_axes=dynamic_axes,
        opset_version=18,
        **export_options,
    )

    onnx_model = onnx.load(str(path))

    def clear_incorrect_branch_shapes(graph):
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    branch = attribute.g
                    if node.op_type == "If":
                        for output in branch.output:
                            tensor_type = output.type.tensor_type
                            if (
                                "Identity_" in output.name
                                and tensor_type.HasField("shape")
                                and len(tensor_type.shape.dim) == 1
                            ):
                                # The legacy exporter mislabels reduction outputs as rank 1.
                                tensor_type.ClearField("shape")
                    clear_incorrect_branch_shapes(branch)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        clear_incorrect_branch_shapes(subgraph)

    clear_incorrect_branch_shapes(onnx_model.graph)
    metadata = {
        "format_version": "1",
        "config": json.dumps(cfg, sort_keys=True),
        "normalization": json.dumps(
            {name: value.tolist() for name, value in norm.items()}, sort_keys=True
        ),
        "feature_names": json.dumps(feature_names, sort_keys=True),
        "input_contract": (
            "Graph tensors are batched and normalized with the saved normalization "
            "statistics; outputs are normalized track reconstruction and graph latent."
        ),
    }
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(path))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    runtime_inputs = {
        name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, tensors)
    }
    with torch.no_grad():
        expected = adapter(*tensors)
    actual = session.run(["reconstruction", "latent"], runtime_inputs)
    for expected_output, actual_output in zip(expected, actual):
        np.testing.assert_allclose(
            actual_output, expected_output.detach().cpu().numpy(), rtol=1e-3, atol=1e-4
        )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    graphs = load_graphs(args)
    names = {k: graphs[k].tolist() for k in ("node_feature_names", "edge_feature_names", "global_feature_names")}
    print(f"{len(graphs['n_nodes'])} graphs from {int(graphs['n_events'])} events, "
          f"{len(graphs['node_features'])} nodes, {len(graphs['edge_features'])} edges")
    print("node features  :", names["node_feature_names"])
    print("edge features  :", names["edge_feature_names"])
    print("global features:", names["global_feature_names"])

    train_graphs, val_graphs = DataProcessor.split_jet_graphs(graphs, args.val_fraction)
    if len(val_graphs["n_nodes"]) == 0 or len(train_graphs["n_nodes"]) == 0:
        raise SystemExit("Train or validation split is empty; use more events or change --val-fraction")
    # Normalisation statistics come from the training split only.
    norm = DataProcessor.get_jet_graph_lin_transform(train_graphs)
    train_data, val_data = GraphData(train_graphs, norm), GraphData(val_graphs, norm)

    cfg = dict(
        n_node=train_data.node.shape[1],
        n_edge=train_data.edge.shape[1],
        n_glob=train_data.glob.shape[1],
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        encoder_blocks=args.encoder_blocks,
        decoder_blocks=args.decoder_blocks,
        mlp_layers=args.mlp_layers,
        max_nodes=max(args.max_tracks, int(graphs["n_nodes"].max())),
    )
    model = GraphAutoEncoder(cfg).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters; training on {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(2, args.patience // 3)
    )
    checkpoint_path = outdir / "graphnet_ae.pt"
    history, best_val, best_epoch = [], float("inf"), -1

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_data, args.batch_size, device, optimizer)
        val_loss = run_epoch(model, val_data, args.batch_size, device)
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise SystemExit(f"Non-finite loss at epoch {epoch}; try a lower --lr or check the inputs")
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "lr": optimizer.param_groups[0]["lr"]})
        improved = val_loss < best_val
        if improved:
            best_val, best_epoch = val_loss, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "norm": norm_to_tensors(norm),
                    "feature_names": names,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
        print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  "
              f"lr {optimizer.param_groups[0]['lr']:.1e}  {time.time() - start:.1f}s"
              f"{'  *' if improved else ''}")
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping (no improvement for {args.patience} epochs)")
            break

    (outdir / "history.json").write_text(json.dumps(history, indent=2))

    # Reload the best model and write the anomaly scores
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Best epoch {checkpoint['epoch']} (val loss {checkpoint['val_loss']:.5f})")

    val_scores = score_graphs(model, val_data, device)
    save_scores(outdir / "scores_validation.npz", val_data, val_scores)
    q = np.quantile(val_scores, [0.5, 0.9, 0.99])
    print(f"Validation jet score  median {q[0]:.4f}  90% {q[1]:.4f}  99% {q[2]:.4f}")

    for path in args.score_graphs:
        other = DataProcessor.load_jet_graph_arrays(path)
        for key, expected in names.items():
            if other[key].tolist() != expected:
                raise SystemExit(f"{path}: {key} differ from the training graphs; "
                                 "rebuild them with the same columns and options")
        other_data = GraphData(other, norm_from_tensors(checkpoint["norm"]))
        scores = score_graphs(model, other_data, device)
        out = outdir / f"scores_{Path(path).stem}.npz"
        save_scores(out, other_data, scores)
        print(f"{path}: {len(scores)} jets, median score {np.median(scores):.4f} -> {out}")

    export_count = min(max(args.batch_size, 1), len(train_data))
    export_batch = train_data.batch(torch.arange(export_count))
    onnx_path = outdir / "graphnet_ae.onnx"
    save_onnx_model(model.cpu(), export_batch, onnx_path, cfg, checkpoint["norm"], names)
    print(f"Saved and verified ONNX model: {onnx_path}")


if __name__ == "__main__":
    main()