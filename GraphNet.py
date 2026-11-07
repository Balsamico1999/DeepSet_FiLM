"""Graph-Network autoencoder for jet track content (anomaly detection).

Each jet is one graph:
    nodes   = tracks of the jet            (trk_columns)
    edges   = ordered track pairs          (differences of the track features, delta_R)
    globals = event + jet features         (event_columns + jets_columns)

Model (pure PyTorch, no torch_geometric needed)
    encoder : node/edge/global MLP embeddings -> L Graph-Network blocks
              (Battaglia et al. 2018: edge update, node update, global update)
              -> graph-level latent vector z (the bottleneck)
    decoder : [z, global features, n_tracks] -> one "slot" per track (learned
              rank embedding; nodes are ordered by descending pT) -> L Graph-Network
              blocks on a fully connected slot graph -> reconstructed track features.
              The decoder never sees the input tracks or edges, only z, the globals
              (conditioning, like the FiLM inputs of the DeepSet model) and n_tracks.
    loss    : MSE on the normalised track features, averaged over the tracks of
              a jet, then over jets. The same per-jet quantity is the anomaly score.

Usage
    # from a ROOT file (builds the graphs with DataProcessor)
    python train_graphnet.py --root-file Multijet_2010B.root --save-graphs graphs.npz

    # from cached graphs
    python train_graphnet.py --graphs graphs.npz --score-graphs other_sample.npz
"""

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import nn


AGGREGATIONS = ("mean", "max")  # permutation-invariant pooling used in the GN blocks

# ----------------------------------------------------------------------------
# Batching: flat ragged graph arrays -> batched tensors (fully vectorised)
# ----------------------------------------------------------------------------
@dataclass
class GraphBatch:
    node: torch.Tensor          # (N, F_node)   normalised node features (also the target)
    edge: torch.Tensor          # (E, F_edge)   normalised edge features
    senders: torch.Tensor       # (E,)          indices into the batch's node axis
    receivers: torch.Tensor     # (E,)
    node_graph: torch.Tensor    # (N,)          graph id of each node
    edge_graph: torch.Tensor    # (E,)
    node_rank: torch.Tensor     # (N,)          position of the node in its graph (pT rank)
    glob: torch.Tensor          # (B, F_glob)   normalised event + jet features
    n_nodes: torch.Tensor       # (B,)
    dec_senders: torch.Tensor   # fully connected graph used by the decoder
    dec_receivers: torch.Tensor
    dec_edge_graph: torch.Tensor
    num_graphs: int

    def to(self, device):
        values = {}
        for f in fields(self):
            v = getattr(self, f.name)
            values[f.name] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        return GraphBatch(**values)


def ragged_index(starts, counts):
    """Flat indices of the concatenated ranges [start, start + count)."""
    total = int(counts.sum())
    exclusive = torch.cumsum(counts, 0) - counts
    within = torch.arange(total) - torch.repeat_interleave(exclusive, counts)
    return torch.repeat_interleave(starts, counts) + within


def complete_graph(n_nodes, node_offsets):
    """All ordered pairs of distinct nodes inside each graph of the batch."""
    n2 = n_nodes * n_nodes
    graph = torch.repeat_interleave(torch.arange(len(n_nodes)), n2)
    pair = torch.arange(int(n2.sum())) - (torch.cumsum(n2, 0) - n2)[graph]
    n = n_nodes[graph]
    receiver = torch.div(pair, n, rounding_mode="floor")
    sender = pair % n
    keep = sender != receiver
    graph, sender, receiver = graph[keep], sender[keep], receiver[keep]
    offset = node_offsets[graph]
    return sender + offset, receiver + offset, graph


class GraphData:
    """Normalised graph arrays as CPU tensors; ``batch(ids)`` gathers a batch."""

    def __init__(self, graphs, norm):
        node_shift, node_scale, edge_shift, edge_scale, glob_shift, glob_scale = norm

        def tensor(a, dtype):
            return torch.from_numpy(np.ascontiguousarray(a)).to(dtype)

        self.node = tensor((graphs["node_features"] - node_shift) / node_scale, torch.float32)
        self.edge = tensor((graphs["edge_features"] - edge_shift) / edge_scale, torch.float32)
        self.glob = tensor((graphs["global_features"] - glob_shift) / glob_scale, torch.float32)
        self.senders = tensor(graphs["senders"], torch.long)
        self.receivers = tensor(graphs["receivers"], torch.long)
        self.n_nodes = tensor(graphs["n_nodes"], torch.long)
        self.n_edges = tensor(graphs["n_edges"], torch.long)
        self.node_offsets = torch.cumsum(self.n_nodes, 0) - self.n_nodes
        self.edge_offsets = torch.cumsum(self.n_edges, 0) - self.n_edges
        self.event_index = graphs["event_index"]
        self.jet_index = graphs["jet_index"]
        self.n_events = int(graphs["n_events"])

    def __len__(self):
        return len(self.n_nodes)

    def batch(self, ids):
        n_nodes, n_edges = self.n_nodes[ids], self.n_edges[ids]
        num_graphs = len(ids)
        node_idx = ragged_index(self.node_offsets[ids], n_nodes)
        edge_idx = ragged_index(self.edge_offsets[ids], n_edges)
        batch_offsets = torch.cumsum(n_nodes, 0) - n_nodes
        graph_ids = torch.arange(num_graphs)
        node_graph = torch.repeat_interleave(graph_ids, n_nodes)
        edge_graph = torch.repeat_interleave(graph_ids, n_edges)
        node_rank = torch.arange(len(node_graph)) - batch_offsets[node_graph]
        dec_s, dec_r, dec_g = complete_graph(n_nodes, batch_offsets)
        return GraphBatch(
            node=self.node[node_idx],
            edge=self.edge[edge_idx],
            senders=self.senders[edge_idx] + batch_offsets[edge_graph],
            receivers=self.receivers[edge_idx] + batch_offsets[edge_graph],
            node_graph=node_graph,
            edge_graph=edge_graph,
            node_rank=node_rank,
            glob=self.glob[ids],
            n_nodes=n_nodes,
            dec_senders=dec_s,
            dec_receivers=dec_r,
            dec_edge_graph=dec_g,
            num_graphs=num_graphs,
        )


def iterate(data, batch_size, shuffle, device):
    order = torch.randperm(len(data)) if shuffle else torch.arange(len(data))
    for start in range(0, len(data), batch_size):
        yield data.batch(order[start:start + batch_size]).to(device)


# ----------------------------------------------------------------------------
# Graph-Network building blocks
# ----------------------------------------------------------------------------
def make_mlp(in_dim, hidden, out_dim, n_layers=2, layer_norm=True):
    """MLP with ``n_layers`` linear layers; optional LayerNorm on the output."""
    layers, dim = [], in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(dim, hidden), nn.SiLU()]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


def aggregate(x, index, num_segments, how=AGGREGATIONS):
    """Pool rows of ``x`` into ``num_segments`` groups; empty groups give zeros."""
    pooled = []
    counts = None
    for kind in how:
        if kind in ("sum", "mean"):
            out = x.new_zeros(num_segments, x.shape[1]).scatter_reduce(
                0,
                index.unsqueeze(1).expand_as(x),
                x,
                reduce="sum",
                include_self=True,
            )
            if kind == "mean":
                counts = x.new_zeros(num_segments).scatter_reduce(
                    0,
                    index,
                    x.new_ones(index.shape[0]),
                    reduce="sum",
                    include_self=True,
                )
                out = out / counts.clamp(min=1).unsqueeze(1)
        elif kind == "max":
            if counts is None:
                counts = x.new_zeros(num_segments).scatter_reduce(
                    0,
                    index,
                    x.new_ones(index.shape[0]),
                    reduce="sum",
                    include_self=True,
                )
            out = x.new_full(
                (num_segments, x.shape[1]), torch.finfo(x.dtype).min
            ).scatter_reduce(
                0, index.unsqueeze(1).expand_as(x), x, reduce="amax", include_self=True
            )
            out = torch.where(counts.unsqueeze(1) > 0, out, torch.zeros_like(out))
        else:
            raise ValueError(f"unknown aggregation '{kind}'")
        pooled.append(out)
    return torch.cat(pooled, dim=1)


class GNBlock(nn.Module):
    """Full Graph-Network block with residual connections.

        e_k' = phi_e([e_k, v_sender, v_receiver, u])
        v_i' = phi_v([pool_{k->i} e_k', v_i, u])
        u'   = phi_u([pool e', pool v', u])
    """

    def __init__(self, dim, mlp_layers):
        super().__init__()
        n_agg = len(AGGREGATIONS)
        self.edge_mlp = make_mlp(4 * dim, dim, dim, mlp_layers)
        self.node_mlp = make_mlp((n_agg + 2) * dim, dim, dim, mlp_layers)
        self.global_mlp = make_mlp((2 * n_agg + 1) * dim, dim, dim, mlp_layers)

    def forward(self, v, e, u, senders, receivers, node_graph, edge_graph):
        num_nodes, num_graphs = v.shape[0], u.shape[0]
        e = e + self.edge_mlp(torch.cat([e, v[senders], v[receivers], u[edge_graph]], dim=1))
        incoming = aggregate(e, receivers, num_nodes)
        v = v + self.node_mlp(torch.cat([incoming, v, u[node_graph]], dim=1))
        u = u + self.global_mlp(
            torch.cat(
                [aggregate(e, edge_graph, num_graphs), aggregate(v, node_graph, num_graphs), u],
                dim=1,
            )
        )
        return v, e, u


class GraphEncoder(nn.Module):
    def __init__(self, n_node, n_edge, n_glob, hidden, latent_dim, n_blocks, mlp_layers):
        super().__init__()
        self.node_embed = make_mlp(n_node, hidden, hidden, mlp_layers)
        self.edge_embed = make_mlp(n_edge, hidden, hidden, mlp_layers)
        self.glob_embed = make_mlp(n_glob, hidden, hidden, mlp_layers)
        self.blocks = nn.ModuleList([GNBlock(hidden, mlp_layers) for _ in range(n_blocks)])
        self.to_latent = make_mlp(hidden, hidden, latent_dim, mlp_layers, layer_norm=False)

    def forward(self, b):
        v, e, u = self.node_embed(b.node), self.edge_embed(b.edge), self.glob_embed(b.glob)
        for block in self.blocks:
            v, e, u = block(v, e, u, b.senders, b.receivers, b.node_graph, b.edge_graph)
        return self.to_latent(u)


class GraphDecoder(nn.Module):
    def __init__(self, n_node, n_glob, hidden, latent_dim, n_blocks, mlp_layers, max_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        self.global_init = make_mlp(latent_dim + n_glob + 1, hidden, hidden, mlp_layers)
        self.rank_embed = nn.Embedding(max_nodes, hidden)
        self.node_init = make_mlp(2 * hidden, hidden, hidden, mlp_layers)
        self.edge_init = nn.Parameter(torch.zeros(hidden))
        self.blocks = nn.ModuleList([GNBlock(hidden, mlp_layers) for _ in range(n_blocks)])
        self.node_out = make_mlp(hidden, hidden, n_node, mlp_layers, layer_norm=False)

    def forward(self, z, b):
        n_frac = (b.n_nodes.to(z.dtype) / self.max_nodes).unsqueeze(1)
        u = self.global_init(torch.cat([z, b.glob, n_frac], dim=1))
        rank = b.node_rank.clamp(max=self.max_nodes - 1)
        v = self.node_init(torch.cat([self.rank_embed(rank), u[b.node_graph]], dim=1))
        e = self.edge_init.expand(b.dec_senders.shape[0], -1)
        for block in self.blocks:
            v, e, u = block(v, e, u, b.dec_senders, b.dec_receivers, b.node_graph, b.dec_edge_graph)
        return self.node_out(v)


class GraphAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = GraphEncoder(
            cfg["n_node"], cfg["n_edge"], cfg["n_glob"], cfg["hidden"],
            cfg["latent_dim"], cfg["encoder_blocks"], cfg["mlp_layers"],
        )
        self.decoder = GraphDecoder(
            cfg["n_node"], cfg["n_glob"], cfg["hidden"], cfg["latent_dim"],
            cfg["decoder_blocks"], cfg["mlp_layers"], cfg["max_nodes"],
        )

    def forward(self, b):
        z = self.encoder(b)
        return self.decoder(z, b), z


class GraphAutoEncoderONNX(nn.Module):
    """Tensor-only adapter exposing the graph autoencoder to ONNX exporters."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        node,
        edge,
        senders,
        receivers,
        node_graph,
        edge_graph,
        node_rank,
        glob,
        n_nodes,
        dec_senders,
        dec_receivers,
        dec_edge_graph,
    ):
        batch = GraphBatch(
            node=node,
            edge=edge,
            senders=senders,
            receivers=receivers,
            node_graph=node_graph,
            edge_graph=edge_graph,
            node_rank=node_rank,
            glob=glob,
            n_nodes=n_nodes,
            dec_senders=dec_senders,
            dec_receivers=dec_receivers,
            dec_edge_graph=dec_edge_graph,
            num_graphs=glob.shape[0],
        )
        return self.model(batch)


def per_graph_error(recon, b):
    """Mean squared reconstruction error over the tracks of each jet -> (B,)."""
    se = (recon - b.node).pow(2).mean(dim=1)
    total = se.new_zeros(b.num_graphs).index_add_(0, b.node_graph, se)
    return total / b.n_nodes.clamp(min=1).to(se.dtype)


# ----------------------------------------------------------------------------
# Training / scoring
# ----------------------------------------------------------------------------
def run_epoch(model, data, batch_size, device, optimizer=None, clip=1.0):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in iterate(data, batch_size, shuffle=training, device=device):
            recon, _ = model(batch)
            error = per_graph_error(recon, batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                error.mean().backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total += error.sum().item()
            count += batch.num_graphs
    return total / max(count, 1)


@torch.no_grad()
def score_graphs(model, data, device, batch_size=1024):
    """Per-jet anomaly score (mean squared reconstruction error) for a GraphData set."""
    model.eval()
    scores = torch.empty(len(data))
    for start in range(0, len(data), batch_size):
        ids = torch.arange(start, min(start + batch_size, len(data)))
        batch = data.batch(ids).to(device)
        recon, _ = model(batch)
        scores[ids] = per_graph_error(recon, batch).cpu()
    return scores.numpy()


def event_scores(data, jet_scores, reduce="max"):
    """Aggregate jet scores to one score per event (NaN for events without graphs)."""
    out = np.full(data.n_events, -np.inf if reduce == "max" else 0.0)
    if reduce == "max":
        np.maximum.at(out, data.event_index, jet_scores)
        out[np.isinf(out)] = np.nan
    else:
        np.add.at(out, data.event_index, jet_scores)
        out[np.bincount(data.event_index, minlength=data.n_events) == 0] = np.nan
    return out


def save_scores(path, data, jet_scores):
    np.savez_compressed(
        path,
        jet_score=jet_scores,
        event_index=data.event_index,
        jet_index=data.jet_index,
        n_tracks=data.n_nodes.numpy(),
        event_score_max=event_scores(data, jet_scores, "max"),
        event_score_sum=event_scores(data, jet_scores, "sum"),
    )


def norm_to_tensors(norm):
    return {
        name: torch.from_numpy(np.asarray(a, dtype=np.float32))
        for name, a in zip(
            ("node_shift", "node_scale", "edge_shift", "edge_scale", "glob_shift", "glob_scale"),
            norm,
        )
    }


def norm_from_tensors(saved):
    keys = ("node_shift", "node_scale", "edge_shift", "edge_scale", "glob_shift", "glob_scale")
    return tuple(saved[k].numpy() for k in keys)