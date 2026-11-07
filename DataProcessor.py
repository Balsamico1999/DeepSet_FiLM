import numpy as np
import ROOT
from tqdm import tqdm

class DataProcessor:

    def __init__(
        self, 
        tree : ROOT.TTree, 
        trk_columns : list,
        event_columns : list,
        jets_columns : list = None,
        variables_to_define : dict = None,
        max_tracks : int = 64,
        max_events : int = None,
        filepath : str = None, # filepath to load data from
        ) -> None:

        self.trk_columns = trk_columns
        self.event_columns = event_columns
        self.jets_columns = jets_columns
        self.max_tracks = max_tracks
        self.filepath = filepath

        self.tree = tree
        print(f"Tree has {tree.GetEntries()} entries.")
        # Produce RDF to apply filters, define vars and get npy arrays
        self.rdf = ROOT.RDataFrame(self.tree)
        # Cap max events if asked
        if max_events is not None:
            self.rdf = self.rdf.Range(max_events)

        # Define variables if provided
        if variables_to_define is not None and self.filepath is None:
            for var in variables_to_define:
                expr = variables_to_define[var]
                self.rdf = self.rdf.Define(var,expr)
        
        # Store last computed arrays
        self.trk_array = None
        self.trk_pairs_array = None
        self.event_array = None
        self.jets_array = None
        self.jet_matched_trk_array = None
        self.jet_matched_trk_pairs_array = None
        self.jet_matched_track_counts = None

    # Function that reads the array for one track variable
    def tracks_to_array(self, array):
        output = None
        for event in tqdm(array):
            # cap maximum of tracks
            tracks = event[:self.max_tracks]
            # pad with zeros if we have less tracks
            if len(tracks) < self.max_tracks:
                tracks = np.pad(
                    tracks, 
                    (0, self.max_tracks - len(tracks)), 
                    constant_values=0 
                    )
            output = np.vstack([output, tracks]) if output is not None else tracks
        return output.T
    
    # Function that reads the array for one jet variable
    def jets_to_array(self, array, max_jets = 6):
        output = None
        for event in tqdm(array):
            jets = event[:max_jets] 
            # pad with zeros if we have less jets
            if len(jets) < max_jets:
                jets = np.pad(
                    jets, 
                    (0, max_jets - len(jets)), 
                    constant_values=0 
                    )
            output = np.vstack([output, jets]) if output is not None else jets
        return output.T


    def get_npy_arrays(self, cut = "1"):

        # If filepath indicated, simply read data from file
        if self.filepath is not None:
            return self.load_arrays(self.filepath)

        rdf_filtered = self.rdf.Filter(cut)
        trk_arrays = rdf_filtered.AsNumpy(self.trk_columns)
        event_arrays = rdf_filtered.AsNumpy(self.event_columns)
        jets_arrays = rdf_filtered.AsNumpy(self.jets_columns) if self.jets_columns is not None else None

        # Convert dict of arrays to stricly arrays
        print(f"Preparing {len(self.trk_columns)} track features:")
        tracks_var_arrays = [self.tracks_to_array(trk_arrays[column]) for column in self.trk_columns]
        trk_array_output = np.stack(tracks_var_arrays)

        # Event array are straight forwardly all the same shape
        print(f"Preparing the event features:")
        event_array_output = event_arrays[self.event_columns[0]]
        for column in self.event_columns[1:]:
            event_array_output = np.vstack([event_array_output,event_arrays[column]])

        # Jets array
        if self.jets_columns is not None:
            print(f"Preparing {len(self.jets_columns)} jet features:")
            jets_var_arrays = [self.jets_to_array(jets_arrays[column]) for column in self.jets_columns]
            jets_array_output = np.stack(jets_var_arrays).T
        else:
            jets_array_output = None
        
        # Store arrays as fields
        self.trk_array = trk_array_output.T
        self.event_array = event_array_output.T
        self.jets_array = jets_array_output
        
        return trk_array_output.T, event_array_output.T, jets_array_output

    def get_jet_matched_npy_arrays(self, cut="1", max_jets=6):
        """Return tracks grouped by their associated jet.

        The track result has shape ``(events, jets, tracks, features)``.
        Tracks with ``trk_jetIdx == -1`` and tracks assigned to jets beyond
        ``max_jets`` are omitted. ``trk_jetIdx`` is read internally and is
        not included among the returned track features.
        """
        if max_jets < 0:
            raise ValueError("max_jets must be non-negative")
        if not self.jets_columns:
            raise ValueError("jets_columns must be configured for jet matching")
        if self.filepath is not None:
            required = (
                "jet_matched_trk_array",
                "jet_matched_track_counts",
                "event_array",
                "jets_array",
            )
            with np.load(self.filepath) as data:
                missing = [name for name in required if name not in data]
                if missing:
                    raise ValueError(
                        "The configured array file does not contain jet-matched "
                        f"arrays; missing: {', '.join(missing)}"
                    )
                trk_array = data["jet_matched_trk_array"]
                event_array = data["event_array"]
                jets_array = data["jets_array"]
                track_counts = data["jet_matched_track_counts"]
            expected_shape = (
                event_array.shape[0],
                max_jets,
                self.max_tracks,
                len(self.trk_columns),
            )
            if trk_array.shape != expected_shape:
                raise ValueError(
                    "Cached jet-matched tracks have shape "
                    f"{trk_array.shape}, expected {expected_shape}"
                )
            if jets_array.shape != (
                event_array.shape[0],
                max_jets,
                len(self.jets_columns),
            ):
                raise ValueError("Cached jet features do not match the requested configuration")
            self.jet_matched_trk_array = trk_array
            self.jet_matched_track_counts = track_counts
            self.event_array = event_array
            self.jets_array = jets_array
            return trk_array, event_array, jets_array

        columns = list(dict.fromkeys(
            self.trk_columns
            + self.event_columns
            + self.jets_columns
            + ["trk_jetIdx"]
        ))
        rdf_filtered = self.rdf.Filter(cut)
        arrays = rdf_filtered.AsNumpy(columns)

        n_events = len(arrays[self.event_columns[0]])
        event_array = np.column_stack(
            [arrays[column] for column in self.event_columns]
        )
        jets_array = np.zeros(
            (n_events, max_jets, len(self.jets_columns)),
            dtype=float,
        )
        trk_array = np.zeros(
            (n_events, max_jets, self.max_tracks, len(self.trk_columns)),
            dtype=float,
        )
        track_counts = np.zeros((n_events, max_jets), dtype=np.int64)

        for event_index in range(n_events):
            jet_features = [
                np.asarray(arrays[column][event_index])
                for column in self.jets_columns
            ]
            n_event_jets = len(jet_features[0])
            if any(len(features) != n_event_jets for features in jet_features):
                raise ValueError("Configured jet feature branches have mismatched lengths")
            n_stored_jets = min(n_event_jets, max_jets)
            for feature_index, features in enumerate(jet_features):
                jets_array[event_index, :n_stored_jets, feature_index] = (
                    features[:n_stored_jets]
                )

            jet_indices = np.asarray(arrays["trk_jetIdx"][event_index])
            track_features = [
                np.asarray(arrays[column][event_index])
                for column in self.trk_columns
            ]
            if any(len(features) != len(jet_indices) for features in track_features):
                raise ValueError(
                    "Configured track feature branches and trk_jetIdx have "
                    "mismatched lengths"
                )

            for track_index, jet_index in enumerate(jet_indices):
                jet_index = int(jet_index)
                if jet_index < 0 or jet_index >= n_stored_jets:
                    continue
                track_slot = track_counts[event_index, jet_index]
                if track_slot >= self.max_tracks:
                    continue
                trk_array[event_index, jet_index, track_slot] = [
                    features[track_index] for features in track_features
                ]
                track_counts[event_index, jet_index] += 1

        self.jet_matched_trk_array = trk_array
        self.jet_matched_track_counts = track_counts
        self.event_array = event_array
        self.jets_array = jets_array
        return trk_array, event_array, jets_array

    def save_jet_matched_arrays(self, filepath):
        """Save arrays produced by ``get_jet_matched_npy_arrays`` for reuse."""
        if (
            self.jet_matched_trk_array is None
            or self.jet_matched_track_counts is None
            or self.event_array is None
            or self.jets_array is None
        ):
            raise ValueError(
                "Jet-matched arrays have not been computed; call "
                "get_jet_matched_npy_arrays() first"
            )
        np.savez_compressed(
            filepath,
            jet_matched_trk_array=self.jet_matched_trk_array,
            jet_matched_track_counts=self.jet_matched_track_counts,
            event_array=self.event_array,
            jets_array=self.jets_array,
        )

    def get_track_pairs(
        self,
        radius,
        max_pairs,
        cut="1",
        position_columns=("trk_x_pca", "trk_y_pca", "trk_z0"),
    ):
        """Build permutation-invariant feature vectors for nearby track pairs.

        A pair is selected when the Euclidean distance between its position
        coordinates is at most ``radius``. Each selected pair is represented
        by feature sums followed by signed
        differences. Pairs are ordered by descending track pT, with a
        deterministic feature-based tie-break. ``trk_phi`` uses a wrapped
        difference and circular sine/cosine sum encoding.
        """
        if radius < 0:
            raise ValueError("radius must be non-negative")
        if max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if len(position_columns) != 3:
            raise ValueError("position_columns must contain three columns")
        n_pair_features = 2 * len(self.trk_columns) + (
            "trk_phi" in self.trk_columns
        )

        trk_array, event_array, jets_array = self.get_npy_arrays(cut)
        position_indices = []
        for column in position_columns:
            if column not in self.trk_columns:
                raise ValueError(f"Track position column '{column}' is not configured")
            position_indices.append(self.trk_columns.index(column))

        n_events, _, n_features = trk_array.shape
        trk_pairs_array = np.zeros(
            (n_events, max_pairs, n_pair_features),
            dtype=trk_array.dtype,
        )

        for event_index, tracks in enumerate(trk_array):
            valid_tracks = np.any(tracks != 0, axis=1)
            valid_indices = np.flatnonzero(valid_tracks)
            if len(valid_indices) < 2 or max_pairs == 0:
                continue

            valid_tracks = tracks[valid_indices]
            positions = valid_tracks[:, position_indices]
            pair_indices = np.triu_indices(len(valid_tracks), k=1)
            pair_distances = np.linalg.norm(
                positions[pair_indices[0]] - positions[pair_indices[1]],
                axis=1,
            )
            selected = np.flatnonzero(pair_distances <= radius)
            if len(selected) == 0:
                continue

            pair_features = self._build_pair_features(
                valid_tracks[pair_indices[0][selected]],
                valid_tracks[pair_indices[1][selected]],
            )
            ordering = np.lexsort(pair_features.T[::-1], axis=0)
            ordering = ordering[np.argsort(pair_distances[selected][ordering], kind="stable")]
            pair_features = pair_features[ordering[:max_pairs]]
            trk_pairs_array[event_index, :len(pair_features)] = pair_features

        self.trk_pairs_array = trk_pairs_array
        return trk_pairs_array, event_array, jets_array

    def _build_pair_features(self, tracks_1, tracks_2):
        """Build pT-canonical pair features with circular encoding for phi."""
        if "trk_pt" not in self.trk_columns:
            raise ValueError("trk_pt must be configured to build canonical track pairs")

        pt_index = self.trk_columns.index("trk_pt")
        swap = tracks_1[:, pt_index] < tracks_2[:, pt_index]
        tied_pt = tracks_1[:, pt_index] == tracks_2[:, pt_index]
        undecided = tied_pt.copy()
        phi_index = (
            self.trk_columns.index("trk_phi")
            if "trk_phi" in self.trk_columns
            else None
        )
        coordinate_indices = {
            self.trk_columns.index(column)
            for column in ("trk_x_pca", "trk_y_pca")
            if column in self.trk_columns
        }
        for feature_index in range(tracks_1.shape[1]):
            if (
                feature_index == pt_index
                or feature_index == phi_index
                or (phi_index is not None and feature_index in coordinate_indices)
            ):
                continue
            feature_tie_swap = undecided & (
                tracks_1[:, feature_index] < tracks_2[:, feature_index]
            )
            swap |= feature_tie_swap
            undecided &= tracks_1[:, feature_index] == tracks_2[:, feature_index]

        if phi_index is not None:
            phi_difference = np.arctan2(
                np.sin(tracks_1[:, phi_index] - tracks_2[:, phi_index]),
                np.cos(tracks_1[:, phi_index] - tracks_2[:, phi_index]),
            )
            swap |= undecided & (phi_difference < 0)

        first = np.where(swap[:, np.newaxis], tracks_2, tracks_1)
        second = np.where(swap[:, np.newaxis], tracks_1, tracks_2)
        sums = first + second
        differences = first - second
        if "trk_phi" not in self.trk_columns:
            return np.concatenate((sums, differences), axis=1)

        phi_1 = first[:, phi_index]
        phi_2 = second[:, phi_index]
        wrapped_difference = np.arctan2(
            np.sin(phi_1 - phi_2),
            np.cos(phi_1 - phi_2),
        )
        differences[:, phi_index] = wrapped_difference
        phi_sum = phi_1 + phi_2
        sums[:, phi_index] = np.cos(phi_sum)
        return np.concatenate(
            (sums, differences, np.sin(phi_sum)[:, np.newaxis]),
            axis=1,
        )

    def get_jet_matched_track_pairs(
        self,
        radius,
        max_pairs,
        cut="1",
        max_jets=6,
        position_columns=("trk_x_pca", "trk_y_pca", "trk_z0"),
    ):
        """Build nearby track pairs independently within each jet.

        The pair result has shape ``(events, jets, pairs, pair_features)``.
        It contains feature sums followed by signed differences. Pairs are
        ordered by descending track pT, with a deterministic tie-break. When
        ``trk_phi`` is configured, its difference is wrapped to the circle,
        and its sum is encoded as cosine and sine components.
        """
        if radius < 0:
            raise ValueError("radius must be non-negative")
        if max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if len(position_columns) != 3:
            raise ValueError("position_columns must contain three columns")

        position_indices = []
        for column in position_columns:
            if column not in self.trk_columns:
                raise ValueError(f"Track position column '{column}' is not configured")
            position_indices.append(self.trk_columns.index(column))

        trk_array, event_array, jets_array = self.get_jet_matched_npy_arrays(
            cut=cut,
            max_jets=max_jets,
        )
        n_events, n_jets, _, n_features = trk_array.shape
        n_pair_features = 2 * n_features + ("trk_phi" in self.trk_columns)
        trk_pairs_array = np.zeros(
            (n_events, n_jets, max_pairs, n_pair_features),
            dtype=trk_array.dtype,
        )
        for event_index in range(n_events):
            for jet_index in range(n_jets):
                n_tracks = self.jet_matched_track_counts[event_index, jet_index]
                tracks = trk_array[event_index, jet_index, :n_tracks]
                if n_tracks < 2 or max_pairs == 0:
                    continue

                positions = tracks[:, position_indices]
                pair_indices = np.triu_indices(n_tracks, k=1)
                pair_distances = np.linalg.norm(
                    positions[pair_indices[0]] - positions[pair_indices[1]],
                    axis=1,
                )
                selected = np.flatnonzero(pair_distances <= radius)
                if len(selected) == 0:
                    continue
                selected = selected[
                    np.argsort(pair_distances[selected], kind="stable")
                ][:max_pairs]
                pair_features = self._build_pair_features(
                    tracks[pair_indices[0][selected]],
                    tracks[pair_indices[1][selected]],
                )
                trk_pairs_array[
                    event_index, jet_index, :len(pair_features)
                ] = pair_features

        self.jet_matched_trk_pairs_array = trk_pairs_array
        return trk_pairs_array, event_array, jets_array

    def get_split_dataset(self, val_fraction, cut = "1") -> np.array:
        trk_array, event_array, jets_array = self.get_npy_arrays(cut)
        nb_events = trk_array.shape[0]
        val_nb_events = int(np.round(val_fraction * nb_events, decimals = 0))

        val_trk_array = trk_array[:val_nb_events]
        train_trk_array = trk_array[val_nb_events:]

        val_event_array = event_array[:val_nb_events]
        train_event_array = event_array[val_nb_events:]

        return train_trk_array, train_event_array, val_trk_array, val_event_array

    def get_jet_matched_split_dataset(
        self,
        val_fraction,
        cut="1",
        max_jets=6,
    ):
        """Split jet-matched tracks, event features, and jets by event."""
        if not 0 <= val_fraction <= 1:
            raise ValueError("val_fraction must be between 0 and 1")
        trk_array, event_array, jets_array = self.get_jet_matched_npy_arrays(
            cut=cut,
            max_jets=max_jets,
        )
        n_val_events = int(np.round(val_fraction * trk_array.shape[0]))
        return (
            trk_array[n_val_events:],
            event_array[n_val_events:],
            jets_array[n_val_events:],
            trk_array[:n_val_events],
            event_array[:n_val_events],
            jets_array[:n_val_events],
        )

    def get_kfold_dataset(self, kfolds, cut = "1", max_pairs = 10, radius = 0.1) -> np.array:

        if max_pairs is not None:
            trk_array, event_array, jets_array, trk_pairs_array = self.get_track_pairs(
                max_pairs = max_pairs,
                radius = radius,
                cut = cut,
                )
        else:
            trk_array, event_array, jets_array = self.get_npy_arrays(cut)
            trk_pairs_array = None

        nb_events = trk_array.shape[0]

        # Get folds indices
        indices = np.arange(nb_events)
        folds_idx = indices % kfolds

        return [
            (
                trk_array[folds_idx == i],
                event_array[folds_idx == i],
                None if jets_array is None else jets_array[folds_idx == i],
                None if trk_pairs_array is None else trk_pairs_array[folds_idx == i]
                ) for i in range(kfolds)
            ]

    def get_jet_matched_kfold_dataset(
        self,
        kfolds,
        cut="1",
        max_jets=6,
        max_pairs=10,
        radius=0.1,
    ):
        """Return event-aligned jet-matched data for each deterministic fold."""
        if kfolds <= 0:
            raise ValueError("kfolds must be positive")

        if max_pairs is None:
            trk_array, event_array, jets_array = self.get_jet_matched_npy_arrays(
                cut=cut,
                max_jets=max_jets,
            )
            trk_pairs_array = None
        else:
            trk_pairs_array, event_array, jets_array = self.get_jet_matched_track_pairs(
                radius=radius,
                max_pairs=max_pairs,
                cut=cut,
                max_jets=max_jets,
            )
            trk_array = self.jet_matched_trk_array

        fold_indices = np.arange(trk_array.shape[0]) % kfolds
        return [
            (
                trk_array[fold_indices == fold],
                event_array[fold_indices == fold],
                jets_array[fold_indices == fold],
                None if trk_pairs_array is None else trk_pairs_array[fold_indices == fold],
            )
            for fold in range(kfolds)
        ]

    def save_arrays(self, filepath):
        print(
            self.trk_array.shape,
            self.event_array.shape,
            self.jets_array.shape,
            np.array(self.trk_array).shape,
              )

        np.savez_compressed(
            filepath,
            trk_array= self.trk_array,
            event_array= self.event_array,
            jets_array= self.jets_array if self.jets_array is not None else np.array([]),
            trk_pairs_array= self.trk_pairs_array if self.trk_pairs_array is not None else np.array([]),
        )
    
    def load_arrays(self, filepath):
        data = np.load(filepath)
        self.trk_array = data["trk_array"]
        self.event_array = data["event_array"]
        self.jets_array = data.get("jets_array", None)
        self.trk_pairs_array = data.get("trk_pairs_array", None)
        return self.trk_array, self.event_array, self.jets_array, self.trk_pairs_array
        
    # Transform the data to be in range [-1,1]
    def get_lin_transform(self):
        if self.trk_array is None or self.event_array is None:
            raise ValueError("Arrays not computed yet. Call get_npy_arrays() first.")
        
        trk_min = np.min(self.trk_array, axis=(0,1))
        trk_max = np.max(self.trk_array, axis=(0,1))

        event_min = np.min(self.event_array, axis=0)
        event_max = np.max(self.event_array, axis=0)

        trk_shift = (trk_max + trk_min) / 2
        trk_scale = (trk_max - trk_min) / 2
        event_shift = (event_max + event_min) / 2
        event_scale = (event_max - event_min) / 2
        
        return trk_shift, trk_scale, event_shift, event_scale

    # ------------------------------------------------------------------
    # GraphNet format: one graph per jet
    #   nodes   = the tracks of the jet
    #   edges   = ordered track pairs (sender -> receiver) with difference features
    #   globals = event features + jet features
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_phi(delta):
        """Wrap angle differences to (-pi, pi]."""
        return np.arctan2(np.sin(delta), np.cos(delta))
 
    def jet_matched_arrays_to_graphs(
        self,
        trk_array,
        track_counts,
        event_array,
        jets_array,
        min_tracks=1,
        k_neighbors=None,
        sort_by="trk_pt",
        edge_columns=None,
        phi_encoding="raw",
    ):
        """Convert jet-matched arrays into a flat, ragged graph representation.
 
        One graph is built per jet that has at least ``min_tracks`` tracks
        (padded jet slots with no tracks are dropped). Inputs are the outputs of
        ``get_jet_matched_npy_arrays`` plus ``jet_matched_track_counts``.
 
        Parameters
        ----------
        min_tracks : minimum number of tracks for a jet to become a graph.
        k_neighbors : ``None`` gives a fully connected graph (every ordered
            pair of distinct tracks is an edge). An integer ``k`` connects each
            track (receiver) to its ``k`` nearest tracks in (eta, phi) (senders);
            this needs ``trk_eta`` and ``trk_phi`` in ``trk_columns``.
        sort_by : track column used to order the nodes inside each graph
            (descending). Defaults to ``trk_pt``; ``None`` keeps the input order.
        edge_columns : track columns whose signed difference
            ``sender - receiver`` is stored as an edge feature (default: all
            ``trk_columns``). ``trk_phi`` differences are wrapped to (-pi, pi].
            When ``trk_eta`` and ``trk_phi`` exist, ``delta_R`` is appended.
        phi_encoding : ``"raw"`` keeps ``trk_phi`` as a node feature,
            ``"sincos"`` replaces it by ``trk_phi_cos`` and ``trk_phi_sin``.
 
        Returns
        -------
        dict of numpy arrays (all graph-level arrays are indexed by graph):
 
        ``node_features``   (total_nodes, n_node_features), nodes of graph g are
                            ``node_offsets[g] : node_offsets[g] + n_nodes[g]``
        ``edge_features``   (total_edges, n_edge_features)
        ``senders``, ``receivers``  (total_edges,) node indices LOCAL to their graph
        ``global_features`` (n_graphs, n_event_features + n_jet_features)
        ``n_nodes``, ``n_edges``, ``event_index``, ``jet_index``  (n_graphs,)
        ``n_events``        number of events the graphs were built from
        ``*_feature_names`` names of the feature columns
 
        where the offsets are ``np.cumsum`` of ``n_nodes`` / ``n_edges`` with a
        leading zero.
        """
        if min_tracks < 1:
            raise ValueError("min_tracks must be at least 1")
        if k_neighbors is not None and k_neighbors < 1:
            raise ValueError("k_neighbors must be a positive integer or None")
        if phi_encoding not in ("raw", "sincos"):
            raise ValueError("phi_encoding must be 'raw' or 'sincos'")
 
        trk_array = np.asarray(trk_array)
        track_counts = np.asarray(track_counts)
        event_array = np.asarray(event_array)
        jets_array = np.asarray(jets_array)
        if trk_array.ndim != 4 or trk_array.shape[-1] != len(self.trk_columns):
            raise ValueError(
                "trk_array must have shape (events, jets, tracks, len(trk_columns)); "
                f"got {trk_array.shape}"
            )
        n_events, n_jets = trk_array.shape[:2]
        if track_counts.shape != (n_events, n_jets):
            raise ValueError("track_counts must have shape (events, jets)")
        if event_array.shape[0] != n_events or jets_array.shape[:2] != (n_events, n_jets):
            raise ValueError("event_array / jets_array are not aligned with trk_array")
        if not self.jets_columns:
            raise ValueError("jets_columns must be configured to build global features")
 
        col = {name: i for i, name in enumerate(self.trk_columns)}
        if sort_by is not None and sort_by not in col:
            raise ValueError(
                f"sort_by column '{sort_by}' is not in trk_columns "
                "(pass sort_by=None to keep the input order)"
            )
        edge_columns = list(self.trk_columns if edge_columns is None else edge_columns)
        for name in edge_columns:
            if name not in col:
                raise ValueError(f"Edge column '{name}' is not in trk_columns")
        has_angles = "trk_eta" in col and "trk_phi" in col
        if k_neighbors is not None and not has_angles:
            raise ValueError("k_neighbors requires trk_eta and trk_phi in trk_columns")
        edge_idx_cols = [col[name] for name in edge_columns]
        edge_phi_pos = edge_columns.index("trk_phi") if "trk_phi" in edge_columns else None
 
        # Feature names
        node_names = list(self.trk_columns)
        if phi_encoding == "sincos" and "trk_phi" in col:
            i = col["trk_phi"]
            node_names[i:i + 1] = ["trk_phi_cos", "trk_phi_sin"]
        edge_names = [f"d_{name}" for name in edge_columns]
        if has_angles:
            edge_names.append("delta_R")
        global_names = list(self.event_columns) + list(self.jets_columns)
 
        # Which jets become graphs (event-major, then jet order)
        event_idx, jet_idx = np.nonzero(track_counts >= min_tracks)
        n_nodes = track_counts[event_idx, jet_idx].astype(np.int64)
        n_graphs = len(n_nodes)
        if n_graphs == 0:
            raise ValueError(f"No jet has at least {min_tracks} tracks")
        degree = (
            n_nodes - 1
            if k_neighbors is None
            else np.minimum(k_neighbors, n_nodes - 1)
        )
        n_edges = n_nodes * degree
        node_offsets = np.concatenate(([0], np.cumsum(n_nodes)))
        edge_offsets = np.concatenate(([0], np.cumsum(n_edges)))
 
        node_features = np.zeros((node_offsets[-1], len(node_names)), dtype=np.float32)
        edge_features = np.zeros((edge_offsets[-1], len(edge_names)), dtype=np.float32)
        senders = np.zeros(edge_offsets[-1], dtype=np.int32)
        receivers = np.zeros(edge_offsets[-1], dtype=np.int32)
 
        # Graphs with the same number of nodes share the same edge pattern, so
        # they are processed together (vectorised) in chunks.
        for n in np.unique(n_nodes):
            n = int(n)
            group = np.flatnonzero(n_nodes == n)
            d = int(degree[group[0]])
            chunk_size = max(1, int(2e7 // (n * n)))
            for start in range(0, len(group), chunk_size):
                ids = group[start:start + chunk_size]
                g = len(ids)
                tracks = trk_array[event_idx[ids], jet_idx[ids], :n].astype(np.float32)
                if sort_by is not None:
                    order = np.argsort(-tracks[:, :, col[sort_by]], axis=1, kind="stable")
                    tracks = np.take_along_axis(tracks, order[:, :, None], axis=1)
 
                # ---- nodes
                encoded = tracks
                if phi_encoding == "sincos" and "trk_phi" in col:
                    i = col["trk_phi"]
                    encoded = np.concatenate(
                        (
                            tracks[:, :, :i],
                            np.cos(tracks[:, :, i:i + 1]),
                            np.sin(tracks[:, :, i:i + 1]),
                            tracks[:, :, i + 1:],
                        ),
                        axis=2,
                    )
                node_pos = node_offsets[ids][:, None] + np.arange(n)
                node_features[node_pos.ravel()] = encoded.reshape(-1, encoded.shape[2])
 
                if d == 0:
                    continue
 
                # ---- edges: local (sender, receiver) indices, shape (g, n * d)
                if k_neighbors is None:
                    recv_local, send_local = np.nonzero(~np.eye(n, dtype=bool))
                    s_idx = np.broadcast_to(send_local, (g, len(send_local)))
                    r_idx = np.broadcast_to(recv_local, (g, len(recv_local)))
                else:
                    eta = tracks[:, :, col["trk_eta"]]
                    phi = tracks[:, :, col["trk_phi"]]
                    delta_r = np.hypot(
                        eta[:, :, None] - eta[:, None, :],
                        self._wrap_phi(phi[:, :, None] - phi[:, None, :]),
                    )  # [graph, receiver, sender]
                    delta_r[:, np.arange(n), np.arange(n)] = np.inf
                    nearest = np.argsort(delta_r, axis=2, kind="stable")[:, :, :d]
                    s_idx = nearest.reshape(g, n * d)
                    r_idx = np.broadcast_to(np.repeat(np.arange(n), d), (g, n * d))
 
                rows = np.arange(g)[:, None]
                tracks_s = tracks[rows, s_idx]  # (g, n_edges_per_graph, n_features)
                tracks_r = tracks[rows, r_idx]
                diff = tracks_s[:, :, edge_idx_cols] - tracks_r[:, :, edge_idx_cols]
                if edge_phi_pos is not None:
                    diff[:, :, edge_phi_pos] = self._wrap_phi(diff[:, :, edge_phi_pos])
                parts = [diff]
                if has_angles:
                    d_eta = tracks_s[:, :, col["trk_eta"]] - tracks_r[:, :, col["trk_eta"]]
                    d_phi = self._wrap_phi(
                        tracks_s[:, :, col["trk_phi"]] - tracks_r[:, :, col["trk_phi"]]
                    )
                    parts.append(np.hypot(d_eta, d_phi)[:, :, None])
                feats = np.concatenate(parts, axis=2)
 
                edge_pos = (edge_offsets[ids][:, None] + np.arange(s_idx.shape[1])).ravel()
                edge_features[edge_pos] = feats.reshape(-1, feats.shape[2])
                senders[edge_pos] = s_idx.ravel()
                receivers[edge_pos] = r_idx.ravel()
 
        global_features = np.concatenate(
            (
                event_array[event_idx].reshape(n_graphs, -1),
                jets_array[event_idx, jet_idx].reshape(n_graphs, -1),
            ),
            axis=1,
        ).astype(np.float32)
        if global_features.shape[1] != len(global_names):
            raise ValueError("event/jet array widths do not match event_columns + jets_columns")
 
        return {
            "node_features": node_features,
            "edge_features": edge_features,
            "senders": senders,
            "receivers": receivers,
            "global_features": global_features,
            "n_nodes": n_nodes.astype(np.int32),
            "n_edges": n_edges.astype(np.int32),
            "event_index": event_idx.astype(np.int32),
            "jet_index": jet_idx.astype(np.int32),
            "n_events": np.asarray(n_events),
            "node_feature_names": np.array(node_names),
            "edge_feature_names": np.array(edge_names),
            "global_feature_names": np.array(global_names),
        }
 
    def get_jet_graph_arrays(
        self,
        cut="1",
        max_jets=6,
        min_tracks=1,
        k_neighbors=None,
        sort_by="trk_pt",
        edge_columns=None,
        phi_encoding="raw",
    ):
        """Build one graph per jet straight from the ROOT tree (or cached arrays).
 
        Runs ``get_jet_matched_npy_arrays`` and then
        ``jet_matched_arrays_to_graphs``; see the latter for the format and
        the meaning of the options.
        """
        trk_array, event_array, jets_array = self.get_jet_matched_npy_arrays(
            cut=cut,
            max_jets=max_jets,
        )
        graphs = self.jet_matched_arrays_to_graphs(
            trk_array,
            self.jet_matched_track_counts,
            event_array,
            jets_array,
            min_tracks=min_tracks,
            k_neighbors=k_neighbors,
            sort_by=sort_by,
            edge_columns=edge_columns,
            phi_encoding=phi_encoding,
        )
        self.jet_graph_arrays = graphs
        return graphs
 
    def save_jet_graph_arrays(self, filepath):
        """Save the arrays produced by ``get_jet_graph_arrays``."""
        if self.jet_graph_arrays is None:
            raise ValueError("Graph arrays have not been computed; call get_jet_graph_arrays() first")
        np.savez_compressed(filepath, **self.jet_graph_arrays)
 
    @staticmethod
    def load_jet_graph_arrays(filepath):
        """Load a graph dict saved by ``save_jet_graph_arrays``."""
        with np.load(filepath) as data:
            return {key: data[key] for key in data.files}
 
    @staticmethod
    def select_jet_graphs(graphs, graph_mask):
        """Keep only the graphs where ``graph_mask`` (bool, per graph) is True."""
        graph_mask = np.asarray(graph_mask, dtype=bool)
        node_mask = np.repeat(graph_mask, graphs["n_nodes"])
        edge_mask = np.repeat(graph_mask, graphs["n_edges"])
        selected = dict(graphs)
        for key in ("global_features", "n_nodes", "n_edges", "event_index", "jet_index"):
            selected[key] = graphs[key][graph_mask]
        selected["node_features"] = graphs["node_features"][node_mask]
        for key in ("edge_features", "senders", "receivers"):
            selected[key] = graphs[key][edge_mask]
        return selected
 
    @staticmethod
    def split_jet_graphs(graphs, val_fraction):
        """Split by event (all jets of an event stay together): first events -> validation.
 
        Mirrors ``get_jet_matched_split_dataset``. Returns ``(train, val)`` graph dicts.
        """
        if not 0 <= val_fraction <= 1:
            raise ValueError("val_fraction must be between 0 and 1")
        n_val_events = int(np.round(val_fraction * int(graphs["n_events"])))
        is_val = graphs["event_index"] < n_val_events
        return (
            DataProcessor.select_jet_graphs(graphs, ~is_val),
            DataProcessor.select_jet_graphs(graphs, is_val),
        )
 
    @staticmethod
    def kfold_jet_graphs(graphs, kfolds):
        """Deterministic event-level folds (``event_index % kfolds``), like ``get_kfold_dataset``."""
        if kfolds <= 0:
            raise ValueError("kfolds must be positive")
        fold = graphs["event_index"] % kfolds
        return [DataProcessor.select_jet_graphs(graphs, fold == i) for i in range(kfolds)]
 
    def get_jet_graph_split_dataset(self, val_fraction, cut="1", max_jets=6, **graph_options):
        """``get_jet_graph_arrays`` followed by ``split_jet_graphs``; returns ``(train, val)``."""
        graphs = self.get_jet_graph_arrays(cut=cut, max_jets=max_jets, **graph_options)
        return self.split_jet_graphs(graphs, val_fraction)
 
    def get_jet_graph_kfold_dataset(self, kfolds, cut="1", max_jets=6, **graph_options):
        """``get_jet_graph_arrays`` followed by ``kfold_jet_graphs``."""
        graphs = self.get_jet_graph_arrays(cut=cut, max_jets=max_jets, **graph_options)
        return self.kfold_jet_graphs(graphs, kfolds)
 
    @staticmethod
    def get_jet_graph_lin_transform(graphs):
        """Shift/scale mapping each feature to [-1, 1] (same convention as ``get_lin_transform``).
 
        Statistics use real nodes / edges only (there is no padding in the graph
        format). Returns ``node_shift, node_scale, edge_shift, edge_scale,
        global_shift, global_scale``; normalise with ``(x - shift) / scale``.
        """
        def shift_scale(x):
            if len(x) == 0:
                return np.zeros(x.shape[1], dtype=np.float32), np.ones(x.shape[1], dtype=np.float32)
            low, high = x.min(axis=0), x.max(axis=0)
            scale = (high - low) / 2
            return (high + low) / 2, np.where(scale > 0, scale, 1.0)
 
        return (
            *shift_scale(graphs["node_features"]),
            *shift_scale(graphs["edge_features"]),
            *shift_scale(graphs["global_features"]),
        )


        
if __name__ == "__main__":

    variables_to_define = {
        "met_px" : "met_pt * cos(met_phi)",
        "met_py" : "met_pt * sin(met_phi)",
    }

    trk_columns = [
        "trk_d0",
        "trk_pt",
        "trk_eta",
        "trk_phi",
    ]

    event_columns = [
        "met_px",
        "met_py",
        "nTrack",
    ]

    root_file = ROOT.TFile.Open("Multijet_2010B.root")
    tree = root_file.Get("analyzer/Events")
    DP = DataProcessor(
        tree,
        trk_columns,
        event_columns,
        variables_to_define,
        max_events = 5000,
        )
    
    data = DP.get_split_dataset(0.2, )
    DP.get_lin_transform()