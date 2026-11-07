import os
from pathlib import Path
import tempfile

import numpy as np
import ROOT
import yaml

from DataProcessor import DataProcessor


# Resolve input and output paths relative to this script so it can be run from
# any working directory.
ROOT_FILE = Path(__file__).with_name("Multijet_2010B.root")
CONFIG_FILE = Path(__file__).with_name("config.yaml")
root_file = ROOT.TFile.Open(str(ROOT_FILE))


def load_config():
	"""Read the input column names and derived-variable expressions."""
	with CONFIG_FILE.open() as config_file:
		config = yaml.safe_load(config_file)
	inputs = config["Inputs"]
	return inputs


def assert_array_equal(actual, expected):
	"""Compare NumPy arrays while showing useful details if they differ."""
	np.testing.assert_array_equal(actual, expected)


def test_array_helpers():
	"""Test padding and truncation for track and jet arrays without ROOT."""
	# __new__ lets us test these small conversion helpers without opening a
	# ROOT file or constructing the full DataProcessor object.
	processor = DataProcessor.__new__(DataProcessor)
	processor.max_tracks = 3

	# Two tracks are padded to three entries; a longer event is truncated.
	tracks = processor.tracks_to_array(
		[np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0, 6.0])]
	)
	assert tracks.shape == (3, 2)
	assert_array_equal(tracks, [[1.0, 3.0], [2.0, 4.0], [0.0, 5.0]])

	# Jet conversion follows the same pattern, with its own requested limit.
	jets = processor.jets_to_array([np.array([10.0]), np.array([20.0, 30.0])], max_jets=2)
	assert jets.shape == (2, 2)
	assert_array_equal(jets, [[10.0, 20.0], [0.0, 30.0]])


def test_pair_phi_encoding():
	"""Check pT ordering, wrapped delta-phi, and circular phi-sum encoding."""
	processor = DataProcessor.__new__(DataProcessor)
	processor.trk_columns = ["trk_pt", "trk_phi"]
	track_1 = np.array([[2.0, np.pi - 0.02]])
	track_2 = np.array([[3.0, -np.pi + 0.02]])

	pair = processor._build_pair_features(track_1, track_2)
	swapped_pair = processor._build_pair_features(track_2, track_1)
	assert pair.shape == (1, 5)
	assert_array_equal(pair, swapped_pair)
	np.testing.assert_allclose(pair[0, 0], 5.0, atol=1e-7)
	np.testing.assert_allclose(pair[0, 2], 1.0, atol=1e-7)
	np.testing.assert_allclose(pair[0, 3], 0.04, atol=1e-7)
	np.testing.assert_allclose(pair[0, 1], 1.0, atol=1e-7)
	np.testing.assert_allclose(pair[0, 4], 0.0, atol=1e-7)


def make_processor(inputs, root_file, max_events=2):
	"""Construct a short ROOT-backed DataProcessor for the integration tests."""
	assert root_file and not root_file.IsZombie(), f"Could not open {ROOT_FILE}"
	tree = root_file.Get("analyzer/Events")
	assert tree, "Could not find analyzer/Events in the ROOT file"
	# Printing the configuration makes failures caused by a changed YAML file
	# easier to diagnose.
	print("trk_columns:", inputs["trk_columns"])
	print("event_columns:", inputs["event_columns"])
	print("jet_columns:", inputs["jet_columns"])
	print("variables_to_define:", inputs["variables_to_define"])
	return DataProcessor(
		tree=tree,
		trk_columns=inputs["trk_columns"],
		event_columns=inputs["event_columns"],
		jets_columns=inputs["jet_columns"],
		variables_to_define=inputs["variables_to_define"],
		max_tracks=4,
		max_events=max_events,
	)


def test_data_processor(inputs, max_events, max_pairs, radius):
	"""Exercise every public DataProcessor method and validate its outputs."""
	processor = make_processor(inputs, root_file, max_events=max_events)

	# Read the configured ROOT branches and check the event, track, and jet
	# dimensions produced by get_npy_arrays().
	trk_array, event_array, jets_array = processor.get_npy_arrays()
	assert trk_array.shape == (max_events, 4, len(inputs["trk_columns"]))
	assert event_array.shape == (max_events, len(inputs["event_columns"]))
	assert jets_array.shape == (max_events, 6, len(inputs["jet_columns"]))

	# Phi adds one feature to the otherwise difference-plus-sum encoding.
	pairs, pair_events, pair_jets = processor.get_track_pairs(
		radius=radius,
		max_pairs=max_pairs,
	)
	n_pair_features = 2 * len(inputs["trk_columns"]) + (
		"trk_phi" in inputs["trk_columns"]
	)
	assert pairs.shape == (max_events, max_pairs, n_pair_features)
	# get_track_pairs returns the event and jet arrays alongside the pair array.
	assert_array_equal(pair_events, event_array)
	assert_array_equal(pair_jets, jets_array)
	assert_array_equal(processor.trk_pairs_array, pairs)

	# Verify that the split preserves event alignment and uses the requested
	# fraction for the validation subset.
	train_trk, train_event, val_trk, val_event = processor.get_split_dataset(0.5)

	assert train_trk.shape[0] == train_event.shape[0] == max_events // 2
	assert val_trk.shape[0] == val_event.shape[0] == max_events - (max_events // 2)

	# Verify that k-fold splitting returns all events exactly once and retains
	# the jet array in each fold.
	folds = processor.get_kfold_dataset(2)
	assert len(folds) == 2
	assert sum(fold[0].shape[0] for fold in folds) == max_events
	assert all(len(fold) == 3 for fold in folds)

	# The linear transform returns one midpoint and half-range per feature.
	trk_shift, trk_scale, event_shift, event_scale = processor.get_lin_transform()
	assert trk_shift.shape == trk_scale.shape == (len(inputs["trk_columns"]),)
	assert event_shift.shape == event_scale.shape == (len(inputs["event_columns"]),)

	# Save every computed array, load it into a new processor, and verify that
	# serialization preserves the values and the optional pair array.
	with tempfile.TemporaryDirectory() as temporary_directory:
		array_file = Path(temporary_directory) / "arrays.npz"
		processor.save_arrays(str(array_file))
		loaded = DataProcessor(
			tree=root_file.Get("analyzer/Events"),
			trk_columns=[],
			event_columns=[],
			filepath=str(array_file),
		)
		loaded_arrays = loaded.load_arrays(str(array_file))
		assert len(loaded_arrays) == 4
		assert_array_equal(loaded_arrays[0], trk_array)
		assert_array_equal(loaded_arrays[1], event_array)
		assert_array_equal(loaded_arrays[2], jets_array)
		assert_array_equal(loaded_arrays[3], pairs)
	return processor, pairs


def make_histograms(inputs):
	"""Create one ROOT histogram for every configured input column."""
	root_file = ROOT.TFile.Open(str(ROOT_FILE))
	assert root_file and not root_file.IsZombie(), f"Could not open {ROOT_FILE}"
	dataframe = ROOT.RDataFrame("analyzer/Events", str(ROOT_FILE))
	# Define derived columns before requesting histograms for them.
	for name, expression in inputs["variables_to_define"].items():
		dataframe = dataframe.Define(name, expression)

	# Combine track, event, and jet columns while preserving their configured
	# order; dict.fromkeys removes duplicate names without reordering them.
	columns = list(inputs["trk_columns"]
		+ inputs["event_columns"]
		+ inputs["jet_columns"]
	)
	histograms = {}
	for column in dict.fromkeys(columns):
		# Store lazy RDataFrame actions. They are evaluated together below.
		histograms[column] = dataframe.Histo1D(
			(f"h_{column}", column, 100, -10.0, 10.0), column
		)
	# Run all histogram actions in one event loop for better runtime efficiency.
	ROOT.RDF.RunGraphs(list(histograms.values()))
	assert all(histogram.GetValue().GetEntries() >= 0 for histogram in histograms.values())
	return histograms


def main():
    """Run the checks and save input and track-pair diagnostic plots."""
    inputs = load_config()
    test_array_helpers()
    test_pair_phi_encoding()
    processor, track_pairs_array = test_data_processor(
        inputs, 
        max_events=100, 
        max_pairs=10,
        radius = 0.1,
        )
    histograms = make_histograms(inputs)
    print(f"DataProcessor assertions passed; created {len(histograms)} histograms.")
    if not os.path.exists("plots/checking_DataProcessor"):
        os.makedirs("plots/checking_DataProcessor")
    canvas = ROOT.TCanvas("canvas", "canvas", 800, 600)
    for histogram in histograms.values():
        histogram.GetValue().Draw()
        canvas.SaveAs(f"plots/checking_DataProcessor/{histogram.GetName()}.png")

    pair_plot_directory = Path("plots/checking_DataProcessor/track_pairs")
    pair_plot_directory.mkdir(parents=True, exist_ok=True)
    pair_feature_names = [
        f"{name}_sum" for name in inputs["trk_columns"]
    ] + [
        f"{name}_diff" for name in inputs["trk_columns"]
    ]
    if "trk_phi" in inputs["trk_columns"]:
        pair_feature_names[
            inputs["trk_columns"].index("trk_phi")
        ] = "trk_phi_sum_cos"
        pair_feature_names[
            len(inputs["trk_columns"]) + inputs["trk_columns"].index("trk_phi")
        ] = "trk_phi_delta"
        pair_feature_names.append("trk_phi_sum_sin")
    pair_histograms = {}
    for feature_index, feature_name in enumerate(pair_feature_names):
        values = track_pairs_array[:, :, feature_index].ravel()
        finite_values = values[np.isfinite(values)]
        assert finite_values.size > 0
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
        if value_min == value_max:
            value_min -= 0.5
            value_max += 0.5
        histogram_name = f"h_track_pairs_{feature_index}_{feature_name}"
        histogram = ROOT.TH1D(
            histogram_name,
            f"Track-pair {feature_name};{feature_name};Entries",
            100,
            value_min,
            value_max,
        )
        histogram.FillN(
            int(finite_values.size),
            np.asarray(finite_values, dtype="double"),
            np.ones(finite_values.size, dtype="double"),
        )
        pair_histograms[histogram_name] = histogram
        histogram.Draw()
        canvas.SaveAs(str(pair_plot_directory / f"{histogram_name}.png"))

    assert len(pair_histograms) == track_pairs_array.shape[2]

    pair_counts = np.count_nonzero(
        np.any(track_pairs_array != 0, axis=2),
        axis=1,
    )
    assert pair_counts.shape == (track_pairs_array.shape[0],)
    assert np.all((pair_counts >= 0) & (pair_counts <= track_pairs_array.shape[1]))
    count_histogram = ROOT.TH1D(
        "h_track_pair_count_per_event",
        "Track-pair count per event;Number of track pairs;Events",
        track_pairs_array.shape[1] + 1,
        -0.5,
        track_pairs_array.shape[1] + 0.5,
    )
    count_histogram.FillN(
        int(pair_counts.size),
        np.asarray(pair_counts, dtype="double"),
        np.ones(pair_counts.size, dtype="double"),
    )
    count_histogram.Draw()
    canvas.SaveAs(str(pair_plot_directory / "h_track_pair_count_per_event.png"))

    print(
        f"Created {len(pair_histograms) + 1} track-pair histograms in "
        f"{pair_plot_directory}."
    )
        
if __name__ == "__main__":
	main()
