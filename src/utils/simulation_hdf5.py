"""Store modal-analysis simulation results in an HDF5 file, one simulation at a time,
with the input variables described by an ``uncertain_variables.VariableSet``.

Layout:

    nodes/coordinates                                   (n_nodes, 3)               float64
    nodes/node_id                                       (n_nodes,)                 int32
    simulations/simulation_id                           (n_sims,)                  int32
    simulations/status                                  (n_sims,)                  int8
    simulations/error_message                           (n_sims,)                  string
    simulations/input_variables/distribution_parameters (n_vars,)                  compound
    simulations/input_variables/samples                 (n_sims, n_vars)           float64
    simulations/mode/frequencies                        (n_sims, n_modes)          float64
    simulations/mode/mode_shapes/{ux,uy,uz}             (n_sims, n_modes, n_nodes) float64
        (only the components chosen at creation, listed in the group's
         "components" attribute, e.g. ["ux", "uy"] for an in-plane model)

``distribution_parameters`` has one record per variable, in VariableSet order:

    name               variable name
    distribution       Q.get_dist_types() entry, e.g. "unif", "norm", "translated"
    base_distribution  for "translated": type of the wrapped distribution, else ""
    parameters         Q.get_dist_params() entry as a float array; for
                       "translated": base parameters followed by shift, scale, center

Workflow:

    create_simulation_file(path, Q, samples, n_modes, coords, node_ids=...)
    for i in pending_simulations(path):
        try:
            ...run the solve...
            write_simulation_result(path, i, freqs, shapes, node_ids=...)
        except Exception as exc:
            mark_failed(path, i, str(exc))

    samples, qoi, Q = read_simulation_file(path)

    # Or import existing results in the read_simulation_file format
    dataframes_to_hdf5(path, Q, "x_df.csv", "y_df.csv")

The ``nodes`` group's ``coordinates_missing`` attribute is True when the file
was created without node coordinates (they are then stored as NaN).

Results not yet written are NaN. ``status`` is 0 = pending, 1 = done, -1 = failed.
Each write opens and closes the file, so a crash loses at most the running solve.
Only one process should write to the file at a time.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from contextlib import nullcontext

import h5py
import numpy as np
import pandas as pd

from uncertain_variables import (
    BetaDistribution,
    ExponentialDistribution,
    LogNormalDistribution,
    NormalDistribution,
    SemiCircleDistribution,
    TranslatedDistribution,
    UniformDistribution,
    Variable,
    VariableSet,
)

ALL_COMPONENTS = ("ux", "uy", "uz")
STR_DTYPE = h5py.string_dtype(encoding="utf-8")
VLEN_FLOAT = h5py.vlen_dtype(np.float64)

PENDING, DONE, FAILED = 0, 1, -1

# get_dist_type() string -> class whose constructor takes get_dist_params() in order.
# Extend this (or pass ``registry=`` to the reader) when you add distribution classes.
DISTRIBUTION_CLASSES = {
    "norm": NormalDistribution,
    "lognorm": LogNormalDistribution,
    "unif": UniformDistribution,
    "beta": BetaDistribution,
    "exp": ExponentialDistribution,
    "wigner": SemiCircleDistribution,
}

RECORD_DTYPE = np.dtype([
    ("name", STR_DTYPE),
    ("distribution", STR_DTYPE),
    ("base_distribution", STR_DTYPE),
    ("parameters", VLEN_FLOAT),
])


# ---------------------------------------------------------------------------
# VariableSet <-> compound records
# ---------------------------------------------------------------------------
def _as_float_array(params) -> np.ndarray:
    """get_dist_params() returns a scalar, list or tuple depending on the class."""
    return np.atleast_1d(np.asarray(params, dtype=np.float64))


def _variable_set_to_records(Q: VariableSet) -> np.ndarray:
    variables = Q.get_variables()
    names = list(variables)
    dists = list(variables.values())
    if not names:
        raise ValueError("The VariableSet has no variables")
    for name, d in zip(names, dists):
        if d is None:
            raise ValueError(
                f"Variable {name!r} has no distribution (fixed value?); only "
                "variables with a distribution can be stored"
            )

    types = Q.get_dist_types()
    try:
        params = Q.get_dist_params()
    except TypeError:
        # TranslatedDistribution.get_dist_params() fails when the wrapped
        # distribution returns a list (uniform) or a scalar (exp, wigner);
        # those entries are rebuilt per distribution below.
        params = [None] * len(dists)

    records = np.empty(len(names), dtype=RECORD_DTYPE)
    for i, (name, dist, kind, p) in enumerate(zip(names, dists, types, params)):
        base_kind = ""
        if kind == "translated":
            inner = dist.dist
            base_kind = inner.get_dist_type()
            if base_kind == "translated":
                raise NotImplementedError(
                    f"Variable {name!r}: nested translated distributions are not supported"
                )
            # Base type is needed to rebuild it, so params are assembled here
            p = np.concatenate([
                _as_float_array(inner.get_dist_params()),
                [dist.shift, dist.scale, dist.center],
            ])
        elif p is None:
            p = dist.get_dist_params()

        check = base_kind or kind
        if check not in DISTRIBUTION_CLASSES:
            raise ValueError(
                f"Variable {name!r}: distribution type {check!r} has no entry in "
                "DISTRIBUTION_CLASSES, so it could not be read back"
            )

        records[i]["name"] = name
        records[i]["distribution"] = kind
        records[i]["base_distribution"] = base_kind
        records[i]["parameters"] = _as_float_array(p)
    return records


def _decode(v) -> str:
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def _records_to_variable_set(records: np.ndarray, normalized: bool,
                             registry: Mapping | None = None) -> VariableSet:
    classes = {**DISTRIBUTION_CLASSES, **(registry or {})}
    Q = VariableSet(normalized=normalized)
    for rec in records:
        name = _decode(rec["name"])
        kind = _decode(rec["distribution"])
        base_kind = _decode(rec["base_distribution"])
        p = [float(x) for x in rec["parameters"]]

        if kind == "translated":
            if base_kind not in classes:
                raise KeyError(f"Variable {name!r}: unknown base distribution {base_kind!r}")
            base = classes[base_kind](*p[:-3])
            shift, scale, center = p[-3:]
            dist = TranslatedDistribution(base, shift, scale, center)
        else:
            if kind not in classes:
                raise KeyError(f"Variable {name!r}: unknown distribution type {kind!r}")
            dist = classes[kind](*p)

        Q.add(Variable(name, dist))
    return Q


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_components(components) -> tuple[str, ...]:
    """Turn "xy", ["x", "y"], ("ux", "uz"), ... into a tuple of dataset names
    in canonical order, e.g. ("ux", "uy")."""
    if isinstance(components, str):
        components = [components] if components in ALL_COMPONENTS else list(components)
    out = set()
    for c in components:
        name = c if str(c).startswith("u") else f"u{c}"
        name = name.lower()
        if name not in ALL_COMPONENTS:
            raise ValueError(f"Unknown component {c!r}; use any of x, y, z")
        if name in out:
            raise ValueError(f"Component {c!r} given twice")
        out.add(name)
    if not out:
        raise ValueError("At least one component is required")
    return tuple(c for c in ALL_COMPONENTS if c in out)


def _file_components(f: h5py.File) -> tuple[str, ...]:
    ms = f["simulations/mode/mode_shapes"]
    if "components" in ms.attrs:
        return tuple(str(c) for c in ms.attrs["components"])
    return tuple(c for c in ALL_COMPONENTS if c in ms)  # files written before this attribute


def _split_components(mode_shapes, ndim: int, components) -> dict[str, np.ndarray]:
    """Accept a dict {component: array of ``ndim`` dims} with exactly
    ``components`` as keys (``"x"`` or ``"ux"`` style), or one array with an
    extra trailing axis holding the components in x, y, z order. With a single
    component, an array of ``ndim`` dims without the trailing axis is accepted
    too. Returns {dataset name: array}."""
    if isinstance(mode_shapes, Mapping):
        given = {normalize_components([k])[0]: v for k, v in mode_shapes.items()}
        if set(given) != set(components):
            raise KeyError(
                f"mode_shapes has components {sorted(given)}, the file stores {list(components)}"
            )
        out = {c: np.asarray(given[c], dtype=np.float64) for c in components}
        for c, a in out.items():
            if a.ndim != ndim:
                raise ValueError(f"mode shape {c} must have {ndim} dims, got shape {a.shape}")
        return out

    arr = np.asarray(mode_shapes, dtype=np.float64)
    if len(components) == 1 and arr.ndim == ndim:
        return {components[0]: arr}
    if arr.ndim != ndim + 1 or arr.shape[-1] != len(components):
        raise ValueError(
            f"mode_shapes array must have {ndim + 1} dims with a last axis of "
            f"{len(components)} ({', '.join(components)}), got {arr.shape}"
        )
    return {c: arr[..., j] for j, c in enumerate(components)}


def _node_permutation(stored_ids: np.ndarray, given_ids) -> np.ndarray | None:
    """Index array that reorders data given in ``given_ids`` order into the
    file's ``stored_ids`` order. None if the orders already match."""
    given_ids = np.asarray(given_ids)
    if given_ids.shape != stored_ids.shape:
        raise ValueError(f"Got {given_ids.size} node IDs, file has {stored_ids.size} nodes")
    if np.array_equal(given_ids, stored_ids):
        return None
    sorter = np.argsort(given_ids, kind="stable")
    pos = np.clip(np.searchsorted(given_ids, stored_ids, sorter=sorter), 0, given_ids.size - 1)
    perm = sorter[pos]
    if not np.array_equal(given_ids[perm], stored_ids):
        raise ValueError("Node IDs of this result do not match the node IDs stored in the file")
    return perm


def _sims_per_chunk(n_sims: int, n_modes: int, n_nodes: int, target_bytes: int = 1 << 20) -> int:
    """Simulations per mode-shape chunk: one for large meshes, several for
    small ones so chunks stay around 1 MiB instead of a few hundred bytes."""
    per_sim = n_modes * n_nodes * 8
    return int(max(1, min(n_sims, target_bytes // per_sim)))


def _open_for_reading(source):
    """Context manager yielding an h5py.File for reading. ``source`` is a path,
    or an already-open h5py.File, which is used as is and left open."""
    if isinstance(source, h5py.File):
        if not source.id.valid:
            raise ValueError("The h5py.File passed in is closed")
        return nullcontext(source)
    return h5py.File(source, "r")


def _check_index(f: h5py.File, index: int) -> int:
    n_sims = f["simulations/status"].shape[0]
    index = int(index)
    if not 0 <= index < n_sims:
        raise IndexError(f"Simulation index {index} out of range 0..{n_sims - 1}")
    return index


def _samples_array(samples, Q: VariableSet) -> np.ndarray:
    """Samples as (n_sims, n_vars) in VariableSet order. A DataFrame is
    reordered by column name; a plain array must already be in Q's order."""
    names = Q.variable_names()
    if isinstance(samples, pd.DataFrame):
        missing = [n for n in names if n not in samples.columns]
        if missing:
            raise KeyError(f"samples DataFrame is missing columns {missing}")
        samples = samples[names].to_numpy()
    arr = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    if arr.shape[1] != len(names):
        raise ValueError(f"samples has {arr.shape[1]} columns, VariableSet has {len(names)} variables")
    return arr


# ---------------------------------------------------------------------------
# 1. Create the file before any solve
# ---------------------------------------------------------------------------
def create_simulation_file(
    path,
    Q: VariableSet,
    samples,
    n_modes: int,
    node_coordinates,
    node_ids=None,
    simulation_ids=None,
    components=ALL_COMPONENTS,
    metadata: Mapping | None = None,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
    overwrite: bool = False,
) -> None:
    """Write everything known up front and pre-allocate the result datasets.

    Parameters
    ----------
    Q : VariableSet describing the input variables
    samples : array (n_sims, n_vars) in Q's variable order, e.g. from
        ``Q.sample(n)``, or a DataFrame with Q's variable names as columns
    n_modes : number of modes requested per solve
    node_coordinates : array, shape (n_nodes, 3), or None if unknown. With
        None, ``node_ids`` is required, the coordinates are stored as NaN and
        the ``nodes`` group gets the attribute ``coordinates_missing = True``
        (False otherwise).
    node_ids : array, shape (n_nodes,), optional — defaults to 0..n_nodes-1.
        Results passed to ``write_simulation_result`` are matched against these.
    simulation_ids : array, shape (n_sims,), optional — defaults to 0..n_sims-1
    components : displacement directions to store, e.g. "xy", ("x", "y") or
        ("ux", "uy"). Defaults to all three. Only these datasets are created,
        and every result written to the file must provide exactly these.
    metadata : optional mapping written as root attributes
        (e.g. {"solver": "MAPDL 2025 R2", "mode_normalization": "mass"})
    compression, compression_opts : passed to h5py for the mode-shape datasets
    overwrite : replace an existing file. False by default so a finished or
        half-finished run is never wiped by accident.
    """
    components = normalize_components(components)
    samples = _samples_array(samples, Q)
    coordinates_missing = node_coordinates is None
    if coordinates_missing:
        if node_ids is None:
            raise ValueError("Pass node_ids when node_coordinates is None")
        node_coordinates = np.full((np.size(node_ids), 3), np.nan)
    node_coordinates = np.asarray(node_coordinates, dtype=np.float64)
    n_sims, n_vars = samples.shape
    n_nodes = node_coordinates.shape[0]
    n_modes = int(n_modes)

    if node_coordinates.shape != (n_nodes, 3):
        raise ValueError(f"node_coordinates must be (n_nodes, 3), got {node_coordinates.shape}")
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1")

    node_ids = (np.arange(n_nodes) if node_ids is None else np.asarray(node_ids)).astype(np.int32)
    simulation_ids = (
        np.arange(n_sims) if simulation_ids is None else np.asarray(simulation_ids)
    ).astype(np.int32)
    if node_ids.shape != (n_nodes,):
        raise ValueError(f"node_ids must have shape ({n_nodes},), got {node_ids.shape}")
    if np.unique(node_ids).size != n_nodes:
        raise ValueError("node_ids contains duplicates")
    if simulation_ids.shape != (n_sims,):
        raise ValueError(f"simulation_ids must have shape ({n_sims},), got {simulation_ids.shape}")

    records = _variable_set_to_records(Q)

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")

    with h5py.File(path, "w") as f:
        f.create_dataset("nodes/coordinates", data=node_coordinates)
        f.create_dataset("nodes/node_id", data=node_ids)
        f["nodes"].attrs["coordinates_missing"] = coordinates_missing

        sim = f.create_group("simulations")
        sim.create_dataset("simulation_id", data=simulation_ids)
        sim.create_dataset("status", shape=(n_sims,), dtype=np.int8, fillvalue=PENDING)
        sim.create_dataset("error_message", shape=(n_sims,), dtype=STR_DTYPE)

        iv = sim.create_group("input_variables")
        iv.attrs["normalized"] = bool(Q.normalized)
        iv.create_dataset("distribution_parameters", data=records)
        iv.create_dataset("samples", data=samples)

        mode = sim.create_group("mode")
        mode.create_dataset("frequencies", shape=(n_sims, n_modes),
                            dtype=np.float64, fillvalue=np.nan)
        ms = mode.create_group("mode_shapes")
        ms.attrs["components"] = list(components)
        for c in components:
            ms.create_dataset(
                c,
                shape=(n_sims, n_modes, n_nodes),
                dtype=np.float64,
                fillvalue=np.nan,
                chunks=(_sims_per_chunk(n_sims, n_modes, n_nodes), n_modes, n_nodes),
                compression=compression,
                compression_opts=compression_opts if compression == "gzip" else None,
            )

        f.attrs["n_simulations"] = n_sims
        f.attrs["n_modes"] = n_modes
        f.attrs["n_nodes"] = n_nodes
        f.attrs["n_input_variables"] = n_vars
        for k, v in (metadata or {}).items():
            f.attrs[k] = v


# ---------------------------------------------------------------------------
# 2. Write one simulation's results
# ---------------------------------------------------------------------------
def write_simulation_result(path, index: int, frequencies, mode_shapes, node_ids=None) -> None:
    """Store the results of simulation ``index`` and mark it done.

    Parameters
    ----------
    index : row of ``samples`` this result belongs to
    frequencies : array, shape (k,), with k <= n_modes. If the solver found
        fewer modes than requested, the missing ones stay NaN.
    mode_shapes : array of shape (k, n_nodes, C), where C is the number of
        components stored in the file (in x, y, z order), or a dict
        {component: (k, n_nodes)} with exactly the file's components as keys
        ("x" or "ux" style). With a single component, (k, n_nodes) also works.
    node_ids : node IDs in the order the mode shapes are given. If provided,
        the data is reordered to the file's node order (and an error is raised
        if the node sets differ). If omitted, the data must already be in the
        file's node order.

    Writing the same index again overwrites its earlier result.
    """
    freqs = np.asarray(frequencies, dtype=np.float64).ravel()

    with h5py.File(path, "r+") as f:
        index = _check_index(f, index)
        shapes = _split_components(mode_shapes, ndim=2, components=_file_components(f))
        n_modes = f["simulations/mode/frequencies"].shape[1]
        n_nodes = f["nodes/node_id"].shape[0]
        k = freqs.size

        if k > n_modes:
            raise ValueError(f"Got {k} modes, file was created for {n_modes}")
        for c, a in shapes.items():
            if a.shape != (k, n_nodes):
                raise ValueError(f"mode shape {c} must be ({k}, {n_nodes}), got {a.shape}")

        if node_ids is not None:
            perm = _node_permutation(f["nodes/node_id"][:], node_ids)
            if perm is not None:
                shapes = {c: a[:, perm] for c, a in shapes.items()}

        freq_row = np.full(n_modes, np.nan)
        freq_row[:k] = freqs
        f["simulations/mode/frequencies"][index] = freq_row
        for c, a in shapes.items():
            block = np.full((n_modes, n_nodes), np.nan)
            block[:k] = a
            f[f"simulations/mode/mode_shapes/{c}"][index] = block

        # Status last: if anything above fails, the simulation stays pending.
        f["simulations/error_message"][index] = ""
        f["simulations/status"][index] = DONE


# ---------------------------------------------------------------------------
# 3. Mark a failed solve
# ---------------------------------------------------------------------------
def mark_failed(path, index: int, message: str = "") -> None:
    """Mark simulation ``index`` as failed and store an optional error message.
    Any earlier results for that index are cleared to NaN."""
    with h5py.File(path, "r+") as f:
        index = _check_index(f, index)
        f["simulations/mode/frequencies"][index] = np.nan
        for c in _file_components(f):
            f[f"simulations/mode/mode_shapes/{c}"][index] = np.nan
        f["simulations/error_message"][index] = str(message)
        f["simulations/status"][index] = FAILED


# ---------------------------------------------------------------------------
# Checking that a file meets the minimal requirements
# ---------------------------------------------------------------------------
class SimulationFileError(ValueError):
    """Raised when a file lacks something the reader needs. ``problems`` lists
    every issue found, not only the first."""

    def __init__(self, problems: list[str], source=None):
        self.problems = list(problems)
        where = f" ({source})" if source else ""
        super().__init__(
            f"Not a valid simulation file{where}:\n  - " + "\n  - ".join(self.problems)
        )


def _get_dataset(f, name: str, problems: list, required: bool = True):
    obj = f.get(name)
    if obj is None:
        if required:
            problems.append(f"missing dataset '{name}'")
        return None
    if not isinstance(obj, h5py.Dataset):
        problems.append(f"'{name}' is a group, expected a dataset")
        return None
    return obj


def _is_numeric(ds) -> bool:
    return ds.dtype.kind in "fiu"


def validate_simulation_file(
    path,
    require_samples: bool = True,
    require_frequencies: bool = True,
    require_mode_shapes: bool = True,
    registry: Mapping | None = None,
) -> dict:
    """Check that a file has what the readers need, and raise one
    ``SimulationFileError`` listing every problem if not.

    Always required
        simulations/input_variables/distribution_parameters
            compound dataset with at least one record and the fields
            ``name``, ``distribution`` and ``parameters`` (plus
            ``base_distribution`` if any distribution is "translated");
            names unique and non-empty, every type known.
    Required unless switched off
        simulations/input_variables/samples   (n_sims, n_vars), numeric
        simulations/mode/frequencies          (n_sims, n_modes), numeric
        nodes/node_id                         (n_nodes,), unique integers
        simulations/mode/mode_shapes/<c>      (n_sims, n_modes, n_nodes), numeric,
            for at least one of ux, uy, uz (the components the group's
            "components" attribute lists, if it has one)
    Optional (defaults used when absent)
        simulations/status          (n_sims,) in {-1, 0, 1}; absent -> all done
        simulations/simulation_id   (n_sims,) unique integers; absent -> 0..n_sims-1
        nodes/coordinates           (n_nodes, 3)
        input_variables "normalized" attribute; absent -> True

    ``path`` can be a file path or an open ``h5py.File`` (left open).

    Returns a summary dict: n_sims, n_vars, n_modes, n_nodes, components,
    has_status, has_simulation_id (unchecked parts are None).
    """
    with _open_for_reading(path) as f:
        problems: list[str] = []
        info = dict(n_sims=None, n_vars=None, n_modes=None, n_nodes=None,
                    components=None, has_status=False, has_simulation_id=False)

        # --- variables -------------------------------------------------------
        dp = _get_dataset(f, "simulations/input_variables/distribution_parameters", problems)
        if dp is not None:
            fields = dp.dtype.names or ()
            missing = [k for k in ("name", "distribution", "parameters") if k not in fields]
            if not fields:
                problems.append("distribution_parameters is not a compound dataset")
            elif missing:
                problems.append(f"distribution_parameters lacks the fields {missing}")
            elif dp.ndim != 1 or dp.shape[0] == 0:
                problems.append(f"distribution_parameters must be 1-D and non-empty, got shape {dp.shape}")
            else:
                recs = dp[:]
                names = [_decode(n) for n in recs["name"]]
                kinds = [_decode(k) for k in recs["distribution"]]
                info["n_vars"] = len(names)
                if any(not n for n in names):
                    problems.append("some variables have an empty name")
                dupes = sorted({n for n in names if names.count(n) > 1})
                if dupes:
                    problems.append(f"duplicate variable names {dupes}")
                known = {**DISTRIBUTION_CLASSES, **(registry or {})}
                if "translated" in kinds and "base_distribution" not in fields:
                    problems.append("translated distributions present but no base_distribution field")
                for n, k, i in zip(names, kinds, range(len(names))):
                    check = _decode(recs["base_distribution"][i]) if k == "translated" and "base_distribution" in fields else k
                    if k == "translated" and "base_distribution" not in fields:
                        continue
                    if check not in known:
                        problems.append(f"variable {n!r}: unknown distribution type {check!r} "
                                        "(add it to DISTRIBUTION_CLASSES or pass registry=)")

        # --- samples -----------------------------------------------------------
        if require_samples:
            ds = _get_dataset(f, "simulations/input_variables/samples", problems)
            if ds is not None:
                if ds.ndim != 2 or not _is_numeric(ds):
                    problems.append(f"samples must be a 2-D numeric array, got shape {ds.shape}, dtype {ds.dtype}")
                else:
                    info["n_sims"] = ds.shape[0]
                    if info["n_vars"] is not None and ds.shape[1] != info["n_vars"]:
                        problems.append(f"samples has {ds.shape[1]} columns but "
                                        f"{info['n_vars']} variables are defined")

        # --- frequencies -------------------------------------------------------
        if require_frequencies:
            ds = _get_dataset(f, "simulations/mode/frequencies", problems)
            if ds is not None:
                if ds.ndim != 2 or not _is_numeric(ds):
                    problems.append(f"frequencies must be a 2-D numeric array, got shape {ds.shape}, dtype {ds.dtype}")
                else:
                    info["n_modes"] = ds.shape[1]
                    if info["n_sims"] is None:
                        info["n_sims"] = ds.shape[0]
                    elif ds.shape[0] != info["n_sims"]:
                        problems.append(f"frequencies has {ds.shape[0]} rows, samples has {info['n_sims']}")

        n_sims = info["n_sims"]

        # --- optional per-simulation datasets ---------------------------------
        ds = _get_dataset(f, "simulations/status", problems, required=False)
        if ds is not None:
            info["has_status"] = True
            if ds.ndim != 1 or ds.dtype.kind not in "iu":
                problems.append(f"status must be a 1-D integer array, got shape {ds.shape}, dtype {ds.dtype}")
            else:
                if n_sims is not None and ds.shape[0] != n_sims:
                    problems.append(f"status has {ds.shape[0]} entries, expected {n_sims}")
                bad = np.setdiff1d(np.unique(ds[:]), [FAILED, PENDING, DONE])
                if bad.size:
                    problems.append(f"status contains unknown values {bad.tolist()} (allowed: -1, 0, 1)")

        ds = _get_dataset(f, "simulations/simulation_id", problems, required=False)
        if ds is not None:
            info["has_simulation_id"] = True
            if ds.ndim != 1 or ds.dtype.kind not in "iu":
                problems.append(f"simulation_id must be a 1-D integer array, got shape {ds.shape}, dtype {ds.dtype}")
            else:
                if n_sims is not None and ds.shape[0] != n_sims:
                    problems.append(f"simulation_id has {ds.shape[0]} entries, expected {n_sims}")
                if np.unique(ds[:]).size != ds.shape[0]:
                    problems.append("simulation_id contains duplicates")

        # --- mesh and mode shapes ----------------------------------------------
        if require_mode_shapes:
            ds = _get_dataset(f, "nodes/node_id", problems)
            if ds is not None:
                if ds.ndim != 1 or ds.dtype.kind not in "iu":
                    problems.append(f"node_id must be a 1-D integer array, got shape {ds.shape}, dtype {ds.dtype}")
                else:
                    info["n_nodes"] = ds.shape[0]
                    if np.unique(ds[:]).size != ds.shape[0]:
                        problems.append("node_id contains duplicates")

            coords = _get_dataset(f, "nodes/coordinates", problems, required=False)
            if coords is not None and info["n_nodes"] is not None and coords.shape != (info["n_nodes"], 3):
                problems.append(f"coordinates must be ({info['n_nodes']}, 3), got {coords.shape}")

            ms = f.get("simulations/mode/mode_shapes")
            if ms is None:
                problems.append("missing group 'simulations/mode/mode_shapes'")
            elif not isinstance(ms, h5py.Group):
                problems.append("'simulations/mode/mode_shapes' must be a group")
            else:
                if "components" in ms.attrs:
                    listed = [str(c.decode() if isinstance(c, bytes) else c) for c in ms.attrs["components"]]
                    wrong = [c for c in listed if c not in ALL_COMPONENTS]
                    if wrong:
                        problems.append(f"mode_shapes 'components' attribute has unknown entries {wrong}")
                    components = tuple(c for c in ALL_COMPONENTS if c in listed)
                    absent = [c for c in components if c not in ms]
                    if absent:
                        problems.append(f"mode_shapes lists {absent} but has no such dataset")
                else:
                    components = tuple(c for c in ALL_COMPONENTS if c in ms)
                if not components:
                    problems.append("mode_shapes has none of the datasets ux, uy, uz")
                info["components"] = components

                expected = (n_sims, info["n_modes"], info["n_nodes"])
                for c in components:
                    ds = _get_dataset(ms, c, problems, required=False)
                    if ds is None:
                        continue
                    if ds.ndim != 3 or not _is_numeric(ds):
                        problems.append(f"mode_shapes/{c} must be a 3-D numeric array, got shape {ds.shape}, dtype {ds.dtype}")
                    elif any(e is not None and s != e for s, e in zip(ds.shape, expected)):
                        exp = tuple("?" if e is None else e for e in expected)
                        problems.append(f"mode_shapes/{c} has shape {ds.shape}, expected (n_sims, n_modes, n_nodes) = {exp}")

        if problems:
            raise SimulationFileError(problems, getattr(f, "filename", None))
        return info


def _build_variable_set(f, registry) -> VariableSet:
    """Rebuild Q, turning constructor failures (e.g. a wrong number of
    parameters) into a SimulationFileError."""
    iv = f["simulations/input_variables"]
    try:
        return _records_to_variable_set(
            iv["distribution_parameters"][:], bool(iv.attrs.get("normalized", True)), registry
        )
    except (TypeError, ValueError, KeyError, AssertionError) as exc:
        raise SimulationFileError(
            [f"could not rebuild the distributions: {type(exc).__name__}: {exc}"],
            getattr(f, "filename", None),
        ) from exc


# ---------------------------------------------------------------------------
# 4. Which simulations still need to run
# ---------------------------------------------------------------------------
def pending_simulations(path, retry_failed: bool = True) -> np.ndarray:
    """Indices of simulations without a stored result. ``path`` can be a file
    path or an open ``h5py.File`` (left open).

    With ``retry_failed=True`` (default) failed simulations are included, so
    rerunning the loop retries them; with False only never-run ones are returned.
    A file without a status dataset counts as complete (empty result).
    """
    with _open_for_reading(path) as f:
        info = validate_simulation_file(f, require_mode_shapes=False)
        if not info["has_status"]:
            return np.array([], dtype=np.int64)
        status = f["simulations/status"][:]
    mask = status != DONE if retry_failed else status == PENDING
    return np.flatnonzero(mask)


# ---------------------------------------------------------------------------
# 5. Read everything back
# ---------------------------------------------------------------------------
def read_variable_set(path, registry: Mapping | None = None) -> VariableSet:
    """Rebuild only the VariableSet stored in the file. ``path`` can be a
    file path or an open ``h5py.File`` (left open)."""
    with _open_for_reading(path) as f:
        validate_simulation_file(f, require_samples=False, require_frequencies=False,
                                 require_mode_shapes=False, registry=registry)
        return _build_variable_set(f, registry)


def read_simulation_file(
    path,
    only_completed: bool = True,
    include_mode_shapes: bool = True,
    registry: Mapping | None = None,
):
    """Read a file written by ``create_simulation_file`` / ``write_simulation_result``.

    The file is first checked with ``validate_simulation_file``; if it lacks
    anything needed, a ``SimulationFileError`` lists every problem. Optional
    parts fall back to defaults: no ``status`` -> every simulation counts as
    done, no ``simulation_id`` -> 0..n_sims-1.

    Parameters
    ----------
    path : file path, or an open ``h5py.File``. A file object is read as is and
        left open; closing it is up to the caller.
    only_completed : keep only simulations with status "done" (default). If
        False, all rows are returned and unfinished ones are NaN.
    include_mode_shapes : if False, the QoI frame holds only frequencies, which
        avoids loading the (large) mode-shape arrays. Mode shapes and node IDs
        are then not required to be in the file.
    registry : extra {dist_type: class} entries for distribution classes not
        in DISTRIBUTION_CLASSES. Each class is called as ``cls(*params)``.

    Returns
    -------
    samples : DataFrame (n_sims, n_vars), columns = Q.variable_names()
    qoi : DataFrame with columns
        f.1 ... f.M, then S.<mode>.<node>.<x|y|z> ordered by mode, then
        component, then node:
        S.1.1.x ... S.1.N.x, S.1.1.y ... S.1.N.z, S.2.1.x ... S.M.N.z.
        Only the components stored in the file appear (e.g. no ".z" columns
        for a file created with components="xy").
        Mode and node numbers are 1-based positions; the node at position n is
        ``nodes/node_id[n-1]`` in the file.
    Q : VariableSet rebuilt from the stored distributions

    Both frames are indexed by ``simulation_id``.
    """
    with _open_for_reading(path) as f:
        info = validate_simulation_file(f, require_mode_shapes=include_mode_shapes,
                                        registry=registry)
        n_sims = info["n_sims"]

        status = f["simulations/status"][:] if info["has_status"] else np.full(n_sims, DONE)
        rows = np.flatnonzero(status == DONE) if only_completed else np.arange(n_sims)
        sim_ids = (f["simulations/simulation_id"][:] if info["has_simulation_id"]
                   else np.arange(n_sims))
        index = pd.Index(sim_ids[rows], name="simulation_id")

        Q = _build_variable_set(f, registry)
        samples = pd.DataFrame(f["simulations/input_variables/samples"][:][rows],
                               index=index, columns=Q.variable_names())

        freqs = f["simulations/mode/frequencies"][:][rows].astype(np.float64)
        n_modes = info["n_modes"]
        freq_cols = [f"f.{m}" for m in range(1, n_modes + 1)]

        if not include_mode_shapes:
            return samples, pd.DataFrame(freqs, index=index, columns=freq_cols), Q

        n_nodes = info["n_nodes"]
        components = info["components"]
        n = rows.size
        # (n, M, C, N): mode, then component, then node -> matches the column order
        shapes = np.empty((n, n_modes, len(components), n_nodes))
        for j, c in enumerate(components):
            ds = f[f"simulations/mode/mode_shapes/{c}"]
            step = ds.chunks[0] if ds.chunks else ds.shape[0]
            # Read whole chunk-sized blocks, keep the requested rows of each
            for start in range(0, ds.shape[0], step):
                stop = min(start + step, ds.shape[0])
                lo, hi = np.searchsorted(rows, [start, stop])
                if lo == hi:
                    continue
                shapes[lo:hi, :, j, :] = ds[start:stop][rows[lo:hi] - start]

    axes = [c[1] for c in components]  # "ux" -> "x"
    shape_cols = [
        f"S.{m}.{node}.{axis}"
        for m in range(1, n_modes + 1)
        for axis in axes
        for node in range(1, n_nodes + 1)
    ]
    data = np.concatenate([freqs, shapes.reshape(n, -1)], axis=1)
    del shapes
    qoi = pd.DataFrame(data, index=index, columns=freq_cols + shape_cols, copy=False)
    return samples, qoi, Q


# ---------------------------------------------------------------------------
# Import from DataFrames (the format read_simulation_file returns)
# ---------------------------------------------------------------------------
_FREQ_COL = re.compile(r"^f\.(\d+)$")
_SHAPE_COL = re.compile(r"^S\.(\d+)\.(\d+)\.([xyz])$")


def _parse_qoi_columns(columns):
    """Work out modes, nodes and components from f.<m> / S.<m>.<n>.<c> headers.

    Returns (n_modes, n_nodes, components, freq_pos, shape_pos) where
    freq_pos[m-1] is the column position of f.m and shape_pos[comp] is an
    (n_modes, n_nodes) array of column positions. Column order in the frame
    does not matter, but the set must be complete.
    """
    freq, shape, unknown = {}, {}, []
    for pos, col in enumerate(columns):
        col = str(col)
        if m := _FREQ_COL.match(col):
            freq[int(m.group(1))] = pos
        elif m := _SHAPE_COL.match(col):
            shape[(int(m.group(1)), int(m.group(2)), m.group(3))] = pos
        else:
            unknown.append(col)
    if unknown:
        raise ValueError(f"QoI columns not of the form f.<mode> or S.<mode>.<node>.<x|y|z>: "
                         f"{unknown[:5]}{' ...' if len(unknown) > 5 else ''}")
    if not freq:
        raise ValueError("No frequency columns (f.1, f.2, ...) found")

    n_modes = max(freq)
    if sorted(freq) != list(range(1, n_modes + 1)):
        raise ValueError(f"Frequency columns must be f.1 ... f.{n_modes} without gaps")
    freq_pos = np.array([freq[m] for m in range(1, n_modes + 1)])

    if not shape:
        raise ValueError("No mode-shape columns (S.<mode>.<node>.<x|y|z>) found")
    components = normalize_components(sorted({c for _, _, c in shape}))
    n_nodes = max(n for _, n, _ in shape)
    shape_modes = max(m for m, _, _ in shape)
    if shape_modes != n_modes:
        raise ValueError(f"{n_modes} frequency columns but mode shapes for {shape_modes} modes")

    expected = n_modes * n_nodes * len(components)
    if len(shape) != expected:
        missing = [
            f"S.{m}.{n}.{c[1]}"
            for m in range(1, n_modes + 1) for c in components for n in range(1, n_nodes + 1)
            if (m, n, c[1]) not in shape
        ]
        raise ValueError(f"Mode-shape columns incomplete, e.g. missing {missing[:5]}")

    shape_pos = {
        c: np.array([[shape[(m, n, c[1])] for n in range(1, n_nodes + 1)]
                     for m in range(1, n_modes + 1)])
        for c in components
    }
    return n_modes, n_nodes, components, freq_pos, shape_pos


def _as_frame(obj, name: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, (str, os.PathLike)):
        return pd.read_csv(obj)
    raise TypeError(f"{name} must be a DataFrame or a path to a CSV file")


def dataframes_to_hdf5(
    path,
    Q: VariableSet,
    samples,
    qoi,
    node_coordinates=None,
    node_ids=None,
    metadata: Mapping | None = None,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
    overwrite: bool = False,
) -> None:
    """Write samples and QoIs given as DataFrames (or CSV paths) to the HDF5 layout.

    Parameters
    ----------
    Q : VariableSet whose variable names match the sample columns. The CSV
        only holds values, so the distributions must come from Q.
    samples : DataFrame (n_sims, n_vars) or CSV path, columns = Q's variable
        names in any order.
    qoi : DataFrame or CSV path with columns f.1 ... f.M and
        S.<mode>.<node>.<x|y|z> for every mode, node and stored direction —
        the format ``read_simulation_file`` returns. Column order does not
        matter; the stored directions are taken from the headers.
    node_coordinates : (n_nodes, 3), optional. If omitted, coordinates are
        stored as NaN and flagged with ``nodes.attrs["coordinates_missing"]``.
    node_ids : (n_nodes,), optional. Defaults to 1..N, i.e. the node numbers
        used in the headers.

    Rows of ``samples`` and ``qoi`` are matched by index, which must be the
    same set in both. Integer indices become ``simulation_id``s. A row whose
    QoIs are all NaN is stored as pending; every other row is stored as done
    (individual NaNs, e.g. modes a solver did not find, are kept as NaN).
    """
    samples = _as_frame(samples, "samples")
    qoi = _as_frame(qoi, "qoi")

    if not samples.index.equals(qoi.index):
        if set(samples.index) != set(qoi.index) or samples.index.has_duplicates:
            raise ValueError("samples and qoi must have the same (unique) index")
        qoi = qoi.loc[samples.index]
    simulation_ids = (
        samples.index.to_numpy() if pd.api.types.is_integer_dtype(samples.index) else None
    )

    n_modes, n_nodes, components, freq_pos, shape_pos = _parse_qoi_columns(qoi.columns)
    if node_ids is None:
        node_ids = np.arange(1, n_nodes + 1)

    create_simulation_file(
        path, Q, samples, n_modes, node_coordinates,
        node_ids=node_ids, simulation_ids=simulation_ids, components=components,
        metadata=metadata, compression=compression,
        compression_opts=compression_opts, overwrite=overwrite,
    )

    values = qoi.to_numpy(dtype=np.float64)
    done = ~np.isnan(values).all(axis=1)
    with h5py.File(path, "r+") as f:
        f["simulations/mode/frequencies"][...] = values[:, freq_pos]
        for c in components:
            f[f"simulations/mode/mode_shapes/{c}"][...] = values[:, shape_pos[c]]
        f["simulations/status"][...] = np.where(done, DONE, PENDING).astype(np.int8)


# ---------------------------------------------------------------------------
# One-shot convenience wrapper
# ---------------------------------------------------------------------------
def write_simulation_hdf5(
    path,
    Q: VariableSet,
    samples,
    frequencies,
    mode_shapes,
    node_coordinates,
    node_ids=None,
    simulation_ids=None,
    components=None,
    metadata: Mapping | None = None,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
    overwrite: bool = True,
) -> None:
    """Write a complete set of results in one call (all data already in memory).

    frequencies : array, shape (n_sims, n_modes)
    mode_shapes : array (n_sims, n_modes, n_nodes, C) or
                  {component: (n_sims, n_modes, n_nodes)}
    components : directions stored. If None, taken from the dict keys, or
                 all three for an array with a last axis of 3.
    Other parameters as in ``create_simulation_file``.
    """
    frequencies = np.atleast_2d(np.asarray(frequencies, dtype=np.float64))
    if components is None:
        if isinstance(mode_shapes, Mapping):
            components = list(mode_shapes)
        elif np.shape(mode_shapes)[-1] == 3:
            components = ALL_COMPONENTS
        else:
            raise ValueError("Pass components=... when mode_shapes is not a dict with 3 directions")
    components = normalize_components(components)
    shapes = _split_components(mode_shapes, ndim=3, components=components)
    n_sims, n_modes = frequencies.shape

    create_simulation_file(
        path, Q, samples, n_modes, node_coordinates,
        node_ids=node_ids, simulation_ids=simulation_ids, components=components,
        metadata=metadata,
        compression=compression, compression_opts=compression_opts, overwrite=overwrite,
    )
    for i in range(n_sims):
        write_simulation_result(path, i, frequencies[i], {c: a[i] for c, a in shapes.items()})


# ---------------------------------------------------------------------------
# Example: a run that crashes partway and is resumed, then read back
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    Q = VariableSet()
    Q.add(Variable("m", UniformDistribution(0.5, 2.5)))
    Q.add(Variable("k", UniformDistribution(0.5, 2.5)))

    n_sims, n_modes, n_nodes = 20, 10, 500
    path = "example.h5"
    node_ids = np.arange(1, n_nodes + 1)  # Ansys-style 1-based node numbers

    create_simulation_file(
        path, Q,
        samples=Q.sample(n_sims, method="QMC_LHS", random_seed=1),
        n_modes=n_modes,
        node_coordinates=rng.random((n_nodes, 3)),
        node_ids=node_ids,
        metadata={"solver": "fake", "mode_normalization": "mass"},
        overwrite=True,
    )

    def fake_solve(i):
        if i == 7:
            raise RuntimeError("Solver did not converge")
        k = 8 if i == 3 else n_modes  # sim 3 finds fewer modes
        order = rng.permutation(node_ids)  # solver returns nodes in another order
        return rng.random(k), rng.random((k, n_nodes, 3)), order

    for i in pending_simulations(path):
        if i == 12:
            break  # simulated crash
        try:
            freqs, shapes, order = fake_solve(i)
            write_simulation_result(path, i, freqs, shapes, node_ids=order)
        except RuntimeError as exc:
            mark_failed(path, i, str(exc))
    print("After crash, pending:", pending_simulations(path))

    for i in pending_simulations(path, retry_failed=False):
        freqs, shapes, order = fake_solve(i)
        write_simulation_result(path, i, freqs, shapes, node_ids=order)
    print("After resume, pending:", pending_simulations(path))

    samples, qoi, Q_read = read_simulation_file(path)
    print(Q_read.get_variables())
    print(samples.head(3))
    print("QoI shape:", qoi.shape)
