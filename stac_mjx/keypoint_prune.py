"""Auto-detect keypoint pruning for STAC.

When the input mocap data is missing keypoints that the body model expects
(e.g. an amputated leg whose markers were never tracked), STAC would otherwise
fail: sites are added dynamically from ``cfg.model.KEYPOINT_MODEL_PAIRS`` and the
IK loss aligns each data column with the corresponding site, so a length / index
mismatch crashes the run.

``prune_model_to_available`` filters the model's keypoint-keyed config down to the
keypoints actually present in the data (intersection, preserving model order) and
reorders the flattened ``kp_data`` columns to match. For a full dataset where every
model keypoint is present this is a no-op, so existing pipelines are unaffected.
"""

from jax import numpy as jnp
from omegaconf import OmegaConf

# Config keys that are dicts keyed by keypoint name.
_KEYPOINT_KEYED_DICTS = (
    "KEYPOINT_INITIAL_OFFSETS",
    "KEYPOINT_COLOR_PAIRS",
    "KEYPOINT_WEIGHTS",
    "TRUNK_OPTIMIZATION_KEYPOINTS",
)
# Config keys that are lists of keypoint names.
_KEYPOINT_LISTS = ("KP_NAMES", "SITES_TO_REGULARIZE")


def prune_model_to_available(cfg, data_kp_names, kp_data):
    """Prune the model config + reorder ``kp_data`` to the data's keypoints.

    Args:
        cfg: Hydra/OmegaConf config; ``cfg.model`` is mutated in place.
        data_kp_names: ordered list of keypoint names present in ``kp_data``
            (column order, where ``kp_data`` is ``(n_frames, n_kp * 3)``).
        kp_data: flattened keypoint array ``(n_frames, n_kp * 3)``.

    Returns:
        ``(kp_data, present_names)`` where ``present_names`` is the model-ordered
        list of kept keypoints and ``kp_data`` columns are reordered to match.
    """
    data_kp_names = list(data_kp_names)
    data_set = set(data_kp_names)
    model_pairs = OmegaConf.to_container(cfg.model.KEYPOINT_MODEL_PAIRS, resolve=True)

    present = [k for k in model_pairs if k in data_set]
    dropped = [k for k in model_pairs if k not in data_set]
    extra = [k for k in data_kp_names if k not in model_pairs]

    # Full dataset (every model keypoint present, nothing extra): no change so
    # existing runs are byte-for-byte unaffected.
    if not dropped and not extra:
        return kp_data, data_kp_names

    print(f"  [prune] {len(present)}/{len(model_pairs)} model keypoints present in data.")
    if dropped:
        print(f"  [prune] Dropped model keypoints absent from data ({len(dropped)}): {dropped}")
    if extra:
        print(f"  [prune] Ignored data keypoints not in model ({len(extra)}): {extra}")

    # Reorder/subset kp_data columns to `present` (model) order.
    model_inds = [data_kp_names.index(k) for k in present]
    n_frames = kp_data.shape[0]
    kp3 = jnp.reshape(kp_data, (n_frames, len(data_kp_names), 3))
    kp3 = kp3[:, jnp.array(model_inds), :]
    kp_data = jnp.reshape(kp3, (n_frames, -1))

    present_set = set(present)
    struct_was = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.model.KEYPOINT_MODEL_PAIRS = {k: model_pairs[k] for k in present}

    for dict_key in _KEYPOINT_KEYED_DICTS:
        if dict_key in cfg.model and cfg.model[dict_key] is not None:
            d = OmegaConf.to_container(cfg.model[dict_key], resolve=True)
            if isinstance(d, dict):
                cfg.model[dict_key] = {k: v for k, v in d.items() if k in present_set}

    for list_key in _KEYPOINT_LISTS:
        if list_key in cfg.model and cfg.model[list_key] is not None:
            names = OmegaConf.to_container(cfg.model[list_key], resolve=True)
            if isinstance(names, list):
                cfg.model[list_key] = [n for n in names if n in present_set]

    # Validate trunk/root/orientation keypoints survived (they should for a leg
    # amputation — they are trunk markers). Warn rather than fail.
    root_kp = cfg.model.get("ROOT_OPTIMIZATION_KEYPOINT", None)
    if root_kp is not None and root_kp not in present_set:
        print(f"  [prune] WARNING: ROOT_OPTIMIZATION_KEYPOINT '{root_kp}' is absent from the data.")
    orient = cfg.model.get("JAXLS_ORIENTATION_KEYPOINTS", None)
    if orient:
        orient_c = OmegaConf.to_container(orient, resolve=True)
        missing = [v for v in orient_c.values() if v not in present_set]
        if missing:
            print(f"  [prune] WARNING: JAXLS_ORIENTATION_KEYPOINTS missing {missing}; "
                  "orientation warm-start will be disabled.")

    OmegaConf.set_struct(cfg, struct_was)
    return kp_data, present
