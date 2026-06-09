"""Rescale utils."""

import numpy as np
import mujoco
from mujoco import MjSpec

_MESH_GEOM = int(mujoco.mjtGeom.mjGEOM_MESH)


def _scale_vec(vec: list[float] | np.ndarray, s: float) -> None:
    """Scale a vector in-place by a scalar.

    Args:
        vec (list[float] | np.ndarray): The vector to scale.
        s (float): The scalar multiplier.

    Returns:
        None
    """
    for i in range(len(vec)):
        vec[i] *= s


def _scale_body_tree(body, s: float) -> None:
    """Recursively scale position, size, and fromto attributes on a body and its descendants.

    Args:
        body (Any): The body object to scale.
        s (float): The scalar multiplier.

    Returns:
        None
    """
    if hasattr(body, "pos"):
        _scale_vec(body.pos, s)

    for geom in body.geoms:
        if hasattr(geom, "pos"):
            _scale_vec(geom.pos, s)
        if hasattr(geom, "size"):
            _scale_vec(geom.size, s)
        if hasattr(geom, "fromto"):
            _scale_vec(geom.fromto, s)

    for site in body.sites:
        if hasattr(site, "pos"):
            _scale_vec(site.pos, s)
        if hasattr(site, "size"):
            _scale_vec(site.size, s)

    for joint in body.joints:
        if hasattr(joint, "pos"):
            _scale_vec(joint.pos, s)

    for child in body.bodies:
        _scale_body_tree(child, s)


def _recolour_geom(geom, rgba: list[float]) -> None:
    """Modify the color and collision group of a geometry, preserving original channels when rgba entry is -1."""
    # Capture original color channels
    original_rgba = list(geom.rgba)
    new_rgba = []
    # Build new RGBA, keeping original channel if new value is -1
    for orig_val, new_val in zip(original_rgba, rgba):
        if new_val == -1:
            new_rgba.append(orig_val)
        else:
            new_rgba.append(new_val)
    # If fewer new values provided, preserve any remaining original channels
    if len(original_rgba) > len(rgba):
        new_rgba.extend(original_rgba[len(rgba) :])
    geom.rgba = new_rgba
    geom.group = 2  # separate collision group


def _recolour_tree(body, rgba: list[float]) -> None:
    """Recursively recolor all geometries in a body and its descendants."""
    for geom in body.geoms:
        _recolour_geom(geom, rgba)
    for child in body.bodies:
        _recolour_tree(child, rgba)


def rescale_per_segment(spec: MjSpec, segments: list, scale_meshes: bool = False) -> MjSpec:
    """Morph individual body segments (subject-specific calibration).

    Unlike ``dm_scale_spec`` (one global scalar), this scales each named segment
    independently so the model matches an individual animal's proportions. For
    each segment it scales:
      - the ``geom_body``'s geoms (size / pos / fromto) -> the visual/collision
        segment lengthens, and
      - the ``length_body``'s ``pos`` -> the distal joint moves out, lengthening
        the kinematic segment.
    Scaling ``pos`` of a body translates its whole subtree without stretching the
    subtree's own segments (those are set by their own children's ``pos``), so
    segments do NOT compound — each is scaled exactly by its own factor. ``pos``
    and ``geom`` are distinct attributes, so a body that is the ``length_body``
    for its parent segment and the ``geom_body`` for its own segment is handled
    correctly (different factors on different attributes).

    Args:
        spec: base MjSpec (not mutated; a scaled copy is returned).
        segments: list of dicts with keys ``geom_body``, ``length_body``,
            ``scale`` (e.g. from utils.segment_calibration.estimate_segment_scales).
            ``self_segment`` entries simply have geom_body == length_body.

    Returns:
        MjSpec: the morphed spec (mutated in place; pass ``spec.copy()`` to isolate).

    Note: this model uses MESH geoms for the visual body and PRIMITIVE geoms for
    collision. MuJoCo ignores ``size`` on mesh geoms (they use the mesh asset's
    own ``scale``) and the mesh geom ``pos`` is a placement offset — scaling
    either of those displaces/does-not-resize the mesh. So for a mesh geom we
    scale its (unique) mesh ASSET; for primitive geoms we scale size/pos/fromto.
    """
    scaled = spec
    mesh_by_name = {m.name: m for m in scaled.meshes}

    def _body(name):
        try:
            return scaled.body(name)
        except (KeyError, ValueError):
            return None

    for seg in segments:
        s = float(seg['scale'])
        if abs(s - 1.0) < 1e-9:
            continue
        gb = _body(seg['geom_body'])
        if gb is not None:
            for geom in gb.geoms:
                if int(geom.type) == _MESH_GEOM:
                    # Visual mesh. Scaling its asset isotropically grows it about
                    # the mesh's local origin (not the proximal joint), which
                    # fragments articulated leg meshes — so it's opt-in. The
                    # kinematic morph (length_body.pos) alone gives correct IK;
                    # meshes left at base size just under-fill long segments.
                    if scale_meshes:
                        mname = getattr(geom, 'meshname', '') or ''
                        if mname in mesh_by_name:
                            _scale_vec(mesh_by_name[mname].scale, s)
                else:
                    # Collision primitive (capsule/ellipsoid/...): scale geometry.
                    if hasattr(geom, 'size'):
                        _scale_vec(geom.size, s)
                    if hasattr(geom, 'pos'):
                        _scale_vec(geom.pos, s)
                    if hasattr(geom, 'fromto'):
                        _scale_vec(geom.fromto, s)
        # Kinematics: move the distal joint out so the segment length matches.
        lb = _body(seg['length_body'])
        if lb is not None and getattr(lb, 'pos', None) is not None:
            _scale_vec(lb.pos, s)
    return scaled


def dm_scale_spec(spec: MjSpec, scale: float) -> MjSpec:
    """Scale a spec by a scalar.

    Args:
        spec (MjSpec): The spec to scale.
        scale (float): The scalar multiplier.

    Returns:
        MjSpec: The scaled spec.
    """
    scaled_spec = spec.copy()

    # Traverse the kinematic tree, scaling all geoms
    def scale_bodies(parent, scale=1.0):
        body = parent.first_body()
        while body:
            if body.pos is not None:
                body.pos = body.pos * scale
            for geom in body.geoms:
                geom.fromto = geom.fromto * scale
                geom.size = geom.size * scale
                if geom.pos is not None:
                    geom.pos = geom.pos * scale
            scale_bodies(body, scale)
            body = parent.next_body(body)

    # if scale_actuators:
    # # scale gear
    for mesh in scaled_spec.meshes:
        mesh.scale = mesh.scale * scale

    for actuator in scaled_spec.actuators:
        # scale the actuator gear by (scale ** 2),
        # this is because muscle force-generating capacity
        # scales with the cross-sectional area of the muscle
        actuator.gear = actuator.gear * scale * scale

    # scale the z-position for all keypoints
    for keypoint in scaled_spec.keys:
        qpos = keypoint.qpos
        qpos[2] = qpos[2] * scale
        keypoint.qpos = qpos
        keypoint.qpos[2] = keypoint.qpos[2] * scale

    scale_bodies(scaled_spec.worldbody.first_body(), scale)
    return scaled_spec
