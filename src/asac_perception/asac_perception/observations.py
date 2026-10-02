"""Measurements/crops independent of ROS. Missing metric values remain None."""

import numpy as np

from asac_perception.geometry import (
    fit_sphere,
    mask_points,
    observation_direction,
    silhouette,
    transform_point,
)


def measure_instances(
    bgr, instances, depth, intrinsics, config, transform=None, depth_reason="", tf_reason=""
):
    observations = []
    h, w = bgr.shape[:2]
    for instance in instances:
        if instance.mask.shape != (h, w):
            raise ValueError("instance_mask_not_original_rgb_resolution")
        mask, shape = silhouette(instance.mask, config)
        if not mask.any():
            continue
        v, u = np.nonzero(mask)
        # Preserve detector bbox; ROI encloses both bbox and silhouette, exclusive right/bottom.
        bbox = np.asarray(instance.bbox, float).copy()
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, w)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, h)
        detector_bbox = bbox.copy()
        bbox[:2] = np.minimum(bbox[:2], [u.min(), v.min()])
        bbox[2:] = np.maximum(bbox[2:], [u.max() + 1, v.max() + 1])
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, w)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, h)
        x1, y1 = np.floor(bbox[:2]).astype(int)
        x2, y2 = np.ceil(bbox[2:]).astype(int)
        o = dict(shape)
        o.update(
            confidence=instance.confidence,
            class_id=instance.class_id,
            bbox=detector_bbox.tolist(),
            roi=[int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            image_width=w,
            image_height=h,
            view_type="top",
            rgb_crop=bgr[y1:y2, x1:x2].copy(),
            mask_crop=(mask[y1:y2, x1:x2].astype(np.uint8) * 255),
            full_mask=mask,
            center_camera=None,
            center_base=None,
            surface_camera=None,
            surface_base=None,
            direction_base=None,
            camera_pose_base=None,
            radius_m=None,
            diameter_m=None,
            center_camera_valid=False,
            center_base_valid=False,
            surface_valid=False,
            surface_base_valid=False,
            direction_valid=False,
            radius_valid=False,
            circularity_valid=shape["mask_valid"],
            valid_depth_ratio=0.0,
            valid_depth_points=0,
            depth_candidate_pixels=0,
            fit_valid=False,
            fit_rmse_m=None,
            fit_condition=None,
            inlier_ratio=None,
            cap_fraction=None,
            method="robust_sphere_visible_surface",
            reasons=[],
            occlusion_state="unknown_single_view",
            tracking_method="",
        )
        if shape["reason"]:
            o["reasons"].append(shape["reason"])
        if shape["solidity"] < config.min_solidity:
            o["occlusion_state"] = "possible_nonconvex_or_occluded"
            o["circularity_valid"] = False
            o["reasons"].append("nonconvex_silhouette_uncertain")
        # Bbox intersection is a conservative overlap cue, not an occlusion measurement.
        overlap = any(other is not instance and np.any(mask & other.mask) for other in instances)
        touching = any(
            other is not instance and _boxes_touch(bbox, other.bbox) for other in instances
        )
        if overlap or touching:
            o["occlusion_state"] = "possible_overlap"
            o["circularity_valid"] = False
            o["reasons"].append("possible_overlap")
        if depth is None or depth_reason:
            o["reasons"].append(depth_reason or "rgb_only_no_depth")
        else:
            try:
                points, stats = mask_points(mask, depth, *intrinsics, config)
                o.update(stats)
                if (
                    len(points) >= config.min_depth_points
                    and stats["valid_depth_ratio"] >= config.min_depth_ratio
                ):
                    # A representative observed surface point, explicitly separate from center.
                    median = np.median(points, axis=0)
                    nearest = np.argmin(np.linalg.norm(points - median, axis=1))
                    o["surface_camera"] = points[nearest].tolist()
                    o["surface_valid"] = True
                    fit = fit_sphere(points, config)
                    o.update({k: value for k, value in fit.items() if k != "reason"})
                    o["center_camera_valid"] = o["radius_valid"] = fit["fit_valid"]
                    if fit["reason"]:
                        o["reasons"].append(fit["reason"])
                else:
                    o["reasons"].append("insufficient_valid_depth")
            except (ValueError, RuntimeError) as error:
                o["reasons"].append(str(error))
        if transform is not None:
            translation, quaternion = transform
            try:
                # Buffer lookup is target=base, source=optical; apply exactly once here.
                o["camera_pose_base"] = {
                    "position": list(translation),
                    "quaternion_xyzw": list(quaternion),
                }
                if o["surface_valid"]:
                    o["surface_base"] = transform_point(
                        o["surface_camera"], translation, quaternion
                    )
                    o["surface_base_valid"] = True
                if o["center_camera_valid"]:
                    o["center_base"] = transform_point(o["center_camera"], translation, quaternion)
                    o["center_base_valid"] = True
                    o["direction_base"] = observation_direction(o["center_base"], translation)
                    o["direction_valid"] = True
            except ValueError as error:
                o.update(
                    center_base=None,
                    surface_base=None,
                    direction_base=None,
                    center_base_valid=False,
                    surface_base_valid=False,
                    direction_valid=False,
                    camera_pose_base=None,
                )
                o["reasons"].append(str(error))
        else:
            o["reasons"].append(tf_reason or "tf_unavailable")
        o["circularity_top_raw"] = o["circularity_top"]
        if not o["circularity_valid"]:
            o["circularity_top"] = None
        observations.append(o)
    return observations


def _boxes_touch(a, b):
    return bool(np.all(np.minimum(a[2:], b[2:]) > np.maximum(a[:2], b[:2])))


def metadata(observation):
    return {k: v for k, v in observation.items() if k not in ("rgb_crop", "mask_crop", "full_mask")}
