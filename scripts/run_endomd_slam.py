import argparse
import json
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.gradslam_datasets import (
    load_dataset_config,
    C3VDDataset
)
from utils.common_utils import seed_everything, save_params, save_means3D
from utils.eval_helpers import report_progress, eval_save
from utils.keyframe_selection import keyframe_selection_distance
from utils.recon_helpers import setup_camera, energy_mask
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians
from utils.vis_utils import plot_video
from utils.time_helper import Timer

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def _inverse_sigmoid_scalar(x: float) -> float:
    x = float(x)
    x = min(max(x, 1e-6), 1.0 - 1e-6)
    return float(np.log(x / (1.0 - x)))


def _transient_init_log_scales_from_points(
    pts_local_xyz: torch.Tensor, *, transient_cfg: dict
) -> torch.Tensor:
    """Return log_scales (N,1) for transient points, matching DeSplat's kNN-based scale init when requested."""
    if not isinstance(pts_local_xyz, torch.Tensor) or pts_local_xyz.ndim != 2 or pts_local_xyz.shape[1] != 3:
        raise ValueError(f"Expected pts_local_xyz as (N,3) tensor, got {None if pts_local_xyz is None else tuple(pts_local_xyz.shape)}")

    mode = str(transient_cfg.get("init_scale_mode", "const")).lower()
    if mode in {"const", "constant"}:
        scale_init = float(transient_cfg.get("scale_init", 0.01))
        return torch.full(
            (pts_local_xyz.shape[0], 1),
            float(np.log(max(scale_init, 1e-8))),
            device=pts_local_xyz.device,
            dtype=pts_local_xyz.dtype,
        )

    if mode in {"knn", "knn3"}:
        n = int(pts_local_xyz.shape[0])
        if n <= 1:
            scale_init = float(transient_cfg.get("scale_init", 0.01))
            return torch.full(
                (n, 1),
                float(np.log(max(scale_init, 1e-8))),
                device=pts_local_xyz.device,
                dtype=pts_local_xyz.dtype,
            )
        # DeSplat: scales_dyn = log(avg_dist_dyn.repeat(1,3)); avg_dist_dyn from 3-NN distances.
        # We use torch.cdist for a small N (default 1000) initialization.
        d = torch.cdist(pts_local_xyz, pts_local_xyz)  # (N,N)
        d.fill_diagonal_(float("inf"))
        k = min(3, n - 1)
        knn = torch.topk(d, k=k, largest=False, dim=-1).values  # (N,k)
        avg = knn.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.log(avg)

    raise ValueError(f"Unknown transient.init_scale_mode={mode!r} (expected 'const' or 'knn3').")


def _init_empty_transient_params(params: dict) -> dict:
    """Add empty transient parameter tensors into `params` (incremental; guarded by config)."""
    if "t_means3D" in params:
        return params
    device = params["means3D"].device
    dtype = params["means3D"].dtype
    params["t_means3D"] = torch.nn.Parameter(torch.zeros((0, 3), device=device, dtype=dtype).requires_grad_(True))
    params["t_rgb_colors"] = torch.nn.Parameter(torch.zeros((0, 3), device=device, dtype=dtype).requires_grad_(True))
    params["t_unnorm_rotations"] = torch.nn.Parameter(torch.zeros((0, 4), device=device, dtype=dtype).requires_grad_(True))
    params["t_logit_opacities"] = torch.nn.Parameter(torch.zeros((0, 1), device=device, dtype=dtype).requires_grad_(True))
    params["t_log_scales"] = torch.nn.Parameter(torch.zeros((0, 1), device=device, dtype=dtype).requires_grad_(True))
    return params


def _get_transient_slice(frame_idx: int, *, num_points_per_frame: int) -> slice:
    start = int(frame_idx) * int(num_points_per_frame)
    end = start + int(num_points_per_frame)
    return slice(start, end)


def _ensure_transient_for_frame(
    *,
    params: dict,
    frame_idx: int,
    transient_cfg: dict,
    init_color_chw: torch.Tensor | None = None,
    init_depth_chw: torch.Tensor | None = None,
    init_intrinsics_3x3: torch.Tensor | None = None,
) -> dict:
    """Lazily append a fixed-size transient set for each new frame (0..frame_idx)."""
    params = _init_empty_transient_params(params)

    n_per = int(transient_cfg.get("num_points_per_frame", 0))
    if n_per <= 0:
        return params

    already = int(params["t_means3D"].shape[0]) // n_per
    if already > frame_idx:
        return params

    init_mode = str(transient_cfg.get("init_mode", "cube")).lower()
    init_distance = float(transient_cfg.get("init_distance", 0.02))
    init_xy_range = float(transient_cfg.get("init_xy_range", 0.02))
    opacity_init = float(transient_cfg.get("opacity_init", 0.1))
    scale_init = float(transient_cfg.get("scale_init", 0.01))

    warm_mode = str(transient_cfg.get("warmstart_mode", "none")).lower()
    warm_ratio = float(transient_cfg.get("warmstart_ratio", 1.0))
    warm_ratio = float(max(0.0, min(1.0, warm_ratio)))
    warm_opacity = str(transient_cfg.get("warmstart_opacity", "reset")).lower()
    warm_opacity_decay = float(transient_cfg.get("warmstart_opacity_decay", 0.5))
    warm_clamp_z = bool(transient_cfg.get("warmstart_clamp_z", True))

    device = params["means3D"].device
    dtype = params["means3D"].dtype

    for j in range(already, int(frame_idx) + 1):
        # Build current estimated camera pose (world -> cam) and invert for init (cam -> world).
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., j].detach())
        cam_tran = params["cam_trans"][..., j].detach()
        rel_w2c = torch.eye(4, device=device, dtype=dtype)
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        c2w = torch.linalg.inv(rel_w2c)

        # Optional warm-start: reuse previous-frame transient in *camera coordinates* (lens-attached assumption).
        warm_pts_cam = None
        warm_rgb = None
        warm_logit = None
        if warm_mode in {"prev_cam", "prevcam", "prev_camera", "previous_camera"} and int(j) > 0 and warm_ratio > 0.0:
            sl_prev = _get_transient_slice(int(j) - 1, num_points_per_frame=int(n_per))
            if int(params["t_means3D"].shape[0]) >= int(sl_prev.stop):
                prev_world = params["t_means3D"].detach()[sl_prev]
                prev_rgb = params["t_rgb_colors"].detach()[sl_prev] if "t_rgb_colors" in params else None
                prev_logit_all = (
                    params["t_logit_opacities"].detach()[sl_prev] if "t_logit_opacities" in params else None
                )

                prev_cam_rot = F.normalize(params["cam_unnorm_rots"][..., int(j) - 1].detach())
                prev_cam_tran = params["cam_trans"][..., int(j) - 1].detach()
                w2c_prev = torch.eye(4, device=device, dtype=dtype)
                w2c_prev[:3, :3] = build_rotation(prev_cam_rot)
                w2c_prev[:3, 3] = prev_cam_tran

                ones_prev = torch.ones((int(prev_world.shape[0]), 1), device=device, dtype=dtype)
                prev4 = torch.cat([prev_world.to(device=device, dtype=dtype), ones_prev], dim=-1)
                prev_cam = (w2c_prev @ prev4.T).T[:, :3].contiguous()

                warm_n = int(round(float(n_per) * float(warm_ratio)))
                warm_n = int(max(0, min(int(n_per), warm_n)))
                if warm_n > 0:
                    sel = torch.randperm(int(n_per), device=device)[:warm_n]
                    warm_pts_cam = prev_cam[sel].contiguous()
                    if warm_clamp_z:
                        warm_pts_cam[:, 2] = torch.clamp(warm_pts_cam[:, 2], min=1e-4)
                    if isinstance(prev_rgb, torch.Tensor):
                        warm_rgb = prev_rgb.to(device=device, dtype=dtype)[sel].contiguous()
                    if isinstance(prev_logit_all, torch.Tensor) and warm_opacity in {"copy", "decay"}:
                        if warm_opacity == "copy":
                            warm_logit = prev_logit_all.to(device=device, dtype=dtype)[sel].contiguous()
                        else:
                            decay = float(max(0.0, min(1.0, warm_opacity_decay)))
                            a_prev = torch.sigmoid(prev_logit_all.to(device=device, dtype=dtype)[sel])
                            a_new = torch.clamp(a_prev * decay, 1e-6, 1.0 - 1e-6)
                            warm_logit = torch.log(a_new / (1.0 - a_new)).contiguous()

        # Sample points in camera coordinates: z positive forward (consistent with get_pointcloud()).
        pts_cam = None
        rgb = None
        if (
            init_mode in {"depth", "depth_backproject", "depth_map"}
            and isinstance(init_depth_chw, torch.Tensor)
            and isinstance(init_intrinsics_3x3, torch.Tensor)
        ):
            depth_hw = init_depth_chw[0].to(device=device, dtype=dtype)
            h, w = int(depth_hw.shape[0]), int(depth_hw.shape[1])
            valid = (depth_hw > 0) & torch.isfinite(depth_hw) & (depth_hw < 1e10)
            idx = torch.where(valid.reshape(-1))[0]
            if idx.numel() > 0:
                # Sample pixels uniformly from valid depth.
                if idx.numel() >= n_per:
                    pick = idx[torch.randperm(idx.numel(), device=device)[:n_per]]
                else:
                    pick = idx[torch.randint(0, idx.numel(), (n_per,), device=device)]
                v = (pick // w).to(dtype=dtype)
                u = (pick % w).to(dtype=dtype)
                z = depth_hw.reshape(-1)[pick].reshape(-1, 1)  # (N,1)
                K = init_intrinsics_3x3.to(device=device, dtype=dtype)
                fx = K[0, 0].clamp_min(1e-12)
                fy = K[1, 1].clamp_min(1e-12)
                cx = K[0, 2]
                cy = K[1, 2]
                x = (u.reshape(-1, 1) - cx) / fx * z
                y = (v.reshape(-1, 1) - cy) / fy * z
                pts_cam = torch.cat([x, y, z], dim=-1).contiguous()

                # Initialize colors from the sampled pixels when available.
                if isinstance(init_color_chw, torch.Tensor) and init_color_chw.shape[0] == 3:
                    col = init_color_chw.to(device=device, dtype=dtype)
                    u_i = u.to(dtype=torch.long).clamp(0, w - 1)
                    v_i = v.to(dtype=torch.long).clamp(0, h - 1)
                    rgb = col[:, v_i, u_i].T.contiguous()  # (N,3)

        if pts_cam is None:
            xy = (torch.rand((n_per, 2), device=device, dtype=dtype) - 0.5) * float(init_xy_range)
            z = torch.full((n_per, 1), float(init_distance), device=device, dtype=dtype)
            pts_cam = torch.cat([xy, z], dim=-1)  # [N,3]

        # Apply warm-start into the sampled camera-space points/colors.
        if isinstance(warm_pts_cam, torch.Tensor) and int(warm_pts_cam.shape[0]) > 0:
            m = int(min(int(n_per), int(warm_pts_cam.shape[0])))
            pts_cam[:m] = warm_pts_cam[:m]
            if isinstance(warm_rgb, torch.Tensor):
                if rgb is None:
                    rgb = torch.rand((n_per, 3), device=device, dtype=dtype).contiguous()
                rgb[:m] = warm_rgb[:m]
        ones = torch.ones((n_per, 1), device=device, dtype=dtype)
        pts4 = torch.cat([pts_cam, ones], dim=-1)  # [N,4]
        pts_world = (c2w @ pts4.T).T[:, :3].contiguous()

        if rgb is None:
            rgb = torch.rand((n_per, 3), device=device, dtype=dtype).contiguous()
        rot = torch.zeros((n_per, 4), device=device, dtype=dtype)
        rot[:, 0] = 1.0  # identity quaternion
        logit = torch.full((n_per, 1), _inverse_sigmoid_scalar(opacity_init), device=device, dtype=dtype)
        if isinstance(warm_logit, torch.Tensor) and int(warm_logit.shape[0]) > 0:
            m = int(min(int(n_per), int(warm_logit.shape[0])))
            logit[:m] = warm_logit[:m]
        # When using const scale init, honor scale_init from config; knn3 ignores it.
        if str(transient_cfg.get("init_scale_mode", "const")).lower() in {"const", "constant"}:
            transient_cfg_eff = dict(transient_cfg)
            transient_cfg_eff["scale_init"] = scale_init
        else:
            transient_cfg_eff = transient_cfg
        log_scales = _transient_init_log_scales_from_points(pts_cam, transient_cfg=transient_cfg_eff)

        params["t_means3D"] = torch.nn.Parameter(
            torch.cat([params["t_means3D"], pts_world], dim=0).requires_grad_(True)
        )
        params["t_rgb_colors"] = torch.nn.Parameter(
            torch.cat([params["t_rgb_colors"], rgb], dim=0).requires_grad_(True)
        )
        params["t_unnorm_rotations"] = torch.nn.Parameter(
            torch.cat([params["t_unnorm_rotations"], rot], dim=0).requires_grad_(True)
        )
        params["t_logit_opacities"] = torch.nn.Parameter(
            torch.cat([params["t_logit_opacities"], logit], dim=0).requires_grad_(True)
        )
        params["t_log_scales"] = torch.nn.Parameter(
            torch.cat([params["t_log_scales"], log_scales], dim=0).requires_grad_(True)
        )

    return params


def _render_transient_alpha_hw(
    *,
    params: dict,
    frame_idx: int,
    cam,
    w2c: torch.Tensor,
    transient_cfg: dict,
) -> torch.Tensor | None:
    n_per = int(transient_cfg.get("num_points_per_frame", 0))
    if n_per <= 0:
        return None
    if "t_means3D" not in params:
        return None

    sl = _get_transient_slice(int(frame_idx), num_points_per_frame=int(n_per))
    if int(params["t_means3D"].shape[0]) < int(sl.stop):
        return None

    t_means = params["t_means3D"].detach()[sl]
    tparams = {
        "means3D": t_means,
        "rgb_colors": params["t_rgb_colors"].detach()[sl] if "t_rgb_colors" in params else None,
        "unnorm_rotations": params["t_unnorm_rotations"].detach()[sl],
        "logit_opacities": params["t_logit_opacities"].detach()[sl],
        "log_scales": params["t_log_scales"].detach()[sl],
    }
    if tparams["rgb_colors"] is None:
        tparams["rgb_colors"] = torch.zeros_like(t_means)
    if bool(transient_cfg.get("clamp_colors", False)):
        tparams["rgb_colors"] = tparams["rgb_colors"].clamp(0.0, 1.0)

    tpose = {
        "means3D": t_means,
        "cam_unnorm_rots": params["cam_unnorm_rots"],
        "cam_trans": params["cam_trans"],
    }
    t_transformed = transform_to_frame(
        tpose,
        int(frame_idx),
        gaussians_grad=False,
        camera_grad=False,
    )
    t_depth_sil_rendervar = transformed_params2depthplussilhouette(tparams, w2c, t_transformed)
    t_depth_sil, _, _ = _unpack_renderer_out(Renderer(raster_settings=cam)(**t_depth_sil_rendervar))
    return t_depth_sil[1, :, :].clamp(0.0, 1.0)


def _transient_refine_inplace(
    *,
    params: dict,
    variables: dict,
    time_idx: int,
    transient_cfg: dict,
    iter_idx: int,
) -> None:
    """DeSplat-style dynamic prune/densify adapted to fixed per-frame capacity via slot recycling.

    - "Cull" becomes: identify bad points (low alpha / invisible / too big), then recycle slots.
    - "Densify" becomes: optionally duplicate high-gradient points into recycled slots.
    """
    if not isinstance(transient_cfg, dict) or not bool(transient_cfg.get("refine_enabled", False)):
        return
    refine_every = int(transient_cfg.get("refine_every", 0))
    if refine_every <= 0 or (int(iter_idx) % refine_every) != 0:
        return

    info = variables.get("transient_refine_info", None)
    if not isinstance(info, dict):
        return
    if int(info.get("frame_idx", -1)) != int(time_idx):
        return
    start = int(info.get("slice_start", -1))
    end = int(info.get("slice_end", -1))
    if start < 0 or end <= start:
        return
    if "t_logit_opacities" not in params or "t_means3D" not in params:
        return

    sl = slice(start, end)
    device = params["t_means3D"].device
    dtype = params["t_means3D"].dtype

    op = torch.sigmoid(params["t_logit_opacities"][sl]).reshape(-1)
    cull_alpha_thresh = float(transient_cfg.get("cull_alpha_thresh", 0.005))
    cull = (op < cull_alpha_thresh)

    # Cull invisible points (DeSplat uses radii_dyn < 0.01).
    radius = info.get("radius", None)
    if isinstance(radius, torch.Tensor) and radius.numel() == op.numel():
        cull_radius_thresh = float(transient_cfg.get("cull_radius_thresh", 0.0))
        if cull_radius_thresh > 0:
            cull = cull | (radius.detach().reshape(-1) < cull_radius_thresh)

    # Cull too-large scales (optional).
    cull_scale_thresh = transient_cfg.get("cull_scale_thresh", None)
    if cull_scale_thresh is not None:
        try:
            cull_scale_thresh_f = float(cull_scale_thresh)
            scales = torch.exp(params["t_log_scales"][sl]).reshape(-1)
            cull = cull | (scales > cull_scale_thresh_f)
        except Exception:
            pass

    dead = torch.where(cull.reshape(-1))[0]
    if dead.numel() == 0:
        return

    # Optional densify: duplicate high-gradient points into dead slots.
    strategy = str(transient_cfg.get("recycle_strategy", "random")).lower()
    means2d = info.get("means2D", None)
    means2d_grad = None
    if isinstance(means2d, torch.Tensor):
        means2d_grad = means2d.grad

    did_dup = False
    if strategy in {"dup_grad_or_random", "dup_grad"} and isinstance(means2d_grad, torch.Tensor):
        g = means2d_grad.detach()
        if g.ndim == 2 and g.shape[0] == op.numel():
            gnorm = torch.linalg.vector_norm(g, dim=-1)
            alive = (~cull).detach()
            alive_idcs = torch.where(alive)[0]
            if alive_idcs.numel() > 0:
                k = min(int(dead.numel()), int(alive_idcs.numel()))
                top = alive_idcs[torch.topk(gnorm[alive_idcs], k=k, largest=True).indices]
                # Duplicate these k points into k dead slots.
                dst = dead[:k]
                src = top
                jitter_mult = float(transient_cfg.get("dup_jitter_mult", 0.25))
                src_scales = torch.exp(params["t_log_scales"][sl][src]).reshape(-1, 1)
                jitter = torch.randn((k, 3), device=device, dtype=dtype) * (src_scales * jitter_mult)

                params["t_means3D"].data[start + dst] = params["t_means3D"].data[start + src] + jitter
                params["t_rgb_colors"].data[start + dst] = params["t_rgb_colors"].data[start + src]
                params["t_unnorm_rotations"].data[start + dst] = params["t_unnorm_rotations"].data[start + src]
                params["t_logit_opacities"].data[start + dst] = params["t_logit_opacities"].data[start + src]
                params["t_log_scales"].data[start + dst] = params["t_log_scales"].data[start + src]
                did_dup = True

                # Remaining dead slots (if any) will be randomized below.
                dead = dead[k:]

    if dead.numel() == 0:
        return

    # Random re-init for remaining dead slots: sample in front of current camera.
    init_distance = float(transient_cfg.get("init_distance", 0.02))
    init_xy_range = float(transient_cfg.get("init_xy_range", 0.02))
    opacity_init = float(transient_cfg.get("opacity_init", 0.1))

    cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
    cam_tran = params["cam_trans"][..., time_idx].detach()
    rel_w2c = torch.eye(4, device=device, dtype=dtype)
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran
    c2w = torch.linalg.inv(rel_w2c)

    n = int(dead.numel())
    xy = (torch.rand((n, 2), device=device, dtype=dtype) - 0.5) * float(init_xy_range)
    z = torch.full((n, 1), float(init_distance), device=device, dtype=dtype)
    pts_cam = torch.cat([xy, z], dim=-1)  # (n,3)
    ones = torch.ones((n, 1), device=device, dtype=dtype)
    pts4 = torch.cat([pts_cam, ones], dim=-1)  # (n,4)
    pts_world = (c2w @ pts4.T).T[:, :3].contiguous()

    rgb = torch.rand((n, 3), device=device, dtype=dtype).contiguous()
    rot = torch.zeros((n, 4), device=device, dtype=dtype)
    rot[:, 0] = 1.0
    logit = torch.full((n, 1), _inverse_sigmoid_scalar(opacity_init), device=device, dtype=dtype)
    log_scales = _transient_init_log_scales_from_points(pts_cam, transient_cfg=transient_cfg)

    idx = (start + dead).to(dtype=torch.long)
    params["t_means3D"].data[idx] = pts_world
    params["t_rgb_colors"].data[idx] = rgb
    params["t_unnorm_rotations"].data[idx] = rot
    params["t_logit_opacities"].data[idx] = logit
    params["t_log_scales"].data[idx] = log_scales


def _make_jsonable(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_make_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _make_jsonable(v) for k, v in x.items()}
    # Fallback: represent non-JSON types as strings (e.g., Path-like, numpy scalars).
    try:
        return _make_jsonable(x.item())  # numpy scalar / torch scalar
    except Exception:
        return str(x)


def _unpack_renderer_out(out, *, require_radius: bool = False):
    if isinstance(out, (tuple, list)):
        if len(out) == 2:
            a, b = out
            c = None
        elif len(out) == 3:
            a, b, c = out
        elif len(out) == 1:
            a = out[0]
            b = None
            c = None
        else:
            a = out[0]
            b = out[1] if len(out) > 1 else None
            c = out[2] if len(out) > 2 else None
    else:
        a, b, c = out, None, None
    if require_radius and b is None:
        raise ValueError("Renderer did not return radius (expected tuple with radius as 2nd element).")
    return a, b, c


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["c3vd"]:
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"EndoMD-SLAM release supports C3VD/C3VDv2 data, got {config_dict['dataset_name']!r}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)
    # Image.fromarray(np.uint8((torch.permute(color, (1, 2, 0)) * mask.reshape(320, 320, 1)).detach().cpu().numpy()*255), 'RGB').save('gaussian.png')

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification=True):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
    }
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() if k != 'feature_rest']
    if 'feature_rest' in params:
        param_groups.append({'params': [params['feature_rest']], 'name': 'feature_rest', 'lr': lrs['rgb_colors'] / 20.0})
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_transient_optimizer(params, lrs_dict):
    """Optimizer for transient-only mapping: only transient parameters (t_*)."""
    lrs = lrs_dict
    keys = ["t_means3D", "t_rgb_colors", "t_unnorm_rotations", "t_logit_opacities", "t_log_scales"]
    param_groups = []
    for k in keys:
        if k in params and k in lrs:
            param_groups.append({"params": [params[k]], "name": k, "lr": float(lrs[k])})
    if not param_groups:
        return torch.optim.Adam([], lr=0.0, eps=1e-15)
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(
    dataset,
    num_frames,
    scene_radius_depth_ratio,
    mean_sq_dist_method,
    use_simplification=True,
):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), use_simplification=use_simplification)

    # Get Initial Point Cloud (PyTorch CUDA Tensor)

    mask = (depth > 0) & energy_mask(color) # Mask out invalid depth values
    # Image.fromarray(np.uint8(mask[0].detach().cpu().numpy()*255), 'L').save('mask.png')
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, intrinsics, w2c,
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification)

    # Scene radius is used by pruning to remove oversized Gaussians.
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    return params, variables, intrinsics, w2c, cam


def get_loss(
    params,
    curr_data,
    variables,
    iter_time_idx,
    loss_weights,
    use_sil_for_loss,
    sil_thres,
    use_l1,
    ignore_outlier_depth_loss,
):
    global w2cs, w2ci
    # Initialize Loss Dictionary
    losses = {}

    freeze_static = bool(curr_data.get("freeze_static", False))

    if freeze_static:
        # Transient-only mapping: keep the static map frozen (no gradients).
        transformed_pts = transform_to_frame(
            params,
            iter_time_idx,
            gaussians_grad=False,
            camera_grad=False,
        )
    else:
        # Mapping updates the Gaussian map; poses come from MASt3R metric tracking.
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)
    
    # Visualize the Rendered Images
    # online_render(curr_data, iter_time_idx, rendervar, dev_use_controller=False)
        
    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _ = _unpack_renderer_out(
        Renderer(raster_settings=curr_data['cam'])(**rendervar),
        require_radius=True,
    )
    variables['means2D'] = rendervar['means2D']


    # Depth & Silhouette Rendering
    depth_sil, _, _ = _unpack_renderer_out(Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar))
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    alpha_s = silhouette.clamp(0.0, 1.0)
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    bg_mask = energy_mask(curr_data['im'])
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 20*depth_error.mean())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask & bg_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if use_sil_for_loss:
        mask = mask & presence_sil_mask

    # DeSplat-style per-frame transient Gaussians (mapping only).
    transient_cfg = curr_data.get("transient", None)
    use_transient = (
        isinstance(transient_cfg, dict)
        and bool(transient_cfg.get("enabled", False))
        and ("t_means3D" in params)
    )
    alpha_t = None
    # DeSplat alpha-bg regularization for static (prevent holes).
    if use_transient and float(loss_weights.get("alpha_bg", 0.0)) > 0.0:
        # DeSplat (without bg model): alpha_bg_loss = lambda * mean(1 - alpha_static).
        losses["alpha_bg"] = (1.0 - alpha_s).mean()
    if use_transient:
        n_per = int(transient_cfg.get("num_points_per_frame", 0))
        if n_per > 0:
            sl = _get_transient_slice(int(iter_time_idx), num_points_per_frame=n_per)
                # If transient was not initialized, skip gracefully.
            if int(params["t_means3D"].shape[0]) >= int(sl.stop):
                t_means = params["t_means3D"][sl]
                tparams = {
                    "means3D": t_means,
                    "rgb_colors": params["t_rgb_colors"][sl],
                    "unnorm_rotations": params["t_unnorm_rotations"][sl],
                    "logit_opacities": params["t_logit_opacities"][sl],
                    "log_scales": params["t_log_scales"][sl],
                }
                if bool(transient_cfg.get("clamp_colors", False)):
                    tparams["rgb_colors"] = tparams["rgb_colors"].clamp(0.0, 1.0)

                tpose = {
                    "means3D": t_means,
                    "cam_unnorm_rots": params["cam_unnorm_rots"],
                    "cam_trans": params["cam_trans"],
                }
                t_transformed = transform_to_frame(
                    tpose,
                    iter_time_idx,
                    gaussians_grad=True,
                    camera_grad=False,
                )

                # Transient RGB.
                t_rendervar = transformed_params2rendervar(tparams, t_transformed)
                want_refine_info = bool(transient_cfg.get("refine_enabled", False)) and int(curr_data.get("online_time_idx", -1)) == int(iter_time_idx)
                if want_refine_info:
                    t_rendervar["means2D"].retain_grad()
                im_t, t_radius, _ = _unpack_renderer_out(
                    Renderer(raster_settings=curr_data["cam"])(**t_rendervar),
                    require_radius=True,
                )
                if want_refine_info:
                    variables["transient_refine_info"] = {
                        "frame_idx": int(iter_time_idx),
                        "slice_start": int(sl.start),
                        "slice_end": int(sl.stop),
                        "means2D": t_rendervar["means2D"],
                        "radius": t_radius,
                    }

                # Transient alpha proxy from silhouette.
                t_depth_sil_rendervar = transformed_params2depthplussilhouette(
                    tparams, curr_data["w2c"], t_transformed
                )
                t_depth_sil, _, _ = _unpack_renderer_out(
                    Renderer(raster_settings=curr_data["cam"])(**t_depth_sil_rendervar)
                )
                alpha_t = t_depth_sil[1, :, :].clamp(0.0, 1.0)

                # Compose: transient overlays static.
                im = (im_t + (1.0 - alpha_t.unsqueeze(0)) * im).clamp(0.0, 1.0)

    loss_mode = None
    if use_transient:
        loss_mode = str(transient_cfg.get("loss_mode", "")).lower()

    # Depth loss (disabled for DeSplat RGB-only transient mode).
    compute_depth_loss = not (use_transient and loss_mode in {"desplat_rgb_only", "desplat"})
    if compute_depth_loss:
        if use_l1:
            mask = mask.detach()
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    # RGB Loss
    if use_transient and loss_mode in {"desplat_rgb_only", "desplat"}:
        # Match DeSplat: main_loss = (1-ssim_lambda)*L1 + ssim_lambda*(1-SSIM), no depth supervision.
        ssim_lambda = float(transient_cfg.get("ssim_lambda", 0.2))
        rgb_mask_mode = str(transient_cfg.get("rgb_loss_mask", "none")).lower()
        if rgb_mask_mode in {"bg", "bg_mask"}:
            rgb_mask = torch.tile(bg_mask, (3, 1, 1)).detach()
            Ll1 = torch.abs(curr_data["im"] - im)[rgb_mask].mean()
        else:
            Ll1 = torch.abs(curr_data["im"] - im).mean()
        simloss = 1.0 - calc_ssim(im, curr_data["im"])
        losses["im"] = (1.0 - ssim_lambda) * Ll1 + ssim_lambda * simloss
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    if use_transient and isinstance(alpha_t, torch.Tensor):
        losses["alpha_transient"] = alpha_t.mean()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.ones((num_pts, 1), dtype=torch.float, device="cuda") * 0.5
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
    }
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, use_simplification=True):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)
    depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 20*depth_error.mean())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0) & (curr_data['depth'][0, :, :] < 1e10)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        valid_color_mask = energy_mask(curr_data['im']).squeeze()
        non_presence_mask = non_presence_mask & valid_color_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep),dim=0)
    return params, variables


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):
    # timer = Timer()
    # timer.start()
    
    # Print Config
    print("Loaded Config:")
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Dump the resolved runtime config to the run directory so eval can be config-aware.
    try:
        cfg_path = os.path.join(output_dir, "config_used.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(_make_jsonable(config), f, ensure_ascii=False, indent=2)
        print("[config] wrote:", cfg_path, flush=True)
    except Exception as e:
        print(f"[config] failed to write config_used.json: {type(e).__name__}: {e}", flush=True)

    # Get Device
    device = torch.device(config["primary_device"])

    tracking_backend = str(config.get("tracking", {}).get("backend", "mast3r")).lower()
    if tracking_backend != "mast3r":
        raise ValueError("EndoMD-SLAM release config supports the paper path: tracking.backend='mast3r'.")

    try:
        from utils.mast3r_tracking_adapter import MASt3RTrackerAdapter
    except Exception as e:
        raise RuntimeError(
            "tracking.backend='mast3r' requires MASt3R-SLAM dependencies (e.g. lietorch). "
            "Install MASt3R-SLAM following `MASt3R-SLAM/README.md` and ensure its checkpoints exist."
        ) from e

    mast3r_cfg = config.setdefault("mast3r_tracking", {})
    mast3r_config_path = mast3r_cfg.get("mast3r_config_path", None)
    if mast3r_config_path is not None and (not os.path.isabs(str(mast3r_config_path))):
        mast3r_config_path = os.path.join(_BASE_DIR, str(mast3r_config_path))

    transient_weighting_cfg = mast3r_cfg.get("transient_weighting", {})
    mast3r_adapter = MASt3RTrackerAdapter(
        device=device,
        mast3r_config_path=None if mast3r_config_path is None else os.path.expanduser(str(mast3r_config_path)),
        retrieval_k=int(mast3r_cfg.get("retrieval_k", 10)),
        retrieval_min_thresh=float(mast3r_cfg.get("retrieval_min_thresh", 1e-3)),
        reloc_min_match_frac=float(mast3r_cfg.get("reloc_min_match_frac", 0.15)),
        reloc_max_candidates=int(mast3r_cfg.get("reloc_max_candidates", 10)),
        metric_pose_min_points=int(mast3r_cfg.get("metric_pose_min_points", 500)),
        transient_weighting_enabled=bool(transient_weighting_cfg.get("enabled", True)),
        transient_weighting_mode=str(transient_weighting_cfg.get("mode", "prev_frame")),
        transient_weighting_gamma=float(transient_weighting_cfg.get("gamma", 2.0)),
        transient_weighting_min_multiplier=float(transient_weighting_cfg.get("min_multiplier", 0.0)),
        transient_weighting_blur_ksize=int(transient_weighting_cfg.get("blur_ksize", 11)),
        transient_weighting_blur_sigma=float(transient_weighting_cfg.get("blur_sigma", 3.0)),
    )

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if not bool(config.get("distance_keyframe_selection", True)):
        raise ValueError("EndoMD-SLAM release keeps the paper path: distance_keyframe_selection=True.")
    print("Using distance keyframe selection. Note that 'mapping window size' is unused.")
    if 'distance_current_frame_prob' not in config:
        config['distance_current_frame_prob'] = 0.1
    if 'gaussian_simplification' not in config:
        config['gaussian_simplification'] = True # simplified in paper
    if not config['gaussian_simplification']:
        print("Using Full Gaussian Representation, which may cause unstable optimization if not fully optimized.")
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "train_or_test" not in dataset_config:
        dataset_config["train_or_test"] = 'all'
    if "preload" not in dataset_config:
        dataset_config["preload"] = False
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        train_or_test=dataset_config["train_or_test"]
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    if dataset_config["train_or_test"] == 'train': # kind of ill implementation here. train_or_test should be 'all' or 'train'. If 'test', you view test set as full dataset.
        eval_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"], # if you eval, you should keep reso as raw image.
            desired_width=dataset_config["desired_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test='test'
        )
    params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(
        dataset,
        num_frames,
        config['scene_radius_depth_ratio'],
        config['mean_sq_dist_method'],
        use_simplification=config['gaussian_simplification'],
    )
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Wall-clock training time excluding initialization (frame 0 + any pre-loop setup).
    train_wall_start = None
    train_wall_start_frame = None
    train_wall_start_idx = 1
    # Transient decomposition (incremental; default disabled).
    transient_cfg = config.get("transient", None)
    if not (isinstance(transient_cfg, dict) and bool(transient_cfg.get("enabled", False))):
        transient_cfg = None
    else:
        params = _init_empty_transient_params(params)
    
    # timer.lap("all the config")
    
    # Iterate over Scan
    for time_idx in tqdm(range(num_frames)):

        if train_wall_start is None and time_idx == train_wall_start_idx:
            train_wall_start = time.perf_counter()
            train_wall_start_frame = int(time_idx)
        
        # timer.lap("iterating over frame "+str(time_idx), 0)
        
        print() # always show global iteration
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Tracking pose is estimated by MASt3R metric tracking.
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']

        pose_is_reliable = True
        mast3r_new_keyframe = False

        # timer.lap("initialized data", 1)

        # Tracking
        tracking_start_time = time.time()
        assert mast3r_adapter is not None

        tracking_color = tracking_curr_data["im"]
        tracking_depth = tracking_curr_data["depth"]

        rgb_hwc = tracking_color.permute(1, 2, 0).detach().cpu().numpy()
        rgb_hwc = np.clip(rgb_hwc, 0.0, 1.0).astype(np.float32, copy=False)

        out = mast3r_adapter.track(
            frame_id=int(time_idx),
            rgb_hwc_01=rgb_hwc,
            depth_chw=tracking_depth,
            intrinsics=tracking_curr_data.get("intrinsics", None),
        )
        pose_is_reliable = bool(out.pose_is_reliable)
        mast3r_new_keyframe = bool(out.new_keyframe)

        with torch.no_grad():
            w2c = out.w2c
            rot_quat = matrix_to_quaternion(w2c[:3, :3].unsqueeze(0))
            params["cam_unnorm_rots"][..., time_idx] = rot_quat
            params["cam_trans"][..., time_idx] = w2c[:3, 3].unsqueeze(0)
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        # timer.lap("tracking done", 2)

        # Densification & KeyFrame-based Mapping
        do_mapping_this_frame = (time_idx == 0) or ((time_idx + 1) % config["map_every"] == 0)
        allow_mapping = True
        transient_only_mapping = False
        allow_mapping = pose_is_reliable
        if (not allow_mapping) and transient_cfg is not None:
            um = transient_cfg.get("unreliable_pose_mapping", None)
            if isinstance(um, dict) and str(um.get("mode", "skip")).lower() in {"transient_only", "transient-only"}:
                allow_mapping = True
                transient_only_mapping = True

        if do_mapping_this_frame and allow_mapping:
            if transient_cfg is not None:
                params = _ensure_transient_for_frame(
                    params=params,
                    frame_idx=int(time_idx),
                    transient_cfg=transient_cfg,
                    init_color_chw=color,
                    init_depth_chw=depth,
                    init_intrinsics_3x3=intrinsics,
                )
            # Densification
            if (not transient_only_mapping) and config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, curr_data,
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], 
                                                      config['gaussian_simplification'])
                post_num_pts = params['means3D'].shape[0]
            
            # Reset Optimizer & Learning Rates
            if transient_only_mapping:
                optimizer = initialize_transient_optimizer(params, config["mapping"]["lrs"])
            else:
                optimizer = initialize_optimizer(params, config['mapping']['lrs']) 

            # timer.lap("Densification Done at frame "+str(time_idx), 3)

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
                
            actural_keyframe_ids = []
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                if transient_only_mapping:
                    iter_time_idx = int(time_idx)
                    iter_gt_w2c = gt_w2c_all_frames[: iter_time_idx + 1]
                    iter_data = {
                        "cam": cam,
                        "im": color,
                        "depth": depth,
                        "id": iter_time_idx,
                        "online_time_idx": int(time_idx),
                        "intrinsics": intrinsics,
                        "w2c": first_frame_w2c,
                        "iter_gt_w2c_list": iter_gt_w2c,
                        "freeze_static": True,
                    }
                    if transient_cfg is not None:
                        iter_data["transient"] = transient_cfg
                    loss, variables, losses = get_loss(
                        params,
                        iter_data,
                        variables,
                        iter_time_idx,
                        config["mapping"]["loss_weights"],
                        config["mapping"]["use_sil_for_loss"],
                        config["mapping"]["sil_thres"],
                        config["mapping"]["use_l1"],
                        config["mapping"]["ignore_outlier_depth_loss"],
                    )
                else:
                    if len(actural_keyframe_ids) == 0:
                        if len(keyframe_list) > 0:
                            curr_position = params['cam_trans'][..., time_idx].detach().cpu()
                            actural_keyframe_ids = keyframe_selection_distance(
                                time_idx,
                                curr_position,
                                keyframe_list,
                                config['distance_current_frame_prob'],
                                num_iters_mapping,
                            )
                        else:
                            actural_keyframe_ids = [0] * num_iters_mapping
                        print(
                            f"\nUsed Frames for mapping at Frame {time_idx}: "
                            f"{[keyframe_list[i]['id'] if i != len(keyframe_list) else 'curr' for i in actural_keyframe_ids]}"
                        )

                    selected_keyframe_ids = actural_keyframe_ids[iter]

                    if selected_keyframe_ids == len(keyframe_list):
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        # Use Keyframe Data
                        iter_time_idx = keyframe_list[selected_keyframe_ids]['id']
                        iter_color = keyframe_list[selected_keyframe_ids]['color']
                        iter_depth = keyframe_list[selected_keyframe_ids]['depth']
                    iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                    iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                                 'online_time_idx': int(time_idx),
                                 'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                    if transient_cfg is not None:
                        iter_data["transient"] = transient_cfg
                    # Loss for current frame
                    loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                                                    )
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if (not transient_only_mapping) and config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                    # Optimizer Update
                    optimizer.step()
                    # DeSplat-style transient refine (prune/densify via slot recycling; config-gated).
                    if transient_cfg is not None:
                        _transient_refine_inplace(
                            params=params,
                            variables=variables,
                            time_idx=int(time_idx),
                            transient_cfg=transient_cfg,
                            iter_idx=int(iter),
                        )
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                        mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                        mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    print('Failed to evaluate trajectory.')

            # Cache transient alpha from each mapped frame for transient-aware MASt3R tracking.
            if (
                mast3r_adapter is not None
                and transient_cfg is not None
                and pose_is_reliable
            ):
                transient_weighting_cfg = config.get("mast3r_tracking", {}).get("transient_weighting", None)
                if isinstance(transient_weighting_cfg, dict) and bool(transient_weighting_cfg.get("enabled", False)):
                    try:
                        with torch.no_grad():
                            alpha_hw = _render_transient_alpha_hw(
                                params=params,
                                frame_idx=int(time_idx),
                                cam=tracking_curr_data["cam"],
                                w2c=tracking_curr_data["w2c"],
                                transient_cfg=transient_cfg,
                            )
                        if isinstance(alpha_hw, torch.Tensor):
                            mast3r_adapter.update_last_frame_transient_alpha(alpha_hw)
                    except Exception as e:
                        print(
                            f"[transient_weighting] frame={time_idx} cache alpha failed: {type(e).__name__}: {e}"
                        )
        elif do_mapping_this_frame and (not allow_mapping):
            print(f"[Tracking] frame={time_idx} skip mapping (unreliable pose)")

        # timer.lap('Mapping Done.', 4)
        
        # Add frame to keyframe list
        should_add_kf = mast3r_new_keyframe

        if (
            should_add_kf
            and (not torch.isinf(curr_gt_w2c[-1]).any())
            and (not torch.isnan(curr_gt_w2c[-1]).any())
            and pose_is_reliable
        ):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)

        torch.cuda.empty_cache()

    # timer.end()
    train_wall_s_excl_init = None
    train_wall_end_frame = None
    if train_wall_start is not None:
        train_wall_s_excl_init = float(time.perf_counter() - train_wall_start)
        train_wall_end_frame = int(num_frames - 1)

    # Compute Average Runtimes
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print()
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    with open(os.path.join(output_dir, "runtimes.txt"), "w") as f:
        f.write(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s\n")
        f.write(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms\n")
        f.write(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s\n")
        f.write(f"Frame Time: {tracking_frame_time_avg + mapping_frame_time_avg} s\n")
        if train_wall_s_excl_init is not None and train_wall_end_frame is not None:
            f.write(
                f"Train Wall Time (frames {train_wall_start_frame}..{train_wall_end_frame}, excl init): {train_wall_s_excl_init} s\n"
            )
        else:
            f.write("Train Wall Time (excl init): n/a\n")

    with open(os.path.join(output_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_wall_s_excl_init": train_wall_s_excl_init,
                "train_wall_start_frame": train_wall_start_frame,
                "train_wall_end_frame": train_wall_end_frame,
                "num_frames": int(num_frames),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    
    # Evaluate Final Parameters
    dataset = [dataset, eval_dataset, 'C3VD'] if dataset_config["train_or_test"] == 'train' else dataset
    with torch.no_grad():
        eval_save(
            dataset,
            params,
            eval_dir,
            sil_thres=config['mapping']['sil_thres'],
            mapping_iters=config['mapping']['num_iters'],
            add_new_gaussians=config['mapping']['add_new_gaussians'],
            pose_eval=config.get("pose_eval", None),
            transient_cfg=transient_cfg,
        )

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)
    save_means3D(params['means3D'], output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    # Common command-line overrides for the release benchmark scripts.
    parser.add_argument("--sequence", type=str, default=None, help="Override dataset sequence name/path")
    parser.add_argument("--basedir", type=str, default=None, help="Override dataset base directory")
    parser.add_argument("--workdir", type=str, default=None, help="Override config['workdir'] output root")
    parser.add_argument("--run_name", type=str, default=None, help="Override run name/output folder")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Apply CLI overrides.
    if args.basedir is not None:
        experiment.config.setdefault("data", {})
        experiment.config["data"]["basedir"] = args.basedir
    if args.sequence is not None:
        experiment.config.setdefault("data", {})
        experiment.config["data"]["sequence"] = args.sequence
        if args.run_name is None:
            experiment.config["run_name"] = args.sequence
    if args.workdir is not None:
        experiment.config["workdir"] = args.workdir
    if args.run_name is not None:
        experiment.config["run_name"] = args.run_name

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)
    
    # Export a quick keyframe video from eval/plots.
    try:
        plot_video(os.path.join(results_dir, "eval", "plots"), os.path.join(results_dir, "keyframes"))
    except Exception as e:
        print(f"[Warning] Failed to export keyframe video: {e}")
