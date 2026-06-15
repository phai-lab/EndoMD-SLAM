from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
def _add_mast3r_slam_to_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    mast3r_root = repo_root / "MASt3R-SLAM"
    sys.path.insert(0, str(mast3r_root))
    return mast3r_root


_MAST3R_ROOT = _add_mast3r_slam_to_syspath()


import lietorch  # noqa: E402

from mast3r_slam.config import load_config as _mast3r_load_config  # noqa: E402
from mast3r_slam.config import config as mast3r_config  # noqa: E402
from mast3r_slam.frame import Frame, SharedKeyframes  # noqa: E402
from mast3r_slam.mast3r_utils import (  # noqa: E402
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
    resize_img,
)


@dataclass
class MASt3RTrackingOutput:
    w2c: torch.Tensor  # (4,4) float32 on device
    pose_is_reliable: bool
    new_keyframe: bool
    used_relocalization: bool


def _sim3_to_se3_matrix(T_WC: lietorch.Sim3) -> torch.Tensor:
    # Sim3 matrix is 4x4 with scaling baked into the rotation block.
    # We discard scale and keep rotation + translation only.
    M = T_WC.matrix()[0]  # (4,4)
    R = M[:3, :3]
    # Orthonormalize via SVD to remove scale/shear.
    # (More stable than blindly dividing by a scalar when scale is noisy.)
    U, _, Vh = torch.linalg.svd(R)
    R_se3 = U @ Vh
    if torch.linalg.det(R_se3) < 0:
        U[:, -1] *= -1
        R_se3 = U @ Vh
    t = M[:3, 3]
    out = torch.eye(4, device=M.device, dtype=M.dtype)
    out[:3, :3] = R_se3
    out[:3, 3] = t
    return out


class MASt3RTrackerAdapter:
    def __init__(
        self,
        *,
        device: torch.device,
        mast3r_config_path: Optional[Path] = None,
        weights_path: Optional[Path] = None,
        retriever_path: Optional[Path] = None,
        retrieval_k: int = 5,
        retrieval_min_thresh: float = 5e-3,
        reloc_min_match_frac: float = 0.3,
        reloc_max_candidates: int = 5,
        metric_pose_min_points: int = 500,
        img_size: int = 512,
        transient_weighting_enabled: bool = False,
        transient_weighting_mode: str = "keyframe",
        transient_weighting_gamma: float = 2.0,
        transient_weighting_min_multiplier: float = 0.0,
        transient_weighting_blur_ksize: int = 0,
        transient_weighting_blur_sigma: float = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.img_size = int(img_size)
        self.metric_pose_min_points = int(metric_pose_min_points)

        if mast3r_config_path is None:
            mast3r_config_path = _MAST3R_ROOT / "config" / "base.yaml"
        if weights_path is None:
            weights_path = (
                _MAST3R_ROOT
                / "checkpoints"
                / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
            )
        if retriever_path is None:
            retriever_path = (
                _MAST3R_ROOT
                / "checkpoints"
                / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
            )

        cfg_path = Path(mast3r_config_path)
        # MASt3R-SLAM config inheritance uses relative paths like "config/base.yaml".
        # Ensure it resolves by temporarily switching CWD to the MASt3R-SLAM repo root.
        old_cwd = os.getcwd()
        try:
            os.chdir(_MAST3R_ROOT)
            if cfg_path.is_absolute():
                try:
                    cfg_path = cfg_path.relative_to(_MAST3R_ROOT)
                except Exception:
                    pass
            _mast3r_load_config(str(cfg_path))
        finally:
            os.chdir(old_cwd)
        mast3r_config.setdefault("single_thread", True)

        self.retrieval_k = int(retrieval_k)
        self.retrieval_min_thresh = float(retrieval_min_thresh)
        self.reloc_min_match_frac = float(reloc_min_match_frac)
        self.reloc_max_candidates = int(reloc_max_candidates)

        self.model = load_mast3r(path=str(weights_path), device=str(self.device))
        self.model.eval()
        self.retrieval_database = load_retriever(
            self.model, retriever_path=str(retriever_path), device=str(self.device)
        )

        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = self.device
        retrieval_device = getattr(self.retrieval_database, "centroids", torch.empty(0)).device
        if torch.cuda.is_available() and str(model_device).startswith("cuda"):
            mem_mb = torch.cuda.memory_allocated(model_device) / (1024**2)
            print(
                f"[MASt3R] model_device={model_device} retrieval_device={retrieval_device} "
                f"cuda_mem_allocated={mem_mb:.1f}MB"
            )
        else:
            print(f"[MASt3R] model_device={model_device} retrieval_device={retrieval_device} (cuda unavailable)")

        self._manager = None
        self.keyframes: Optional[SharedKeyframes] = None
        self._last_T_WC: Optional[lietorch.Sim3] = None
        self._idx_f2k_init: Optional[torch.Tensor] = None
        self._K_mast3r: Optional[torch.Tensor] = None
        self._u_flat: Optional[torch.Tensor] = None
        self._v_flat: Optional[torch.Tensor] = None
        self._kf_depths: list[torch.Tensor] = []
        self._debug_current_dir: Optional[Path] = None
        self._last_frame: Optional[Frame] = None
        self._last_meta: Optional[dict] = None

        # Optional: transient-aware tracking (Method A: soft downweighting).
        self.transient_weighting_enabled = bool(transient_weighting_enabled)
        self.transient_weighting_mode = str(transient_weighting_mode).lower().strip()
        if self.transient_weighting_mode not in {"keyframe", "prev_frame", "prev", "previous"}:
            raise ValueError(
                "mast3r transient_weighting.mode must be one of: 'keyframe', 'prev_frame'"
            )
        self.transient_weighting_gamma = float(transient_weighting_gamma)
        self.transient_weighting_min_multiplier = float(transient_weighting_min_multiplier)
        self.transient_weighting_blur_ksize = int(transient_weighting_blur_ksize)
        self.transient_weighting_blur_sigma = float(transient_weighting_blur_sigma)
        self._transient_alpha_by_frame_id: dict[int, torch.Tensor] = {}

    def _ensure_pixel_grid(self, *, h: int, w: int) -> None:
        if self._u_flat is not None and self._v_flat is not None:
            if int(self._u_flat.numel()) == int(h * w):
                return
        u = torch.arange(int(w), device=self.device, dtype=torch.float32).repeat(int(h))
        v = torch.arange(int(h), device=self.device, dtype=torch.float32).repeat_interleave(int(w))
        self._u_flat = u
        self._v_flat = v

    def _compute_mast3r_intrinsics(self, intrinsics: Any, *, meta: dict) -> torch.Tensor:
        """
        Convert original intrinsics (for the provided rgb/depth) into the intrinsics of the
        MASt3R cropped + downsampled frame coordinate system.
        """
        if intrinsics is None:
            raise ValueError("intrinsics is required for calibrated/metric pose estimation")

        if isinstance(intrinsics, torch.Tensor):
            K0 = intrinsics.detach().to(device=self.device, dtype=torch.float32)
        else:
            K0 = torch.tensor(np.asarray(intrinsics), device=self.device, dtype=torch.float32)

        if K0.shape != (3, 3):
            raise ValueError(f"Expected intrinsics shape (3,3), got {tuple(K0.shape)}")

        scale_w, scale_h = meta.get("scale_wh", (None, None))
        if scale_w is None or scale_h is None:
            raise ValueError("meta missing scale_wh; cannot compute calibrated intrinsics")

        # resize_img reports scale_w = W_orig / W_resized (i.e. inverse resize factor).
        r_w = 1.0 / float(scale_w)
        r_h = 1.0 / float(scale_h)

        off_h, off_w = meta["crop_offset_hw"]  # in resized pixels
        downsample = int(meta["downsample"])

        K = K0.clone()
        K[0, :] = K[0, :] * r_w
        K[1, :] = K[1, :] * r_h
        K[0, 2] = K[0, 2] - float(off_w)
        K[1, 2] = K[1, 2] - float(off_h)
        if downsample > 1:
            K[0, :] = K[0, :] / float(downsample)
            K[1, :] = K[1, :] / float(downsample)
        return K

    def _set_intrinsics_if_needed(self, intrinsics: Any, *, meta: dict) -> None:
        if not bool(mast3r_config.get("use_calib", False)):
            return
        if self._K_mast3r is None:
            self._K_mast3r = self._compute_mast3r_intrinsics(intrinsics, meta=meta)
        if self.keyframes is not None:
            self.keyframes.set_intrinsics(self._K_mast3r)

    def _backproject_depth(self, depth_hw: torch.Tensor, *, K: torch.Tensor) -> torch.Tensor:
        if depth_hw.dim() != 2:
            raise ValueError(f"Expected depth shape (H,W), got {tuple(depth_hw.shape)}")
        h, w = int(depth_hw.shape[0]), int(depth_hw.shape[1])
        self._ensure_pixel_grid(h=h, w=w)
        assert self._u_flat is not None and self._v_flat is not None

        z = depth_hw.reshape(-1).to(dtype=torch.float32)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        x = (self._u_flat - cx) / fx * z
        y = (self._v_flat - cy) / fy * z
        return torch.stack([x, y, z], dim=-1)  # (N,3)

    def _rotmat_to_quat_xyzw(self, R: torch.Tensor) -> torch.Tensor:
        # Reuse EndoGSLAM's stable conversion (returns wxyz).
        from utils.slam_helpers import matrix_to_quaternion

        q_wxyz = matrix_to_quaternion(R.unsqueeze(0))[0]  # (4,)
        # Sim3 expects xyzw (w is last).
        return torch.stack([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dim=0)

    def _sim3_from_se3_rt(self, R: torch.Tensor, t: torch.Tensor) -> lietorch.Sim3:
        quat_xyzw = self._rotmat_to_quat_xyzw(R)
        s = torch.ones(1, device=self.device, dtype=torch.float32)
        data = torch.cat([t.to(dtype=torch.float32), quat_xyzw.to(dtype=torch.float32), s], dim=0).view(1, -1)
        return lietorch.Sim3(data)

    def _solve_se3_kabsch(
        self, *, A: torch.Tensor, B: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve min sum_i w_i ||B_i - (R A_i + t)||^2  for R in SO(3), t in R^3.
        A, B: (M,3), w: (M,)
        """
        if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[1] != 3:
            raise ValueError(f"Invalid A/B shapes: A={tuple(A.shape)} B={tuple(B.shape)}")
        w = w.reshape(-1).to(dtype=torch.float32)
        if w.numel() != A.shape[0]:
            raise ValueError(f"Invalid weight shape: w={tuple(w.shape)} A={tuple(A.shape)}")

        w = torch.clamp(w, min=1e-6)
        W = w.sum()
        mu_A = (A * w[:, None]).sum(dim=0) / W
        mu_B = (B * w[:, None]).sum(dim=0) / W
        A0 = A - mu_A
        B0 = B - mu_B
        H = (A0 * w[:, None]).T @ B0  # (3,3)
        U, _S, Vh = torch.linalg.svd(H)
        R = Vh.T @ U.T
        if torch.linalg.det(R) < 0:
            Vh[-1, :] *= -1.0
            R = Vh.T @ U.T
        t = mu_B - R @ mu_A
        return R, t

    def _make_frame(
        self, frame_id: int, rgb_hwc_01: np.ndarray, *, T_WC_init: lietorch.Sim3
    ) -> tuple[Frame, dict]:
        res, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
            rgb_hwc_01, self.img_size, return_transformation=True
        )

        rgb = res["img"].to(device=self.device)
        img_true_shape = torch.tensor(res["true_shape"], device=self.device)
        img_shape = img_true_shape.clone()

        # Keep unnormalized img on CPU (used only for viz / export in MASt3R-SLAM).
        uimg = torch.from_numpy(res["unnormalized_img"].copy()).to(dtype=torch.float32) / 255.0

        downsample = int(mast3r_config.get("dataset", {}).get("img_downsample", 1))
        if downsample > 1:
            uimg = uimg[::downsample, ::downsample]
            img_shape = img_shape // downsample

        frame = Frame(
            int(frame_id),
            rgb,
            img_shape,
            img_true_shape,
            uimg,
            T_WC_init,
        )

        crop_h, crop_w = int(img_true_shape[0, 0]), int(img_true_shape[0, 1])
        resized_h = int(round(crop_h + 2 * float(half_crop_h)))
        resized_w = int(round(crop_w + 2 * float(half_crop_w)))

        meta = {
            "orig_hw": (int(rgb_hwc_01.shape[0]), int(rgb_hwc_01.shape[1])),
            "resized_hw": (resized_h, resized_w),
            "crop_hw": (crop_h, crop_w),
            "crop_offset_hw": (float(half_crop_h), float(half_crop_w)),
            "scale_wh": (float(scale_w), float(scale_h)),
            "downsample": downsample,
        }
        return frame, meta

    def _process_depth_to_frame(
        self, depth_chw: torch.Tensor, *, meta: dict
    ) -> Optional[torch.Tensor]:
        if depth_chw is None:
            return None
        if depth_chw.dim() != 3 or depth_chw.shape[0] != 1:
            raise ValueError(f"Expected depth shape (1,H,W), got {tuple(depth_chw.shape)}")

        depth = depth_chw.to(device=self.device, dtype=torch.float32, non_blocking=True)
        depth4 = depth.unsqueeze(0)  # (1,1,H,W)

        H_resize, W_resize = meta["resized_hw"]
        depth4 = F.interpolate(depth4, size=(int(H_resize), int(W_resize)), mode="bilinear", align_corners=False)

        crop_h, crop_w = meta["crop_hw"]
        off_h, off_w = meta["crop_offset_hw"]
        top = int(round(off_h))
        left = int(round(off_w))
        depth4 = depth4[:, :, top : top + int(crop_h), left : left + int(crop_w)]

        downsample = int(meta["downsample"])
        if downsample > 1:
            depth4 = depth4[:, :, ::downsample, ::downsample]

        return depth4[0, 0]  # (H,W)

    def _process_alpha_to_frame(self, alpha_hw: Any, *, meta: dict) -> Optional[torch.Tensor]:
        if alpha_hw is None:
            return None

        if isinstance(alpha_hw, np.ndarray):
            a = torch.from_numpy(alpha_hw)
        elif isinstance(alpha_hw, torch.Tensor):
            a = alpha_hw
        else:
            a = torch.tensor(np.asarray(alpha_hw))

        if a.dim() != 2:
            raise ValueError(f"Expected alpha shape (H,W), got {tuple(a.shape)}")

        a = a.to(device=self.device, dtype=torch.float32, non_blocking=True)
        a4 = a.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        H_resize, W_resize = meta["resized_hw"]
        a4 = F.interpolate(a4, size=(int(H_resize), int(W_resize)), mode="bilinear", align_corners=False)

        crop_h, crop_w = meta["crop_hw"]
        off_h, off_w = meta["crop_offset_hw"]
        top = int(round(off_h))
        left = int(round(off_w))
        a4 = a4[:, :, top : top + int(crop_h), left : left + int(crop_w)]

        downsample = int(meta["downsample"])
        if downsample > 1:
            a4 = a4[:, :, ::downsample, ::downsample]

        return a4[0, 0].clamp(0.0, 1.0)  # (H,W)

    def update_last_frame_transient_alpha(self, alpha_hw_01: Any) -> None:
        if not self.transient_weighting_enabled:
            return
        if self._last_frame is None or self._last_meta is None:
            return

        alpha_frame_hw = self._process_alpha_to_frame(alpha_hw_01, meta=self._last_meta)
        if alpha_frame_hw is None:
            return

        # Optional blur (expand transient regions; "sticky to lens" prior).
        k = int(self.transient_weighting_blur_ksize)
        sigma = float(self.transient_weighting_blur_sigma)
        if k > 0:
            if k % 2 == 0:
                k += 1
            if sigma <= 0.0:
                sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8  # OpenCV-like heuristic
            try:
                # 1D gaussian kernel.
                x = torch.arange(k, device=alpha_frame_hw.device, dtype=torch.float32) - (k - 1) / 2.0
                g = torch.exp(-0.5 * (x / sigma) ** 2)
                g = (g / g.sum()).view(1, 1, -1)  # (1,1,k)

                a = alpha_frame_hw.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                pad = k // 2
                a = F.pad(a, (pad, pad, 0, 0), mode="replicate")
                a = F.conv2d(a, g.unsqueeze(-2))  # horizontal
                a = F.pad(a, (0, 0, pad, pad), mode="replicate")
                a = F.conv2d(a, g.unsqueeze(-1))  # vertical
                alpha_frame_hw = a[0, 0].clamp(0.0, 1.0)
            except Exception:
                pass

        fid = int(self._last_frame.frame_id)
        self._transient_alpha_by_frame_id[fid] = alpha_frame_hw.detach()

        # Keep only alphas for frames that are still keyframes to bound memory.
        if self.keyframes is not None:
            try:
                keep = {int(self.keyframes[i].frame_id) for i in range(len(self.keyframes))}
                for k in list(self._transient_alpha_by_frame_id.keys()):
                    if k not in keep:
                        del self._transient_alpha_by_frame_id[k]
            except Exception:
                pass

    def _apply_transient_weighting(
        self,
        *,
        frame: Frame,
        keyframe: Frame,
        idx_f2k: torch.Tensor,
        mask_kf: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        if not self.transient_weighting_enabled:
            return w
        try:
            mode = self.transient_weighting_mode
            if mode in {"prev_frame", "prev", "previous"}:
                prev_id = int(frame.frame_id) - 1
                alpha_hw = self._transient_alpha_by_frame_id.get(prev_id, None)
                if alpha_hw is None or alpha_hw.dim() != 2:
                    return w
                if int(alpha_hw.numel()) != int(idx_f2k.numel()):
                    # idx_f2k is defined over all keyframe pixels (N = H*W).
                    return w
                idx_corr = idx_f2k.reshape(-1)[mask_kf].to(dtype=torch.long)
                alpha = alpha_hw.reshape(-1)[idx_corr].to(dtype=torch.float32)
            else:
                alpha_hw = self._transient_alpha_by_frame_id.get(int(keyframe.frame_id), None)
                if alpha_hw is None or alpha_hw.dim() != 2:
                    return w
                if int(alpha_hw.numel()) != int(mask_kf.numel()):
                    return w
                alpha = alpha_hw.reshape(-1)[mask_kf].to(dtype=torch.float32)

            gamma = float(self.transient_weighting_gamma)
            mult = (1.0 - alpha).clamp(0.0, 1.0).pow(gamma)
            floor = float(self.transient_weighting_min_multiplier)
            if floor > 0.0:
                mult = torch.clamp(mult, min=floor)
            return (w.to(dtype=torch.float32) * mult).to(dtype=w.dtype)
        except Exception:
            return w

    def _estimate_metric_pose_against_keyframe(
        self,
        *,
        frame: Frame,
        keyframe: Frame,
        depth_frame_hw: torch.Tensor,
        depth_keyframe_hw: torch.Tensor,
        idx_init: Optional[torch.Tensor],
        min_match_frac: Optional[float] = None,
    ) -> dict:
        """
        Metric pose estimation (B2):
        - Use MASt3R descriptor correspondences (idx_f2k).
        - Use GT depth (+ intrinsics) to backproject to metric 3D.
        - Solve SE3 with weighted Kabsch: min ||Xk - (R Xf + t)||.
        Returns dict with pose_ok / try_reloc / new_kf and optional debug tensors.
        """
        from mast3r_slam.mast3r_utils import mast3r_match_asymmetric

        cfg = mast3r_config["tracking"]
        thr = float(cfg["min_match_frac"]) if min_match_frac is None else float(min_match_frac)

        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=idx_init
        )
        # Save idx for next frame (sequential tracking).
        idx_f2k_init = idx_f2k.clone()

        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]  # (N,1)

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)  # (N,1)
        frame.update_pointmap(Xff, Cff)

        Cf = Cff[idx_f2k]
        Ck = keyframe.get_average_conf()
        valid_Cf = Cf > cfg["C_conf"]
        valid_Ck = Ck > cfg["C_conf"]
        valid_Q = Qk > cfg["Q_conf"]

        depth_f = depth_frame_hw.reshape(-1)
        depth_k = depth_keyframe_hw.reshape(-1)
        depth_f_corr = depth_f[idx_f2k]
        valid_depth = (
            torch.isfinite(depth_f_corr)
            & torch.isfinite(depth_k)
            & (depth_f_corr > 0)
            & (depth_k > 0)
        )

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q & valid_depth.unsqueeze(-1)

        match_frac = (valid_opt.sum() / valid_opt.numel()).item()
        n_valid = int(valid_opt.sum().item())
        if match_frac < thr or n_valid < self.metric_pose_min_points:
            return {
                "pose_ok": False,
                "try_reloc": True,
                "new_kf": False,
                "match_frac": float(match_frac),
                "thr": float(thr),
                "idx_f2k_init": idx_f2k_init,
            }

        if self._K_mast3r is None:
            raise RuntimeError("Metric pose requires intrinsics; _K_mast3r is None")

        pts_f = self._backproject_depth(depth_frame_hw, K=self._K_mast3r)
        pts_k = self._backproject_depth(depth_keyframe_hw, K=self._K_mast3r)
        pts_f_corr = pts_f[idx_f2k]

        mask = valid_opt[:, 0]
        A = pts_f_corr[mask]
        B = pts_k[mask]
        w = Qk[mask][:, 0]
        w = self._apply_transient_weighting(frame=frame, keyframe=keyframe, idx_f2k=idx_f2k, mask_kf=mask, w=w)

        try:
            R, t = self._solve_se3_kabsch(A=A, B=B, w=w)
        except Exception as e:
            return {
                "pose_ok": False,
                "try_reloc": True,
                "new_kf": False,
                "reason": "kabsch_failed",
                "exception": f"{type(e).__name__}: {e}",
                "match_frac": float(match_frac),
                "thr": float(thr),
                "idx_f2k_init": idx_f2k_init,
            }

        T_CkCf = self._sim3_from_se3_rt(R, t)
        frame.T_WC = keyframe.T_WC * T_CkCf

        # Keyframe selection (same heuristic as MASt3R tracker).
        valid_kf = valid_match_k & valid_Q
        match_frac_k = float((valid_kf.sum() / valid_kf.numel()).item())
        unique_frac_f = float(torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0]) / float(valid_kf.numel())
        new_kf = min(match_frac_k, unique_frac_f) < float(cfg["match_frac_thresh"])

        if new_kf:
            idx_f2k_init = None

        # Update keyframe pointmap using estimated relative pose (optional but matches upstream behavior).
        Xkk = T_CkCf.act(Xkf)
        return {
            "pose_ok": True,
            "try_reloc": False,
            "new_kf": bool(new_kf),
            "match_frac": float(match_frac),
            "thr": float(thr),
            "idx_f2k_init": idx_f2k_init,
            "Xkk": Xkk,
            "Ckf": Ckf,
        }

    def _try_relocalize_metric(
        self, frame: Frame, depth_frame_hw: Optional[torch.Tensor]
    ) -> tuple[bool, list[int], list[dict]]:
        assert self.keyframes is not None

        if depth_frame_hw is None:
            return False, [], []

        retrieval_inds = self.retrieval_database.update(
            frame,
            add_after_query=False,
            k=self.retrieval_k,
            min_thresh=self.retrieval_min_thresh,
        )
        if not retrieval_inds:
            print(f"[MASt3R][Reloc] frame={frame.frame_id} no retrieval candidates")
            return False, [], []

        cand_kf_idx = list(retrieval_inds)[: self.reloc_max_candidates]
        best_T_WC = None
        best_kf_idx = None
        best_match_frac = -1.0
        debug_results: list[dict] = []

        for kf_idx in cand_kf_idx:
            kf_idx_int = int(kf_idx)
            try:
                keyframe = self.keyframes[kf_idx_int]
                depth_kf_hw = self._kf_depths[kf_idx_int]
            except Exception as e:
                debug_results.append(
                    {
                        "kf_idx": kf_idx_int,
                        "ok": False,
                        "reason": "missing_keyframe_depth",
                        "exception": f"{type(e).__name__}: {e}",
                    }
                )
                continue

            out = self._estimate_metric_pose_against_keyframe(
                frame=frame,
                keyframe=keyframe,
                depth_frame_hw=depth_frame_hw,
                depth_keyframe_hw=depth_kf_hw,
                idx_init=None,
                min_match_frac=self.reloc_min_match_frac,
            )
            ok = bool(out.get("pose_ok", False)) and (not bool(out.get("try_reloc", False)))
            debug_results.append(
                {
                    "kf_idx": kf_idx_int,
                    "ok": ok,
                    "reason": out.get("reason"),
                    "match_frac": out.get("match_frac"),
                    "thr": out.get("thr"),
                    "exception": out.get("exception"),
                }
            )
            if not ok:
                continue
            if float(out["match_frac"]) > best_match_frac:
                best_match_frac = float(out["match_frac"])
                best_T_WC = lietorch.Sim3(frame.T_WC.data.clone())
                best_kf_idx = kf_idx_int

        if best_T_WC is None or best_kf_idx is None:
            print(f"[MASt3R][Reloc] frame={frame.frame_id} failed candidates={cand_kf_idx}")
            return False, [int(x) for x in cand_kf_idx], debug_results

        frame.T_WC = best_T_WC
        print(
            f"[MASt3R][Reloc] frame={frame.frame_id} success kf={best_kf_idx} match_frac={best_match_frac:.3f}"
        )
        return True, [int(x) for x in cand_kf_idx], debug_results

    def initialize(
        self,
        *,
        frame_id: int,
        rgb_hwc_01: np.ndarray,
        depth_chw: torch.Tensor,
        intrinsics: Any = None,
    ) -> MASt3RTrackingOutput:
        import torch.multiprocessing as mp

        if self.keyframes is not None:
            raise RuntimeError("MASt3RTrackerAdapter is already initialized")

        T0 = lietorch.Sim3.Identity(1, device=self.device)
        frame0, meta0 = self._make_frame(frame_id, rgb_hwc_01, T_WC_init=T0)
        self._last_frame = frame0
        self._last_meta = meta0

        depth0 = self._process_depth_to_frame(depth_chw, meta=meta0)
        if depth0 is None:
            raise ValueError("depth_chw is required for EndoMD-SLAM metric tracking")

        # Set calibrated intrinsics (if enabled in MASt3R config).
        self._set_intrinsics_if_needed(intrinsics, meta=meta0)

        X0, C0 = mast3r_inference_mono(self.model, frame0)
        frame0.update_pointmap(X0, C0)

        h, w = frame0.img_shape.flatten().tolist()
        self._manager = mp.Manager()
        self.keyframes = SharedKeyframes(self._manager, int(h), int(w), device=str(self.device))
        self.keyframes.append(frame0)
        self._kf_depths = [depth0.detach()] if depth0 is not None else []

        # If intrinsics were computed before keyframes existed, write them now.
        if bool(mast3r_config.get("use_calib", False)) and self._K_mast3r is not None:
            self.keyframes.set_intrinsics(self._K_mast3r)

        # Add first keyframe to retrieval DB (keeps DB indices aligned with keyframe indices).
        self.retrieval_database.update(
            frame0,
            add_after_query=True,
            k=self.retrieval_k,
            min_thresh=self.retrieval_min_thresh,
        )

        self._last_T_WC = frame0.T_WC
        self._idx_f2k_init = None

        T_WC_se3 = _sim3_to_se3_matrix(frame0.T_WC)
        w2c = torch.linalg.inv(T_WC_se3)
        return MASt3RTrackingOutput(
            w2c=w2c,
            pose_is_reliable=True,
            new_keyframe=True,
            used_relocalization=False,
        )

    def track(
        self,
        *,
        frame_id: int,
        rgb_hwc_01: np.ndarray,
        depth_chw: torch.Tensor,
        intrinsics: Any = None,
    ) -> MASt3RTrackingOutput:
        if self.keyframes is None or self._last_T_WC is None:
            return self.initialize(
                frame_id=frame_id,
                rgb_hwc_01=rgb_hwc_01,
                depth_chw=depth_chw,
                intrinsics=intrinsics,
            )

        frame, meta = self._make_frame(frame_id, rgb_hwc_01, T_WC_init=self._last_T_WC)
        depth_frame_hw = self._process_depth_to_frame(depth_chw, meta=meta)
        self._last_frame = frame
        self._last_meta = meta

        # Update intrinsics if calibrated mode is enabled.
        self._set_intrinsics_if_needed(intrinsics, meta=meta)

        used_reloc = False
        pose_ok = True
        new_kf = False
        try_reloc = False

        if depth_frame_hw is None:
            raise ValueError("depth_chw is required for EndoMD-SLAM metric tracking")
        if self._K_mast3r is None:
            self._K_mast3r = self._compute_mast3r_intrinsics(intrinsics, meta=meta)
            if bool(mast3r_config.get("use_calib", False)) and self.keyframes is not None:
                self.keyframes.set_intrinsics(self._K_mast3r)

        keyframe = self.keyframes.last_keyframe()
        if keyframe is None:
            raise RuntimeError("No keyframe available")
        kf_idx = len(self.keyframes) - 1
        if kf_idx >= len(self._kf_depths):
            raise RuntimeError("Keyframe depth cache out of sync")
        depth_kf_hw = self._kf_depths[kf_idx]

        out = self._estimate_metric_pose_against_keyframe(
            frame=frame,
            keyframe=keyframe,
            depth_frame_hw=depth_frame_hw,
            depth_keyframe_hw=depth_kf_hw,
            idx_init=self._idx_f2k_init,
        )
        pose_ok = bool(out["pose_ok"])
        new_kf = bool(out["new_kf"])
        try_reloc = bool(out["try_reloc"])
        if pose_ok:
            self._idx_f2k_init = out.get("idx_f2k_init")
            try:
                keyframe.update_pointmap(out["Xkk"], out["Ckf"])
                self.keyframes[kf_idx] = keyframe
            except Exception:
                pass

        if try_reloc:
            # Need a pointmap for retrieval + re-estimation.
            X, C = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X, C)
            print(f"[MASt3R][Reloc] frame={frame.frame_id} triggered (tracking failed)")

            pose_ok, _candidates, _candidate_results = self._try_relocalize_metric(frame, depth_frame_hw)
            used_reloc = True
            new_kf = pose_ok  # successful relocalization -> treat as keyframe

        if pose_ok:
            self._last_T_WC = frame.T_WC

            if new_kf:
                assert self.keyframes is not None
                self.keyframes.append(frame)
                if depth_frame_hw is not None:
                    self._kf_depths.append(depth_frame_hw.detach())
                # New keyframe -> reset correspondence initialization.
                self._idx_f2k_init = None
                self.retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=self.retrieval_k,
                    min_thresh=self.retrieval_min_thresh,
                )
        T_WC_se3 = _sim3_to_se3_matrix(frame.T_WC)
        w2c = torch.linalg.inv(T_WC_se3)

        return MASt3RTrackingOutput(
            w2c=w2c,
            pose_is_reliable=bool(pose_ok),
            new_keyframe=bool(new_kf and pose_ok),
            used_relocalization=bool(used_reloc),
        )
