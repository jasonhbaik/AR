"""
Shared AR anchor node + MediaPipe gesture controls.

Both ar_anchor.py (VIO only) and ar_anchor_slam.py (VIO + SLAM) import
ARAnchorNode from here. They differ only in how they wire the upstream
pipeline; the AR rendering, gesture state machine, and hand interaction
logic are identical.

Multiple "windows" (anchored pictures) can coexist. Each is a 4-corner
quad in world space, hit-tested independently. Pinching inside an
existing window grabs that window; pinching in empty space drops a new
one after a short hold.

Gesture controls (no keyboard):
    pinch + hold (~0.4 s) in empty space   drop a new window 1.5 m ahead
    pinch inside a window                  drag that window
    pinch near a corner                    drag just that corner
    pinch with both hands inside a window  scale that window around the midpoint
    both palms open (~1 s)                 clear ALL windows
    close the OpenCV window                quit
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import depthai as dai

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as e:
    raise SystemExit(
        "[ar_anchor] mediapipe is required for hand tracking.\n"
        "Install with: .venv/Scripts/pip install mediapipe"
    ) from e


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = "hand_landmarker.task"

# Pinch threshold in normalized image coords (thumb tip <-> index tip distance).
PINCH_THRESH_NORM = 0.06
# Snap-to-corner radius in pixels when picking what a pinch is grabbing.
CORNER_GRAB_RADIUS_PX = 35.0
# Hold time required to drop a fresh anchor with a pinch, in seconds.
ANCHOR_HOLD_SEC = 0.4
# Hold time required for both-palms-open to clear all anchors, in seconds.
CLEAR_HOLD_SEC = 1.0

# Skeleton connections for the 21-landmark hand model.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

NEAR_Z = 0.05  # near plane in meters; corners closer than this get clipped


def _ensure_hand_model(path=HAND_MODEL_PATH):
    if os.path.exists(path):
        return path
    print(f"[ar_anchor] downloading hand landmark model -> {path}")
    urllib.request.urlretrieve(HAND_MODEL_URL, path)
    return path


def quat_to_rotmat(qx, qy, qz, qw):
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw),     s * (qx * qz + qy * qw)],
        [s * (qx * qy + qz * qw),     1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
        [s * (qx * qz - qy * qw),     s * (qy * qz + qx * qw),     1 - s * (qx * qx + qy * qy)],
    ])


# Reusable FLU(camera) -> CV(right,down,forward) basis change as a matrix:
#   x_cv = -y_flu,  y_cv = -z_flu,  z_cv = x_flu
FLU_TO_CV = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
])


class HandTracker:
    """Wraps MediaPipe HandLandmarker in VIDEO running mode for streaming use.

    Inference runs on a downscaled copy of the input frame; the returned
    landmark coordinates are normalized [0, 1], so consumers that multiply
    by full-resolution image dims still get correct pixel coordinates."""
    def __init__(self, model_path, infer_scale=0.5):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.infer_scale = float(infer_scale)
        self._t0 = time.monotonic()

    def detect(self, bgr_img):
        if self.infer_scale != 1.0:
            h, w = bgr_img.shape[:2]
            bgr_img = cv2.resize(
                bgr_img,
                (max(1, int(w * self.infer_scale)), max(1, int(h * self.infer_scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.monotonic() - self._t0) * 1000)
        return self.landmarker.detect_for_video(mp_image, ts_ms)


def _pinch_state(landmarks, w, h):
    """(landmarks, image w/h) -> ((u,v) midpoint in pixels, normalized pinch dist)."""
    thumb = landmarks[4]
    index = landmarks[8]
    d_norm = float(np.hypot(thumb.x - index.x, thumb.y - index.y))
    u = float((thumb.x + index.x) * 0.5 * w)
    v = float((thumb.y + index.y) * 0.5 * h)
    return (u, v), d_norm


def _is_open_palm(landmarks):
    """Heuristic: the four non-thumb fingertips are all extended (far from wrist
    relative to the palm size). Robust enough for clear-the-anchor detection
    without a separate gesture model."""
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    mcp = np.array([landmarks[9].x, landmarks[9].y])
    ref = float(np.linalg.norm(mcp - wrist)) + 1e-6
    for tip in (8, 12, 16, 20):  # index, middle, ring, pinky tips
        p = np.array([landmarks[tip].x, landmarks[tip].y])
        if np.linalg.norm(p - wrist) / ref < 1.7:
            return False
    return True


class ARAnchorNode(dai.node.ThreadedHostNode):
    def __init__(self, overlay_path="overlay.png", anchor_distance=1.5,
                 anchor_half_height=0.25, window_title="AR anchor"):
        dai.node.ThreadedHostNode.__init__(self)
        self.inputTrans = dai.Node.Input(self)
        self.inputImg = dai.Node.Input(self)
        # Non-blocking + maxSize=1: always process the freshest pose/frame
        # instead of draining a backlog when MediaPipe / the host loop runs
        # slower than the device. Trades dropped frames for low latency.
        for _q in (self.inputTrans, self.inputImg):
            _q.setBlocking(False)
            _q.setMaxSize(1)

        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            raise FileNotFoundError(f"Could not load {overlay_path}")
        self.overlay = overlay
        self.anchor_distance = anchor_distance
        self.anchor_half_height = anchor_half_height
        self.window_title = window_title

        self.K = None
        # List of (4, 3) world-FLU corner arrays; one entry per AR window.
        # Drawn in list order, so later anchors render on top.
        self.anchors = []
        self._frame_count = 0
        self._prev_t_wc = None
        self._motion_accum = 0.0  # cumulative |Δt_wc| since startup

        # Drag/scale interaction state. drag/scale carry the index into
        # self.anchors of the window they're operating on.
        self.interaction_mode = "idle"  # "idle" | "drag" | "scale"
        self.drag = None
        self.scale = None

        # Global gesture timing state
        self._anchor_pinch_start = None
        self._open_palm_start = None

        # Latency stats.
        self._loop_t_prev = None
        self._fps = 0.0
        self._loop_ms = 0.0

    def _read_intrinsics(self, imgFrame):
        p = self.getParentPipeline()
        calib = p.getDefaultDevice().readCalibration()
        intr = calib.getCameraIntrinsics(
            dai.CameraBoardSocket(imgFrame.getInstanceNum()),
            imgFrame.getWidth(), imgFrame.getHeight(),
        )
        self.K = np.array(intr, dtype=np.float64)

    def _build_anchor_corners(self, t_wc, R_wc):
        oh, ow = self.overlay.shape[:2]
        half_h = self.anchor_half_height
        half_w = half_h * (ow / oh)
        d = self.anchor_distance
        corners_cam = np.array([
            [d,  half_w,  half_h],
            [d, -half_w,  half_h],
            [d, -half_w, -half_h],
            [d,  half_w, -half_h],
        ])
        return (R_wc @ corners_cam.T).T + t_wc

    def _to_camera_cv(self, P_w, t_wc, R_wc):
        # Batched equivalent of `R_wc.T @ (p - t_wc)` for each row of P_w.
        p_flu = (P_w - t_wc) @ R_wc
        return np.column_stack((-p_flu[:, 1], -p_flu[:, 2], p_flu[:, 0]))

    def _project_cv(self, P_c_cv):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        inv_z = 1.0 / P_c_cv[:, 2]
        return np.column_stack((fx * P_c_cv[:, 0] * inv_z + cx,
                                fy * P_c_cv[:, 1] * inv_z + cy))

    def _unproject_world(self, uv, depth_cv, t_wc, R_wc):
        """Pixel (u,v) at camera-CV depth z -> world-FLU 3D point."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x_cv = (uv[0] - cx) * depth_cv / fx
        y_cv = (uv[1] - cy) * depth_cv / fy
        # CV (right, down, forward) -> FLU camera (forward, left, up)
        p_flu_cam = np.array([depth_cv, -x_cv, -y_cv])
        return R_wc @ p_flu_cam + t_wc

    def _hit_test_anchors(self, uv, t_wc, R_wc):
        """Return (anchor_idx, kind, corner_idx, depth_cv) for the topmost
        window the pinch is grabbing, or None if it's in empty space.
        Topmost = highest list index, since later anchors render on top."""
        for i in reversed(range(len(self.anchors))):
            corners_world = self.anchors[i]
            P_c_cv = self._to_camera_cv(corners_world, t_wc, R_wc)
            if not np.all(P_c_cv[:, 2] > NEAR_Z):
                continue
            quad_uvs = self._project_cv(P_c_cv)
            dists = np.linalg.norm(quad_uvs - np.array(uv), axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] < CORNER_GRAB_RADIUS_PX:
                return (i, "corner", nearest, float(P_c_cv[nearest, 2]))
            if cv2.pointPolygonTest(quad_uvs.astype(np.float32), uv, False) >= 0:
                return (i, "whole", None, float(np.mean(P_c_cv[:, 2])))
        return None

    def _composite_roi(self, img, x0, y0, warped, mask):
        """Blend `warped` (shape h_tile x w_tile [x4]) into img[y0:y0+h, x0:x0+w]
        wherever mask is non-zero. Operates on the ROI only — tiny vs full-frame."""
        h, w = warped.shape[:2]
        roi = img[y0:y0 + h, x0:x0 + w]
        m = mask > 0
        if warped.ndim == 3 and warped.shape[2] == 4:
            alpha = (warped[..., 3].astype(np.float32) / 255.0) * m.astype(np.float32)
            alpha = alpha[..., None]
            rgb = warped[..., :3].astype(np.float32)
            roi[:] = ((1.0 - alpha) * roi + alpha * rgb).astype(np.uint8)
        else:
            m3 = m[..., None]
            roi[:] = np.where(m3, warped, roi)

    def _draw_overlay_full_quad(self, img, dst_uvs):
        h_img, w_img = img.shape[:2]
        oh, ow = self.overlay.shape[:2]

        # Bounding box of the destination quad, clipped to image.
        x0 = max(0, int(np.floor(dst_uvs[:, 0].min())))
        y0 = max(0, int(np.floor(dst_uvs[:, 1].min())))
        x1 = min(w_img, int(np.ceil(dst_uvs[:, 0].max())))
        y1 = min(h_img, int(np.ceil(dst_uvs[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            return  # entirely off-screen
        tile_w, tile_h = x1 - x0, y1 - y0

        src = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst_uvs.astype(np.float32))
        # Translate H so tile origin is (0, 0) in the warp output.
        T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
        H_tile = T @ H
        warped = cv2.warpPerspective(
            self.overlay, H_tile, (tile_w, tile_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
        poly_local = dst_uvs.astype(np.int32) - np.array([x0, y0], dtype=np.int32)
        cv2.fillConvexPoly(mask, poly_local, 255)
        self._composite_roi(img, x0, y0, warped, mask)

    def _clip_triangle_near(self, verts_cv, uvs):
        zs = verts_cv[:, 2]
        front = zs >= NEAR_Z
        n_front = int(front.sum())
        if n_front == 3:
            yield verts_cv, uvs
            return
        if n_front == 0:
            return

        poly_v, poly_u = [], []
        for i in range(3):
            j = (i + 1) % 3
            if front[i]:
                poly_v.append(verts_cv[i])
                poly_u.append(uvs[i])
            if front[i] != front[j]:
                t = (NEAR_Z - verts_cv[i, 2]) / (verts_cv[j, 2] - verts_cv[i, 2])
                poly_v.append(verts_cv[i] + t * (verts_cv[j] - verts_cv[i]))
                poly_u.append(uvs[i] + t * (uvs[j] - uvs[i]))

        poly_v = np.array(poly_v)
        poly_u = np.array(poly_u)
        for k in range(1, len(poly_v) - 1):
            yield poly_v[[0, k, k + 1]], poly_u[[0, k, k + 1]]

    def _render_triangle(self, img, verts_cv, uvs):
        h_img, w_img = img.shape[:2]
        oh, ow = self.overlay.shape[:2]
        dst = self._project_cv(verts_cv).astype(np.float32)
        src = np.column_stack([uvs[:, 0] * ow, uvs[:, 1] * oh]).astype(np.float32)

        area = abs((dst[1, 0] - dst[0, 0]) * (dst[2, 1] - dst[0, 1])
                   - (dst[2, 0] - dst[0, 0]) * (dst[1, 1] - dst[0, 1]))
        if area < 1.0:
            return

        x0 = max(0, int(np.floor(dst[:, 0].min())))
        y0 = max(0, int(np.floor(dst[:, 1].min())))
        x1 = min(w_img, int(np.ceil(dst[:, 0].max())))
        y1 = min(h_img, int(np.ceil(dst[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            return
        tile_w, tile_h = x1 - x0, y1 - y0

        M = cv2.getAffineTransform(src, dst)
        # Shift output so tile origin = (0, 0).
        M_tile = M.copy()
        M_tile[0, 2] -= x0
        M_tile[1, 2] -= y0
        warped = cv2.warpAffine(
            self.overlay, M_tile, (tile_w, tile_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
        poly_local = dst.astype(np.int32) - np.array([x0, y0], dtype=np.int32)
        cv2.fillConvexPoly(mask, poly_local, 255)
        self._composite_roi(img, x0, y0, warped, mask)

    def _draw_overlay_clipped(self, img, P_c_cv):
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        for tri_idx in [(0, 1, 2), (0, 2, 3)]:
            v3 = P_c_cv[list(tri_idx)]
            u3 = uvs[list(tri_idx)]
            for cv_v, cv_u in self._clip_triangle_near(v3, u3):
                self._render_triangle(img, cv_v, cv_u)

    # ----- Drag / scale interaction -----

    def _end_interaction(self):
        self.drag = None
        self.scale = None
        self.interaction_mode = "idle"

    def _begin_drag(self, label, uv, t_wc, R_wc):
        hit = self._hit_test_anchors(uv, t_wc, R_wc)
        if hit is None:
            return
        anchor_idx, kind, corner_idx, depth_cv = hit
        P_pinch_world = self._unproject_world(uv, depth_cv, t_wc, R_wc)
        self.drag = {
            "label": label,
            "anchor_idx": anchor_idx,
            "kind": kind,
            "corner_idx": corner_idx,
            "depth_cv": depth_cv,
            "anchor_world_start": P_pinch_world,
            "corners_world_start": self.anchors[anchor_idx].copy(),
        }

    def _update_drag(self, label, uv, t_wc, R_wc):
        if self.drag is None:
            return
        if self.drag["label"] != label:
            self._end_interaction()
            self._begin_drag(label, uv, t_wc, R_wc)
            self.interaction_mode = "drag"
            return
        idx = self.drag["anchor_idx"]
        if idx >= len(self.anchors):  # anchor removed mid-drag
            self._end_interaction()
            return
        P_now = self._unproject_world(uv, self.drag["depth_cv"], t_wc, R_wc)
        delta = P_now - self.drag["anchor_world_start"]
        if self.drag["kind"] == "whole":
            self.anchors[idx] = self.drag["corners_world_start"] + delta
        else:
            c_idx = self.drag["corner_idx"]
            new_corners = self.drag["corners_world_start"].copy()
            new_corners[c_idx] = self.drag["corners_world_start"][c_idx] + delta
            self.anchors[idx] = new_corners

    def _begin_scale(self, uv1, uv2, t_wc, R_wc):
        # Choose the anchor based on whichever pinch is over a window
        # (uv1 first, fall back to uv2). Either both pinches are inside
        # the same window, or only one is — we still scale that window
        # using the geometry of both pinch points.
        hit = self._hit_test_anchors(uv1, t_wc, R_wc)
        if hit is None:
            hit = self._hit_test_anchors(uv2, t_wc, R_wc)
        if hit is None:
            return
        anchor_idx = hit[0]
        corners_world = self.anchors[anchor_idx]
        P_c_cv = self._to_camera_cv(corners_world, t_wc, R_wc)
        if not np.all(P_c_cv[:, 2] > NEAR_Z):
            return
        depth_cv = float(np.mean(P_c_cv[:, 2]))
        P1 = self._unproject_world(uv1, depth_cv, t_wc, R_wc)
        P2 = self._unproject_world(uv2, depth_cv, t_wc, R_wc)
        dist0 = float(np.linalg.norm(P2 - P1))
        if dist0 < 1e-6:
            return
        self.scale = {
            "anchor_idx": anchor_idx,
            "depth_cv": depth_cv,
            "midpoint_start": (P1 + P2) * 0.5,
            "dist_start": dist0,
            "corners_start": corners_world.copy(),
            "centroid_start": np.mean(corners_world, axis=0),
        }

    def _update_scale(self, uv1, uv2, t_wc, R_wc):
        if self.scale is None:
            return
        idx = self.scale["anchor_idx"]
        if idx >= len(self.anchors):
            self._end_interaction()
            return
        P1 = self._unproject_world(uv1, self.scale["depth_cv"], t_wc, R_wc)
        P2 = self._unproject_world(uv2, self.scale["depth_cv"], t_wc, R_wc)
        midpoint_now = (P1 + P2) * 0.5
        dist_now = float(np.linalg.norm(P2 - P1))
        s = float(np.clip(dist_now / self.scale["dist_start"], 0.2, 5.0))
        midpoint_delta = midpoint_now - self.scale["midpoint_start"]
        new_centroid = self.scale["centroid_start"] + midpoint_delta
        self.anchors[idx] = (
            (self.scale["corners_start"] - self.scale["centroid_start"]) * s + new_centroid
        )

    def _update_interaction(self, hand_result, img_shape, t_wc, R_wc):
        h, w = img_shape[:2]
        pinches = []
        if hand_result is not None and self.K is not None:
            for i, lms in enumerate(hand_result.hand_landmarks):
                label = hand_result.handedness[i][0].category_name
                uv, d = _pinch_state(lms, w, h)
                if d < PINCH_THRESH_NORM:
                    pinches.append((label, uv))

        # Decide target mode. Only pinches that hit an existing window count;
        # pinches in empty space are handled by _process_global_gestures.
        if len(self.anchors) == 0 or len(pinches) == 0:
            target = "idle"
        elif len(pinches) == 1:
            hit = self._hit_test_anchors(pinches[0][1], t_wc, R_wc)
            target = "drag" if hit is not None else "idle"
        else:
            hit1 = self._hit_test_anchors(pinches[0][1], t_wc, R_wc)
            hit2 = self._hit_test_anchors(pinches[1][1], t_wc, R_wc)
            target = "scale" if (hit1 is not None or hit2 is not None) else "idle"

        if target != self.interaction_mode:
            self._end_interaction()
            if target == "drag":
                self._begin_drag(pinches[0][0], pinches[0][1], t_wc, R_wc)
            elif target == "scale":
                self._begin_scale(pinches[0][1], pinches[1][1], t_wc, R_wc)
            self.interaction_mode = target
        else:
            if target == "drag":
                self._update_drag(pinches[0][0], pinches[0][1], t_wc, R_wc)
            elif target == "scale":
                self._update_scale(pinches[0][1], pinches[1][1], t_wc, R_wc)

    # ----- Global gestures: place new anchor / clear all -----

    def _process_global_gestures(self, hand_result, img, t_wc, R_wc, vio_tracking):
        h, w = img.shape[:2]
        now = time.monotonic()

        # Pinch + hold in *empty space* drops a new window. Pinches that fall
        # inside an existing window are reserved for drag/scale.
        empty_pinch_uv = None
        if vio_tracking and hand_result is not None:
            for lms in hand_result.hand_landmarks:
                uv, d = _pinch_state(lms, w, h)
                if d < PINCH_THRESH_NORM and self._hit_test_anchors(uv, t_wc, R_wc) is None:
                    empty_pinch_uv = uv
                    break

        if empty_pinch_uv is not None:
            if self._anchor_pinch_start is None:
                self._anchor_pinch_start = now
            elapsed = now - self._anchor_pinch_start
            if elapsed >= ANCHOR_HOLD_SEC:
                self.anchors.append(self._build_anchor_corners(t_wc, R_wc))
                print(f"[ar_anchor] anchored window #{len(self.anchors)}")
                self._anchor_pinch_start = None
            else:
                progress = elapsed / ANCHOR_HOLD_SEC
                cv2.ellipse(img, (int(empty_pinch_uv[0]), int(empty_pinch_uv[1])),
                            (22, 22), 0, 0, int(360 * progress), (0, 255, 255), 3)
        else:
            self._anchor_pinch_start = None

        # Clear-all-on-both-palms-open.
        both_open = (
            hand_result is not None
            and len(hand_result.hand_landmarks) >= 2
            and all(_is_open_palm(lms) for lms in hand_result.hand_landmarks)
        )
        if len(self.anchors) > 0 and both_open:
            if self._open_palm_start is None:
                self._open_palm_start = now
            elapsed = now - self._open_palm_start
            if elapsed >= CLEAR_HOLD_SEC:
                count = len(self.anchors)
                self.anchors.clear()
                self._end_interaction()
                print(f"[ar_anchor] cleared all {count} window(s) (both palms open)")
                self._open_palm_start = None
            else:
                progress = elapsed / CLEAR_HOLD_SEC
                cv2.ellipse(img, (w // 2, h - 60), (28, 28), 0, 0,
                            int(360 * progress), (0, 0, 255), 4)
                cv2.putText(img, f"clearing {len(self.anchors)} window(s)...",
                            (w // 2 - 100, h - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            self._open_palm_start = None

    def _draw_hands(self, img, hand_result):
        if hand_result is None:
            return
        h, w = img.shape[:2]
        for i, lms in enumerate(hand_result.hand_landmarks):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], (180, 180, 180), 1)
            for j, p in enumerate(pts):
                color = (0, 0, 255) if j in (4, 8) else (0, 255, 0)
                cv2.circle(img, p, 3, color, -1)
            d_norm = float(np.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y))
            if d_norm < PINCH_THRESH_NORM:
                mid = (int((lms[4].x + lms[8].x) * 0.5 * w),
                       int((lms[4].y + lms[8].y) * 0.5 * h))
                cv2.circle(img, mid, 12, (255, 255, 0), 2)

    def _draw_anchors(self, img, t_wc, R_wc):
        """Render every anchor in self.anchors. Returns the index of the
        anchor currently being interacted with (drag/scale), or None."""
        active_idx = None
        if self.drag is not None:
            active_idx = self.drag.get("anchor_idx")
        elif self.scale is not None:
            active_idx = self.scale.get("anchor_idx")

        for idx, corners_world in enumerate(self.anchors):
            P_c_cv = self._to_camera_cv(corners_world, t_wc, R_wc)
            zs = P_c_cv[:, 2]
            n_front = int(np.sum(zs > NEAR_Z))
            if n_front == 0:
                continue
            elif n_front == 4:
                uvs = self._project_cv(P_c_cv)
                self._draw_overlay_full_quad(img, uvs)
                outline_color = (0, 255, 255) if idx == active_idx else (0, 255, 0)
                cv2.polylines(img, [uvs.astype(np.int32)], isClosed=True,
                              color=outline_color, thickness=1 + (idx == active_idx))
                # Per-window label at the top-left corner
                tl = uvs[0].astype(int)
                cv2.putText(img, f"#{idx + 1}", (int(tl[0]) + 4, int(tl[1]) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, outline_color, 1)
            else:
                self._draw_overlay_clipped(img, P_c_cv)

    # ----- Main loop -----

    def run(self):
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        print(f"[ar_anchor] running ({self.window_title}).")
        print("  Hand gestures:")
        print(f"    pinch + hold ~{ANCHOR_HOLD_SEC:.1f}s in empty space   drop a new window")
        print("    pinch inside a window           drag that window")
        print("    pinch near a corner             drag that corner")
        print("    pinch with both hands           scale that window")
        print(f"    both palms open ~{CLEAR_HOLD_SEC:.0f}s             clear ALL windows")
        print("  Close the window to quit.")

        model_path = _ensure_hand_model()
        hand_tracker = HandTracker(model_path)

        while self.mainLoop():
            transData = self.inputTrans.get()
            imgFrame = self.inputImg.get()
            if transData is None or imgFrame is None:
                continue
            if self.K is None:
                self._read_intrinsics(imgFrame)

            now = time.monotonic()
            if self._loop_t_prev is not None:
                dt_ms = (now - self._loop_t_prev) * 1000.0
                if self._loop_ms == 0.0:
                    self._loop_ms = dt_ms
                    self._fps = 1000.0 / max(dt_ms, 1.0)
                else:
                    self._loop_ms = 0.9 * self._loop_ms + 0.1 * dt_ms
                    self._fps = 0.9 * self._fps + 0.1 * (1000.0 / max(dt_ms, 1.0))
            self._loop_t_prev = now

            t = transData.getTranslation()
            q = transData.getQuaternion()
            t_wc = np.array([t.x, t.y, t.z], dtype=np.float64)
            R_wc = quat_to_rotmat(q.qx, q.qy, q.qz, q.qw)

            if self._prev_t_wc is not None:
                self._motion_accum += float(np.linalg.norm(t_wc - self._prev_t_wc))
            self._prev_t_wc = t_wc.copy()
            vio_tracking = self._motion_accum > 0.05

            img = imgFrame.getCvFrame()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            hand_result = hand_tracker.detect(img)

            self._process_global_gestures(hand_result, img, t_wc, R_wc, vio_tracking)
            self._update_interaction(hand_result, img.shape, t_wc, R_wc)

            self._frame_count += 1

            status_color = (0, 255, 0) if vio_tracking else (0, 0, 255)
            status_text = "tracking" if vio_tracking else "NOT TRACKING - move camera"
            cv2.putText(img, status_text, (10, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

            # Latency / throughput overlay. Frame and pose ages come from
            # device-side timestamps converted to host monotonic time; if the
            # epoch reference doesn't line up they may read very large/negative,
            # in which case those fields are omitted.
            try:
                img_age_ms = (now - imgFrame.getTimestamp().total_seconds()) * 1000.0
            except Exception:
                img_age_ms = None
            try:
                pose_age_ms = (now - transData.getTimestamp().total_seconds()) * 1000.0
            except Exception:
                pose_age_ms = None
            parts = [f"FPS {self._fps:.1f}", f"loop {self._loop_ms:.1f}ms"]
            #if img_age_ms is not None and 0 <= img_age_ms < 60000:
                #parts.append(f"img {img_age_ms:.0f}ms")
            if pose_age_ms is not None and 0 <= pose_age_ms < 60000:
                parts.append(f"pose {pose_age_ms:.0f}ms")
            cv2.putText(img, " | ".join(parts), (10, img.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            if len(self.anchors) == 0:
                msg = ("pinch + hold in empty space to anchor"
                       if vio_tracking else "wait for tracking, then pinch")
                cv2.putText(img, msg, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                self._draw_anchors(img, t_wc, R_wc)
                cv2.putText(img, f"{len(self.anchors)} window(s)", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if self._frame_count % 30 == 0:
                    print(f"[pose] t_wc={t_wc.round(3)} "
                          f"anchors={len(self.anchors)} mode={self.interaction_mode}")

            if self.interaction_mode != "idle":
                cv2.putText(img, f"[{self.interaction_mode}]",
                            (img.shape[1] - 90, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            self._draw_hands(img, hand_result)

            cv2.imshow(self.window_title, img)
            cv2.waitKey(1)
            if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
