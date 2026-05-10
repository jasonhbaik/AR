# AR anchor

![AR anchor demo](demo.gif)

A small Python prototype that pins a picture to a fixed point in 3D space
using an **OAK-D Lite** camera, then lets you grab and move it with your
hands. The picture stays where you put it as you walk around — its
on-screen position updates from visual-inertial odometry, while
MediaPipe drives a pinch-based gesture interface.

## What it does

- **Visual-inertial odometry** on the OAK-D's stereo + IMU streams
  estimates the camera's pose in 3D world space (via DepthAI's
  `RTABMapVIO` node).
- **Anchored picture(s)** are stored as 3D corner points in world
  coordinates. Each frame they're projected back into the camera image
  with a perspective warp, so the picture appears to stay glued to a
  fixed point in the room.
- **MediaPipe `HandLandmarker`** detects two hands per frame and feeds a
  small gesture state machine for pinch-to-grab / scale interactions.
- **Multiple "windows"** can coexist — pinch + hold in empty space drops
  another one. Each is hit-tested independently.

## Hardware + dependencies

- **OAK-D Lite** (Luxonis) — stereo mono pair on `CAM_B` / `CAM_C` and
  an internal IMU. Plug it in via USB before running.
- **Python 3.13** (the venv at `.venv/`).
- Python packages — see `requirements.txt`:
  - [`depthai`](https://docs.luxonis.com/projects/api/) — pipeline,
    cameras, stereo depth, feature tracker, RTABMap VIO node.
  - [`opencv-python`](https://docs.opencv.org/) — frame display, the
    perspective warp / affine warp used to draw the overlay, polygon
    fill for masks, hand skeleton drawing.
  - `numpy` (`<3.0`) — pose math, projection, unprojection.
  - [`mediapipe`](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
    — `HandLandmarker` task in `VIDEO` running mode for per-frame hand
    landmark detection.

The MediaPipe hand model file (`hand_landmarker.task`, ~7 MB) is
auto-downloaded on first run.

## Running

```bash
.venv/Scripts/python ar_anchor.py
```

You'll see an OpenCV window titled **"AR anchor (VIO)"** with the
rectified left mono camera feed. Move the camera side-to-side for a few
seconds so VIO can bootstrap (the bottom-left text flips from
`NOT TRACKING - move camera` to `tracking`). Once it's tracking, you can
drop your first window.

## Gestures

All controls are hand gestures — there are no keyboard bindings.

| Gesture | Action |
|---|---|
| Pinch (thumb + index touching) and hold ~0.4 s **in empty space** | Drop a new window 1.5 m ahead of the camera. A yellow ring fills around the pinch as the hold timer counts. |
| Pinch *inside* a window | Drag the whole window |
| Pinch *near a corner* (within ~35 px) | Drag just that corner |
| Pinch with **both hands** in / near a window | Scale that window around the midpoint between your pinches |
| Both hands open (4+ fingers extended) for ~1 s | Clear **all** windows. A red ring at the bottom of the frame fills as the timer counts. |
| Close the OpenCV window (X button) | Quit |

The hand skeleton is drawn on every frame; thumb and index tips are red,
the rest are green, and the pinch midpoint shows as a cyan ring while
you're pinching.

## On-screen status

- **Top-left** — `N window(s)` once anchors are placed; `pinch + hold in
  empty space to anchor` while empty.
- **Top-right** — `[drag]` or `[scale]` while you're interacting.
- **Bottom-left**, two lines:
  - `tracking` (green) / `NOT TRACKING - move camera` (red).
  - Latency: `FPS X | loop Yms | img Zms | pose Wms` —
    - `FPS / loop` measure the host loop.
    - `img / pose` are the staleness of the most recent frame and pose
      message in milliseconds (capture-on-device → host-now). Hidden if
      the host/device timestamp epochs don't line up.

## Files

- **`ar_anchor.py`** — DepthAI pipeline wiring. Builds the camera/IMU/
  stereo/VIO graph and links its outputs into an `ARAnchorNode`.
- **`ar_anchor_node.py`** — the AR rendering + gesture logic, packaged
  as a custom `dai.node.ThreadedHostNode`. Owns the MediaPipe hand
  tracker, the per-window gesture state machine, the perspective /
  triangle-clipped overlay rendering, and the latency overlay.
- **`overlay.png`** — the picture that gets anchored. Replace with any
  image (alpha channel honored).

## Architecture (one paragraph)

The DepthAI pipeline runs on the OAK-D's VPU: stereo rectification +
disparity from `CAM_B` / `CAM_C`, Harris feature tracking on the
rectified left, IMU at 200 Hz, all fused into a 6-DoF pose by
`RTABMapVIO`. That pose plus the rectified-left frame are sent to
`ARAnchorNode`, which runs on the host. Each frame the node:
unprojects nothing yet, just receives the world-frame pose and image →
runs MediaPipe on a downscaled copy of the frame → updates the gesture
state machine (anchor placement, drag, scale, clear) → projects every
anchored window's four 3D corners back into the image and draws them
with `cv2.warpPerspective` (or per-triangle `cv2.warpAffine` when
corners are partially behind the camera) → composites with alpha →
shows the result.

## Performance notes

The host loop is the bottleneck, dominated by MediaPipe inference and
the OpenCV warp + composite. Optimizations already applied:

- DepthAI input queues are non-blocking with `maxSize=1`, so the loop
  always sees the freshest pose / frame instead of draining a backlog.
- MediaPipe inference runs on a 50 %-downscaled frame; landmarks are
  normalized so consumers don't change.
- Overlay warps and composites are clipped to the on-screen bounding
  box of each window — only the visible polygon area gets touched.
- The per-point projection math is vectorized with numpy instead of
  Python loops.

If you need more throughput, the next levers are dropping
`num_hands=2 → 1` in `HandTracker`, or detecting on every other frame
and reusing the last result.
