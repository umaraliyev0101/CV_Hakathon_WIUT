# Vendored from rtmlib 0.0.16 (Apache-2.0)

Files: `rtmpose.py`, `pose_pre.py` (pre_processings), `pose_post.py` (post_processings),
`det_post.py` (object_detection/post_processings), `LICENSE`.

Changes from upstream:
- `base.py` replaced by a minimal onnxruntime-only `BaseTool` (no OpenCV-DNN/OpenVINO backends,
  no checkpoint auto-download; takes a local model path; `providers` argument).
- `rtmpose.py`: imports re-pointed to the vendored module names; constructor takes `providers`
  instead of `backend`/`device`.
- Each file carries a vendoring header. Algorithms unchanged.
