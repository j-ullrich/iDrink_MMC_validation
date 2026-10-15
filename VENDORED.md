# Vendored third-party code

None of the upstream commit hashes were recorded when this code was copied in. Fill them in when known
(`git log` of the upstream repository around the copy date, or compare against upstream tags).

| Path | Upstream | Local changes |
|------|----------|---------------|
| `src/Pose2Sim/` | [perfanalytics/pose2sim](https://github.com/perfanalytics/pose2sim), see `src/Pose2Sim/P2S_Version_Info.txt` | modified for iDrink (calibration, triangulation, filtering). The unused `Utilities/` converters were removed. |
| `src/Metrabs_PoseEstimation/metrabs_pytorch/` | `metrabs_pytorch` of [isarandi/metrabs](https://github.com/isarandi/metrabs) | `multiperson/person_detector.py`: `int(...)` instead of `np.int32(...)` for the target size. |
| `src/Metrabs_PoseEstimation/metrabs/` | [isarandi/metrabs](https://github.com/isarandi/metrabs) (full clone, kept for `metrabs_tf` used by `metrabs_pytorch/convert_model_from_tf.py`) | none known. Its nested `metrabs_pytorch/` duplicate was removed in favour of the patched top-level copy. |
| `src/Metrabs_PoseEstimation/posepile/` | [isarandi/posepile](https://github.com/isarandi/posepile) | unknown |

The iDrink wrappers around Metrabs are `metrabsPose{2D,3D}_{TF,pt}.py`. Code shared by them lives in
`metrabs_common.py` (filtering, .trc writing) and `metrabs_pt_models.py` (PyTorch model loading).
