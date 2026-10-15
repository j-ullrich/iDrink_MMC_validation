# Vendored third-party code

None of the upstream commit hashes were recorded when this code was copied in. Fill them in when known
(`git log` of the upstream repository around the copy date, or compare against upstream tags).

| Path | Upstream | Local changes |
|------|----------|---------------|
| `src/Pose2Sim/` | [perfanalytics/pose2sim](https://github.com/perfanalytics/pose2sim), version 0.9 according to `src/Pose2Sim/P2S_Version_Info.txt` | modified for iDrink (calibration, triangulation, filtering). The unused `Utilities/` converters were removed. |
| `src/Metrabs_PoseEstimation/metrabs_pytorch/` | `metrabs_pytorch` of [isarandi/metrabs](https://github.com/isarandi/metrabs) | `multiperson/person_detector.py`: `int(...)` instead of `np.int32(...)` for the target size. |
| `src/Metrabs_PoseEstimation/metrabs/` | [isarandi/metrabs](https://github.com/isarandi/metrabs) (full clone, kept for `metrabs_tf` used by `metrabs_pytorch/convert_model_from_tf.py`) | none known. Its nested `metrabs_pytorch/` duplicate was removed in favour of the patched top-level copy. |
| `src/Metrabs_PoseEstimation/posepile/` | [isarandi/posepile](https://github.com/isarandi/posepile) | unknown |

The iDrink wrappers around Metrabs are `metrabsPose{2D,3D}_{TF,pt}.py`. Code shared by them lives in
`metrabs_common.py` (filtering, .trc writing) and `metrabs_pt_models.py` (PyTorch model loading).

## Finding the upstream commits

The only version hints in this repository: Pose2Sim "version 0.9" (`src/Pose2Sim/P2S_Version_Info.txt`) and the newest
entry of the vendored Metrabs README is dated 2023-08-02. To pin the exact commits, run in a clone of each upstream
repository (not possible from the sandbox this repository was refactored in):

```
git log --oneline --since=2023-08-01 --until=2024-12-31 -- metrabs_pytorch   # isarandi/metrabs
git diff --stat <candidate-commit> -- <path-in-upstream> <path-to-our-copy>   # an empty diff means a match
```

Add the hash to the table above once found.
