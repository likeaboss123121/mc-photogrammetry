# M4 Evaluator E2: immutable R2 versus R1

E2 ran the accepted `scripts/evaluate_frozen_planes.py` implementation exactly once against the immutable R2 image-only freeze. The explicit capture, ground truth, vanilla JAR, and palette cache paths and bytes match E1. No reconstruction, source, threshold, capture, or configuration code changed. The evaluator continues to report `promotion_eligible=false`: the region-bounded GT omits exterior world occluders.

## Primary full-cube-only score

| Freeze | TP pixels | FP pixels | FN pixels | Precision | Recall | F1 | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | 21,287 | 1,219,342 | 482,495 | 0.017158 | 0.042254 | 0.024406 | 0.012354 |
| R2 | 157,521 | 10,761,705 | 346,261 | 0.014426 | 0.312677 | 0.027580 | 0.013983 |

R2 adds proposals in every frame beyond frame 00, increasing matched full-cube pixels and recall while also increasing false-positive pixels substantially; precision is lower. These are diagnostic comparisons of frozen masks, not a promotion result.

## Proposal counts and recall at K

Recall is the one-to-one matched visible full-cube plane count divided by visible GT plane count at the requested top K. R1 has no proposals in frames 01-07. R2 proposal counts total 35 across all eight frames.

| Frame | Visible planes | R1 proposals | R1 R@1/3/5/10/32 | R2 proposals | R2 R@1/3/5/10/32 |
| --- | ---: | ---: | --- | ---: | --- |
| 00 | 42 | 3 | .0238/.0714/.0714/.0714/.0714 | 3 | .0238/.0714/.0714/.0714/.0714 |
| 01 | 44 | 0 | .0000/.0000/.0000/.0000/.0000 | 5 | .0227/.0455/.0909/.0909/.0909 |
| 02 | 32 | 0 | .0000/.0000/.0000/.0000/.0000 | 2 | .0312/.0625/.0625/.0625/.0625 |
| 03 | 45 | 0 | .0000/.0000/.0000/.0000/.0000 | 6 | .0222/.0444/.0889/.0889/.0889 |
| 04 | 38 | 0 | .0000/.0000/.0000/.0000/.0000 | 4 | .0263/.0526/.0789/.0789/.0789 |
| 05 | 21 | 0 | .0000/.0000/.0000/.0000/.0000 | 7 | .0476/.0952/.0952/.1429/.1429 |
| 06 | 33 | 0 | .0000/.0000/.0000/.0000/.0000 | 4 | .0303/.0909/.1212/.1212/.1212 |
| 07 | 41 | 0 | .0000/.0000/.0000/.0000/.0000 | 4 | .0244/.0732/.0976/.0976/.0976 |

## Exclusions, provisional all-face score, and calibration

The primary score excludes 232,755 non-full-cube-owned pixels in both R1 and R2. The separately labeled provisional all-face score remains subject to repeated same-slot owner ambiguity:

| Freeze | TP pixels | FP pixels | FN pixels | Precision | Recall | F1 | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 provisional | 26,597 | 1,276,900 | 709,940 | 0.020404 | 0.036111 | 0.026075 | 0.013210 |
| R2 provisional | 180,943 | 11,141,063 | 555,594 | 0.015982 | 0.245667 | 0.030011 | 0.015234 |

Calibration comparisons are identical for R1 and R2: vertical-FOV focal error is at most 2% on 5/8 frames; best-permutation Manhattan axis-line error is at most 0.5 degrees on 5/8; median focal error is 1.4579%, and median axis-line error is 0.1738 degrees.

## Command and artifact integrity

The single scoring command was:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_frozen_planes.py --freeze exports\evaluation\frozen-image-run-r2-atom-a1-canal-town-2026-09-12\freeze.json --capture exports\datasets\fresh-c4-canal-town-framed-2026-09-12\scenes\canal-town-architectural-component-bbox\capture.json --ground-truth exports\datasets\fresh-c4-canal-town-framed-2026-09-12\scenes\canal-town-architectural-component-bbox\ground_truth.json --vanilla-jar "$env:APPDATA\.minecraft\versions\1.21.5\1.21.5.jar" --palette-cache "$env:APPDATA\.minecraft\versions\1.21.5\1.21.5.mc-textures.json" --output exports\evaluation\frozen-plane-e2-canal-town-2026-09-12\evaluation-r2.json
```

The run took approximately 115 seconds. E2 output is `exports/evaluation/frozen-plane-e2-canal-town-2026-09-12/evaluation-r2.json`, 335,926 bytes, SHA-256 `a7bb0f4d6ecac9082c325e43b207e5d0e8a3ece76f3c5634580dd397b2fc2560`, artifact ID `d9643b97afd6d381fda4b9c837258c8e7214275ed27f262c11bf3b90b1b51301`. It binds R2 artifact ID `74e0435188ba83f4a1a58d50b641c10c887d5887fa0672814394cb7bb4cb3d6c`, R2 JSON SHA-256 `3c4e8e1b63f73748bde1f072d97dbfb4276ebc90c957889123760edcc6a1c90f`, and sidecar SHA-256 `68277b112b4b7ad755878501c8c04617119477f5ceff76be9dff2c0403bc74db`.

The evaluator code hash remains `2c032d1798ca962a2361e480019b222e8080e7b96ceeb5397c2b78153a1fe031`. Before and after scoring, hashes were unchanged for R1 JSON/sidecar (`60eb97973473596ca7ba44c9c305e386615043ff336d9bd81d30e2de28744231` / `e55383ac4279a64137497823ff4f7ca07199fc56fff980e577a9dcb36767115c`), R2 JSON/sidecar (above), E1 JSON (`a7a8a8e93e0921d0fc5f1dfd4a008d44de3fc396fc1205dbb5a4f4a3cd7557fa`), capture (`c6c44d43408779006854c81742a68d00cfe5b546f506ffa08ea59ba7b812abe6`), GT (`36114feebe61d7b358927095a159d56dc5762e02c5a52a52e8a12ae17ce4a35a`), vanilla JAR (`522672ad20b460c02c2e39b6c5035ef6a849af28eb11ab2ac7eb293d395a8c11`), and palette cache (`d119381cf3717d6e300c7da73701c3d0f8358a5b7b9d6091697bee2c1e2c1934`).
