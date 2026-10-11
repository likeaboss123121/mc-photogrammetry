# SOL-IMAGE-POINT-ROSTER-01 ? mandatory source-preflight stop

The task stops before implementation. The inspected sources establish continuous pixel coordinates in the supplied detector raster and a native-resolution identity ingress path, but do not close the required crop+resize pixel-centre backprojection contract. A separate permutation-stable feature-ID rule is also not established by the returned feature record. No project code was imported/executed, no tests ran, no synthetic or ordinary image was read/decoded, and no detector/descriptor/geometry computation occurred. Neither implementation path was created. This is a missing provenance/coordinate contract, not a demonstrated production bug.

The source preflight and hashes below were recorded before any project import. All 17 inspected source/test files remain unchanged. Tests were read as evidence, never run. The frozen task card SHA256 is 2D24EDF269E8CE9769BE9E4C12B13FE8566C635543FA68FB95DE207EDECEDBD3.

## Exact source chain and established evidence

- `input/media.py:74-118` discovers retained frames in stable path order, hashes the single encoded-byte buffer, de-duplicates, and creates frame IDs; `pipeline/image_derived_entry.py:486-501` decodes that same buffer and attaches an identity `FrameTransform` based on decoded shape. `pipeline/frame_ingress.py:219-248` binds encoded byte digest/length, decoded dtype/shape/pixel digest and supplied transform, but does not validate which resampler produced a transformed raster.
- `input/image.py:56-60` applies EXIF transpose and RGB conversion and owns a uint8 array. The native source domain is the resulting oriented RGB raster; mapping to pre-EXIF encoded-file coordinates is not separately declared here.
- `vision/gui_mask.py:26-30` defines True as exclusion; `pipeline/correspondence.py:314-334` inverts the default GUI mask to usable=True, treats a missing explicitly supplied frame mask as all-false, and checks dimensions. `detect_features:356-365` converts RGB to grey through `_to_grey:782-787`, zeros unusable pixels before SIFT, and supplies its detector mask. There is no cross-view pose/period/mode dependency in that detector seam.
- `pipeline/correspondence.py:363-394` calls the genuine OpenCV SIFT detector and returns aligned points/descriptor/sizes tables. `keypoint.pt` becomes float64 continuous Nx2 coordinates in the supplied image. Neither the point values nor returned record contain a transform back to a different parent raster.
- `pipeline/correspondence.py:658-695` provides the genuine blocked float64 Gram expansion for L2 descriptor distance. `pipeline/directed_matching.py:222-281` wraps it into pre-pose directed edges. Fixed-k `argpartition` followed by selected-subset sorting is not an all-cutoff-ties guarantee; index tie breaks and later caps cannot be treated as physical identity. A future bounded producer would need full allowed right-table distances before preserving every kth-distance tie. No existing selected `best.matches`, ratio/mutual geometry selection, solve_pair or RANSAC was used.

## First mandatory blocker: crop/resize pixel-centre registration

`pipeline/contracts.py:13-38` declares crop origin, sizes and output/crop scale; its prose says source coordinates subtract crop origin and then scale. `CameraIntrinsics.transformed:59-67` implements that plain affine metadata rule. It does not declare whether continuous coordinates use raster centres/edges, identify an actual crop/resampler, or retain that sampler's centre registration. `tests/test_camera.py:30-40` verifies projection/unprojection within transformed intrinsics and expects principal point (475,215); it does not generate resized pixels, detect a point in them, and prove an original-raster centre backprojection. `tests/test_pipeline_contracts.py:19-24` tests identity sizes only.

The actual native entry attaches identity transforms. The available image pyramid (`compute/cache.py:14-17,54-61`) uses `cv2.resize(..., INTER_AREA)` and returns only image levels, without per-level source transforms or a coordinate-mapping record. `tests/test_compute_cache.py:7-15` verifies sizes/read-only buffers/bytes, not pixel-centre registration. `input/frames.py` performs selection/hash/quality work and supplies no missing crop/resize transform.

Thus a metadata inverse `u/scale+crop_origin` can be written algebraically, but its equivalence to the centre coordinates in a sampled detector raster has not been established from these pinned sources. Choosing that rule or a half-pixel-adjusted inverse would introduce a new convention. The task explicitly forbids inventing the missing mapping. This does not assert that either convention is universally correct or that the current native production path is broken. Native identity-only acquisition would require a scope amendment because this task requires a crop+resize original-centre control.

## Additional unclosed identity/determinism provenance

`FrameFeatures:137-157` retains caller frame ID, points, descriptors, dimensions, usable count, optional sizes and aggregate boundary-drop count. It has no source-bound per-feature ID, original detector-row map, or full keypoint identity (angle/octave/response). `detect_features:378-388` converts point/size and may filter rows, without retaining the pre-filter indices. Graph rows consequently identify current table indices. Those indices do not themselves prove stable identity across a feature-row permutation. Content hashing could become a reviewed evaluation contract, but this task does not authorize silently inventing that rule when mandatory source preflight cannot prove it.

Descriptor distances are deterministic functions of a pinned finite table/runtime. SIFT calls/config and RGB/mask handling are visible; the feature record does not bind the OpenCV detector binary/runtime/version, original inventory IDs or complete exclusion roster. Existing correspondence/direct-graph tests exercise repeatability and masked-RGB invariance, not the requested complete stable-ID/preprocessing/source-pixel contract. No claim of cross-runtime detector determinism or upstream completeness is made.

## Disposition and required design boundary

Blocked under the mandatory preflight rule; 0 tests and 0 runtime attempts. No synthetic control freeze/run or comparison budget charge took place. Acquisition caps and ordinary image-only behavior remain unchanged. The minimal next design decision must explicitly define the source raster/centre convention and source-bound crop/resampler/mask transform, plus a deterministic feature-ID/original-row/exclusion rule for the accepted inventory (or authorize a narrower identity-only domain). That is a design recommendation, not an implementation or production amendment.

Latest v2 archival receipt pins and frozen test bytes remain unchanged, and active consumed checker test remains absent. No project/helper/source edits, imports, feature extraction, image inference, solver/camera/pose/translation/mode scoring, truth/capture/manifest access, thresholds/caps/schema or output/accuracy claims. Preserve 90% visible geometry F1 / 80% exact-state gates. STOP for independent Astra/lead review.

## Exact read-only source/test pins

| Path | SHA256 |
|---|---|
| `src/mc_photogrammetry/pipeline/correspondence.py` | `192A4851E428D8116E0FBFFD28C7FF157D1AEDD8035E499C6DF7EC2A9D787FA1` |
| `src/mc_photogrammetry/pipeline/contracts.py` | `DB24E730BFED52EC983AE387A26F1B612FD94ABC3E7608053080915C899ED5CC` |
| `src/mc_photogrammetry/pipeline/frame_ingress.py` | `3411B4067281B19A6AC2E530ECE5DFC2E53E0F63B978E83E09FFADB27F715F0A` |
| `src/mc_photogrammetry/pipeline/image_derived_entry.py` | `1767B416061B20B1E60CD5C2D3D5B0814F232A3BC56D01328FC294818B20B6FC` |
| `src/mc_photogrammetry/input/image.py` | `43C318A45BC6965B6A9D3D8F43FC2375052CBE5AD2B13B07A7B7B4313F0BF732` |
| `src/mc_photogrammetry/input/media.py` | `8F872B165283FB65A01E7470062CB51040FEEEE46ECB25ED26CDB0B67B7C2F89` |
| `src/mc_photogrammetry/input/frames.py` | `ED9B166D29887EF849D3D601FDEF89306FCCAC4D40CAE4D39A48A4A6FF290D8F` |
| `src/mc_photogrammetry/compute/cache.py` | `16B52FB1F3D7D7C175C8CB9B374C6FA124C90F5C4C966AA0A8895C11DF9C1CCF` |
| `src/mc_photogrammetry/pipeline/directed_matching.py` | `AF5E5424A8E590762CB217D7D4E7D0CEBBBB1AA2945E648683093499DEF7B8FA` |
| `src/mc_photogrammetry/vision/gui_mask.py` | `1D8A6D9ED9537A2E5DD98C794AA9C45B07C748090E1A17F9DB3D230E5E74CDA9` |
| `tests/test_correspondence.py` | `550EB96EE8E1A1AB68921A2243D4B84173AE19158E36F3414C1855B4F4E2D7E0` |
| `tests/test_pipeline_contracts.py` | `3E213380AD1CD88568BD95A92ECDC603FD1204613E980BCE0EFD8142BD450B9B` |
| `tests/test_camera.py` | `6B138FBA74B3D1B5116D780670C0111335830A723B22E508B0FD3D8F43913B2F` |
| `tests/test_image_loader.py` | `269362E3E814E373F644F3ECEA54A1B97D5CFA4F8879B2AFB42E762CD80F4111` |
| `tests/test_compute_cache.py` | `853E9EE8E0E498699608BA57D06C3707BCE48D9D84E9AD9514C7A35CD657BBD3` |
| `tests/test_directed_matching.py` | `DA1AE5BD501C2CBBB9262A14A6934C37A0AB7D48D9CC7B061E592613E3B90C3A` |
| `tests/test_image_derived_entry.py` | `BD91391B0EAE739FB70601950A671106707EB8BDA3A1D69AFA620AE99E28D392` |

Evidence SHA256: `A2DA029FF86A8A079CF8A3B040541BC1FBAA76FDC05D8E4E087E73A783817AE9`.
