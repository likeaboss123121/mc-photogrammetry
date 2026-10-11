# OPENCV CPU support producer dtypes ? frozen implementation review

Task identity: 2026-10-05; implementation checkpoint: 2026-10-06. State: FROZEN / REVISE_DOCUMENTATION_PENDING_REVIEW / PURE_TESTS_PASS / NATIVE_NOT_STARTED. Astra accepted the conditional numerical contract; Luna identified inventory omissions. This report-only revision remains REVISE until Astra/Luna close the documentation gap and root confirms it. No runtime, image, capture, build, observer or reconstruction work was performed. The native harness remains paused; A-SCALE remains held.

The evaluator-only V2 module adds interleaved uint8 C3 and bounded float32 C1 beside the existing uint8 C1 symbolic oracle. All old C1 numerical/layout/border/FP/ledger behavior is retained. Packed version advances from 3 to 4 because dtype/channel/border interpretation is now sealed explicitly. Schema-3 witnesses are historical and are rejected by the schema-4 verifier; their original report is unchanged. This is not a native output or minimal-influence certificate.

## Exact producer-to-contract inventory

Every listed producer forwards its source dtype/channels unchanged through `lattice.rectify_plane` (`lattice.py:803-804`). Maps are distinct absolute planar float32 arrays (`:800-802`), with invalid projection maps replaced by -1; interpolation is INTER_LINEAR and border is integer 0/BORDER_CONSTANT. Neither producer call supplies an accurate hint nor attests dispatch/FP. Accurate resolution and FP/dispatch must therefore be independently verified before any native applicability claim.

| Producer source anchors | Required source contract | Admission here |
| --- | --- | --- |
| `lattice_planes.py:937-940` image period patch; `:1821` signature original image; `lattice.py:1306` generic period image | Caller dtype forwarded; ordinary image uint8 C3, grayscale uint8 C1 | Exact uint8 C1 or tightly interleaved uint8 C3 only; no coercion |
| `lattice_planes.py:942-944` period region mask; `:1531` boolean-region conversion | float32 C1 0/1 | Positive zero or normal [0,1], checked per active pixel |
| `lattice_planes.py:1824-1827` coverage | float32 C1 ones or converted usable mask | Boolean/0-1 coverage admitted; arbitrary usable values require the same byte-level domain validation |
| `lattice_planes.py:1863` decoded region membership | float32 C1 0/1 | Same float domain |
| `lattice_planes.py:1220-1237,:1381-1383` rasterized structural weights; `image_derived_lattice_measure.py:481-486,:506` structural replay | float32 C1 raster, default confidences clipped to [.25,1], background +0 | Same float domain; no changed production measurement or observer calls |
| `lattice.py:1313` generic caller mask | Arbitrary caller dtype/values | Only an explicitly matching admitted contract; all other values/dtypes refused |
| `supported.py:539-541` legacy helper caller image | Caller image passed unchanged | Conditional uint8 C1/C3 only when exact actual dtype/channel/layout metadata matches; no arbitrary-image coverage claim |
| `supported.py:549-552` legacy helper generated coverage | float32 C1 ones | Admitted source domain; same FP/dispatch/layout/alias gates still required |
| `supported.py:558-560` legacy helper caller mask | Caller mask passed unchanged before post-remap comparison | Exact uint8 C1/C3 or bounded float32 C1 (+0/normal [0,1]) only; bool/float64/other dtypes and invalid float values refused before native applicability; no conversion |
| `pipeline/image_derived_unit_face.py:259-262` crop image | Crop source passed unchanged; existing advanced/evaluation adapter with V1 witness only | Conditional uint8 C1/C3 only; crop origin, pixel/row strides, native address and detached byte span must bind the exact native crop view; no V2 integration claim |

The `supported.py` legacy helper and `image_derived_unit_face.py` advanced/evaluation adapter are additional shared-seam producer groups, outside the observer-required producer set described above. They are included for inventory completeness, not newly enabled or adapted. `supported.py` converts rectified `inside` and `seen` to float64 only after remap (`:554,:562`); later channel averaging and `>0.5` do not change the source remap contract. In particular, a uint8 C1/C3 mask is conditionally supported as its exact uint8 values, not silently normalized to a float mask. A bool mask, arbitrary unsupported dtype or invalid float32 domain is refused before claiming native applicability. Crop metadata must identify the true native crop origin/strides/address/span including padding; a copied contiguous snapshot cannot substitute its own layout metadata for the original view. Neither group establishes universal coverage for arbitrary caller images/masks, V2 observer readiness or native output parity.

`calibration.py:365,423-451` uses default contrast floor .25 and clips weights to [floor,1], with an all-ones fallback. `_weight_image` writes those confidences into a float32 zero raster using LINE_8, replacing rather than accumulating overlapping segments. No required default structural producer above 1 was found by source audit. This does not prove every caller is well-formed: custom floor/weights, arbitrary masks, NaN-bearing inputs and native address metadata must pass the strict validator. Float64, int/uint16, float C3, bool arrays, planar RGB, relative/interleaved maps, arbitrary pixel/channel strides and unsupported input values remain unsupported.

## Pinned source numerical argument

Source commit `40738fb16ceddb5fb3fea747585f7ce6abb0605b` (OpenCV 5.0.0). Existing detached source files were read and rehashed; no native import or call.

| Pinned source | SHA-256 | Audited anchors |
| --- | --- | --- |
| imgwarp.cpp | b7a95e4bb5c2c5adc5378c43151fa4bc6e461ec3bb267d9d72eda91a004595ae | :1525 C1; :1529-1535 accurate C3; :1557-1559 float C1 |
| warp_kernels.simd.hpp | 3e19fdf116c99babd38775f543301276db59ae8bbfa50edaaba7afe89c1a7c73 | :26-46 coordinates/offsets; :5674-5808 C3; :6286-6393 float C1; :6296 float stride units |
| warp_common.scalar.hpp | 3da7138a38c611aedd4f4805b35f35cde9e5e038107bedce04c4295fe1f35af6 | :60-65 C1 fetch; :74-83 C3 same-channel fetch/border; :156-178 separable arithmetic; :191-207 stores |
| warp_common.vector.hpp | ca5314ca884189cf790afc4d14ef121f3fd30f7b25bf3054f977cef81160df21 | :176-199 C1/C3 fetch; :443-459 per-channel alpha/beta FMA |

RGB has three independent same-index channels. Its CSR lists spatial pixels once; schema 4 explicitly means `output channel c -> input channel c at each listed pixel`, for c=0,1,2. Storage order is preserved; no grayscale averaging, color reordering or cross-channel operation. Constant terms mean +0 in every new RGB channel. Existing C1 integer-border saturation remains unchanged.

Float32 sources admit exactly raw +0 or positive normal binary32 values no larger than1 (bits0 or0x00800000..0x3f800000). Negative zero is deliberately refused with other negative/out-of-domain/subnormal/nonfinite values. The scan checks every logical source pixel, including zero-weight or completely unsampled pixels, after reservation and before coordinate/output allocation. Padding is not a pixel but is sealed in the input hash. Float border is exactly integer0 converted to float +0; arbitrary float border, -0.0, bool border and nonzero new-type border are refused.

Finite source p in[0,1] and admitted rounded alpha/beta in[0,1] keep p01-p00 in[-1,1], each rounded alpha product in[-1,1], alpha sum in[-1,2], beta difference/product in[-3,3], and final sum in[-4,5]. Endpoints are exactly representable, so RNE or fused forms cannot overflow or produce NaN/Inf. This deliberately loose bound is sufficient for zero coefficients: native `p01-p00` still runs, but all operands are finite and bounded, preventing0*Inf or NaN propagation. Source NaN/Inf/extreme values are rejected even when their ideal symbolic coefficient is zero. Intermediate underflow, FTZ, rounding cancellation, FMA choice and uint8 quantization can still change numeric output or actual minimal influence. The retained dyadics are exact symbolic products of the sealed rounded float32 fraction bits, not claimed native coefficients or exact output arithmetic. No epsilon comparison or native byte prediction occurs.

C3 uses signed32 byte arithmetic `stride_bytes*iy + 3*ix`; the implementation separately guards `3*ix`, the row product, sum and +2 channel extent at all four corners before omitting zero weights. Float C1 uses `stride_bytes/4` element arithmetic, not byte stride; stride/address align to4, signed32 product/sum are guarded in element units and in-bounds byte extents are checked after multiplying by4. uint8 aligns to1, so C3 row strides/addresses need not divide by3. Destination pixel/row byte extent uses the same exact source type. Pointer64 span checks and all pairwise source/maps/destination alias checks include padding.

## Packed schema, seal and resource lifecycle

Schema4 retains192-byte header,16-byte per-row bases/fraction bits, two uint32 CSR pointer streams and40-byte terms. Header's former36 reserved bytes are nine uint32 contract words: dtype code(1=u8C1,2=u8C3,3=f32C1), channels, exact pixel byte stride, native offset unit bytes, same-channel mapping code1, constant scalar bits/value, source byte span, destination byte span, and float-domain code. Source/map/output dtype and channels, full bytes/padding, native addresses/strides and source identity remain in the canonical request digest. The verifier recomputes every descriptor, base, raw fraction, ordered term, CSR boundary and seal. Dtype/channel/unit/domain resealing cannot bypass request consistency. It does not authenticate wholesale replacement of a trusted ingress request; ingress trust is external.

For N<=262144: W(N)=4096+184N+8, Wmax=48,238,600 bytes. Actual all-four-term max payload=192+184N+8=48,234,696 bytes, identical across all three contracts. I=sum actual source and two map spans in bytes, each<=16MiB, Imax=50,331,648. R=2I+2W+16MiB; formal Rmax=213,917,712<268,435,456. Padded512? fixtures have I=50,331,528, R=213,917,472. Destination is checked to16MiB but detached output bytes are not part of this symbolic module; a future native harness must account its own IPC/output/artifacts separately. There is no RGB CSR triplication. The existing16MiB scratch margin and bound do not grow; new validation/descriptor work is constant-space, with source scan time proportional to logical source pixels (bounded by source span).

The authoritative exclusive provisional/committed ledger is unchanged: failed/pre-output builds clean inactive frames before refund, escaped/committed witnesses retain their charge permanently, exact owned verifier reuse reserves no second witness, foreign verification requires a temporary full reservation, and a second worst-case witness refuses. No production release/reset; existing interruption-handoff and concurrent/nested tests retained. This is a module-lifetime memory envelope, not a process RSS measurement or multi-view lifecycle admission.

## Validation

Exact full pure command (only assigned focused test module):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv/Scripts/python.exe -B -m pytest tests/test_evaluation_remap_support_v2.py -o addopts= -p no:cacheprovider -q --no-header
```

Result: **286 passed in 138.30s (0:02:18)**, terminal exit 0, session 69840. One full-suite process only; no duplicate execution. Preliminary subset: 269 passed,5 deselected in 0.54s before additional controls. The original 171 C1 cases remain; new cases cover RGB impulses/channel order/zero membership/padding, float domain/+0/zero-weight hazards, element-byte offsets and overflow, map alignment, typed metadata, canonical resealed descriptor tamper, source padding sealing, bounded pre-work refusal, tiny-normal dyadics and both new 512? worst-case lifetime cases. Tests import the module directly under a cv2/NumPy/production import guard; Fraction evaluations are symbolic expectations only. Source/test AST parsing passed; production symbolic module has no bare assertions or native/production imports. Byte QA passed: UTF8, LF, final newline, no tabs/trailing whitespace. Report and source/test pins are frozen; no further source edits expected.

## Frozen project pins

- src/mc_photogrammetry/reconstruction/evaluation_remap_support_v2.py: `7f332788762ec308ad396aa3c8c82a4364b81ccd09d6311200824ce0c844769d`
- tests/test_evaluation_remap_support_v2.py: `36a39b02d7f93e74825f9fdf7defce622f8ecfe0df5eb6e4257e0b1f553e9bfa`

Production/source audit closure (read only):

- calibration.py:2ee1690a817a59f8b6ffd7858b65ce9d36bb01a6506577261b267c79e8912bfc
- lattice.py:3077b4c380b293fe7b97a7a127cd09e42e79fc7277de654168a8ec0274a87bd7
- lattice_planes.py:4c6853a0ad24589c244a3d0f472039ea53e9c079095e7069f0af930686a22f10
- pipeline/image_derived_lattice_measure.py:491841a58e5faca5b07200d88a0115c3aca2d25c6328206755b3295e0b785d27
- scripts/evaluation/capture_plane_period_support.py:7d17734cbce8cbbc5490b6f82363d72edecccd5ff098b4a4a63f3a0f2d23e34d
- reconstruction/supported.py: `4c94564d3842f1c3a03883f6f95a78a4ac08260150eb9fc12251bf708e35f29c` (additional read-only inventory source)
- pipeline/image_derived_unit_face.py: `912d70b21ca24f5277391766bf92a76700135db8fc0216e44e40f52520163307` (additional read-only inventory source)
- Historical docs/reports/OPENCV-CPU-SUPPORT-V2-2026-10-05.md:24b44c5a25659be249e92391abbc913b3f8f83be98dac3388c9315ef93c84b54 (unchanged)

REFERENCE_KERNEL artifact/build identities are unchanged: manifest46d3b51cd4b3ea2efa2592e47b3204ae079652ef9ac7a6c3cb31d5f38ffe3421; extension15e8c9f063076e4e6c07bbba053b0ce02fcfb201eff35145ce2395aae686e4e0; core2149d6ca8914330ac9d2ebfbbade70af4e9d58f6a427fb8010664fd45fb55159; imgproc6a7be1e21436a9f909eed1dfdf12f9277fe472010de51b82108537cbae8d1276; compile_commandsf7b7540b3afff8edea808dee333c0d4d43640aeedfc95a2a2f7e92e047d24280; build_information20e38ff4f7bb09560492f2b4bca416f6d626feb7762b62dfe89abda66f8e7eab; precompile_gates4db1525235cab2261a30a29e09e4fdd19ed9fc4415bbeb0cff19aa33482cb077. These are sealed historical reference labels, not re-attested runtime observations.

No native harness paths, observer evidence or observer run directory exist at this checkpoint. No native cv2/NumPy call/import was made. Installed custom-HAL certification, actual branch/FP ingress, numeric dtype/shape/output-byte replay, historical R8 equivalence, multi-view ownership, observer execution/capture and reconstruction accuracy remain blocked. Review acceptance can establish only these expanded pure pinned-source symbolic contracts. No product gate or advanced-only manual camera/F3/coordinate policy changed.
