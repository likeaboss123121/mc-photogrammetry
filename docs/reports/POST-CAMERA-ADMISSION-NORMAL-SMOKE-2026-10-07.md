# Normal-mode smoke after camera admission

**Outcome: incomplete / telemetry failure; one-shot consumed; STOP for lead review.** Completion attempt 7/8. The captured pipeline result refused geometry and produced no `.litematic`. Application exit status and final resource telemetry were not persisted, so this is not a successful smoke acceptance.

## Observed pipeline result

The only application invocation used the prescribed normal-mode command and four ordered ordinary train-scene PNG paths, views 04-07. Captured stdout:

```json
{"camera_admission":{"admitted":4,"excluded":0,"exclusions":[],"original":4},"delivery_sha256":"","production_sha256":"740556556f1fd0a46c18a053dba2b48a758d161fe9cd0ae5cc0116bca39abf5c","reason":"g_unavailable","result_sha256":"13a04b7374ec1e9418fe3f1672bde36b4f0b4503420bd3000f7ffbb69906a4c3","stage":"delivery","status":"geometry_unavailable"}
```

Stderr is empty (zero bytes). Camera admission is **4 original, 4 admitted, 0 excluded**, with no exclusions. The persisted nested session is v1, because all inputs were admitted; this run did not exercise a real camera abstention.

The new `project.mcscan` is schema 3 and retains all four full source paths. Its production record provides the first reported refusal: **stage geometry, status geometry_unavailable, reason g_unavailable, detail plane_integer_residual**. Its volume is empty and promotion is null. `reconstruction.litematic` does not exist; structural validation was therefore not run.

| Stage | Retained evidence |
| --- | --- |
| Camera | Four estimates admitted; nested session v1; metric scale unknown, positions unset |
| Entry | Digest `8d605c69850222ed1f521a704029a72639b709a4580d0310afd9dc99ccd428e5` |
| Connections / pose | Individual statuses not retained in console/project; no guessed status |
| Lattice | Digest `a481595466f851db7a016dcf7d8141b0a3c05da34e278772b8ee520ce680cbf8`; individual status not retained |
| Geometry | geometry_unavailable / g_unavailable / plane_integer_residual |
| Output / delivery | No promotion, empty output/delivery digests, no litematic |

Session digest: `ce23da55adb62e0626300691bd15b175a3e8d5f8f8d91254c24275c9b4a0e037`.
Production record digest: `9f54ee419dadc23f183ff4b3fd5f568319abd27f76789bdce23aa938b60b437e`.
The refusal token is observed evidence, not a claim about its image-level cause or accuracy.

## Capture deviation and resources

The retained tool session was **26001**; launched root Python PID **7712**. A PowerShell supervisor used hidden process launch, captured stdout/stderr to the exclusive task directory, and sampled launcher/descendant process RSS and CPU via Windows process APIs. Preflight confirmed those APIs returned valid data before launch.

Supervisor start: `2026-10-07T05:14:47.6855504Z`. A target 500ms sample interval included process-query overhead; heartbeats were emitted about every ten seconds. Caps were 20 wall minutes and observed aggregate RSS over 16 GiB, with termination paths configured. CPU-only environment: `CUDA_VISIBLE_DEVICES=-1`, `HIP_VISIBLE_DEVICES=-1`, `ROCR_VISIBLE_DEVICES=-1`, `OPENCV_OPENCL_RUNTIME=disabled`; bytecode writes disabled. No model/checkpoint was used.

After application `WaitForExit()`, the supervisor failed while assembling the final receipt:

```text
Argument types do not match
At line:111 char:1
+ $receipt['samples']=@($samples)
CategoryInfo: OperationStopped: (:) [], ArgumentException
FullyQualifiedErrorId: System.ArgumentException
```

The supervisor tool exit is **1**. Application exit code, exact elapsed wall time, final sampled peak RSS/CPU, dense samples and final stop reason existed only in supervisor memory and are **unrecoverable** after this failure. No application exit code is inferred from a normal-looking JSON message or the CLI's expected behavior.

Twenty heartbeat samples remain in the captured tool console and recovered receipt. Highest retained heartbeat aggregate RSS: **950,771,712 bytes (about 0.886 GiB)**. This is not the unavailable dense-sample peak. Last retained sample: **197.65 seconds**, aggregate RSS **265,613,312 bytes**, aggregate CPU **217.58 seconds**. Exact CPU total is unknown. Stdout's last-write UTC `2026-10-07T05:18:06.5350907Z` is about 198.85 seconds after recorded supervisor start; process completion was roughly 200 seconds, with exact elapsed time unknown. No cap stop appears in retained console; missing dense telemetry does not prove all resource maxima.

Lead explicitly directed read-only recovery within this output directory, no rerun, incomplete/telemetry-failure status if application exit remained unavailable, and preservation of stdout/stderr/artifacts. Recovery followed that boundary. The initial receipt remains byte-for-byte in `run.initial.json`; `run.json` now clearly records incomplete status, unknown fields and recovered evidence. `supervisor-terminal.txt` preserves retained heartbeats and the exception. No second application invocation, structural probe or retry was launched.

## Exact invocation and preflight

Working directory: `C:\Users\Likea\Documents\vscode projects\mc photogrammetry project`.
Windows PowerShell was run with `login=false`; the app process was hidden. Bootstrap only inserted repository `src` and called the app entry.

```powershell
.venv\Scripts\python.exe -B -c "import sys; sys.path.insert(0, 'src'); from mc_photogrammetry.cli import main; raise SystemExit(main())" image-derived-deliver exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-04.png exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-05.png exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-06.png exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-07.png --destination exports/diagnostics/post-camera-admission-normal-smoke-2026-10-07/reconstruction.litematic --project exports/diagnostics/post-camera-admission-normal-smoke-2026-10-07/project.mcscan --name canal-town-normal-smoke --description "unscored normal-mode development smoke" --max-frames 4 --max-pairs 6 --max-total-support 4096 --max-edges 6
```

Output directory plus both requested product paths were absent before exclusive creation. Source inputs were verified as existing nonempty files and hashed; they are the only image inputs. The consumed C1 input list was verified from its report to contain views 00-03 only; those PNGs were not opened, hashed or decoded. No sidecar, manifest, plan, truth, camera/F3/bounds, JAR, model cache or sealed fixture was supplied.

| Input path | Bytes | Raw PNG SHA256 |
| --- | ---: | --- |
| `exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-04.png` | 2343567 | `8797d70e0e09475d296c3cf18e5892f6b22d41750ed18d6c3364da80ba60a81f` |
| `exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-05.png` | 3896541 | `6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b` |
| `exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-06.png` | 3258563 | `53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923` |
| `exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-07.png` | 2603703 | `fdd8a3709479795b204f67153b10f44b49e85f2bc0050a46d9fdd0461f025556` |

Interpreter: `C:\Users\Likea\Documents\vscode projects\mc photogrammetry project\.venv\Scripts\python.exe`, file version `3.10.11`, SHA256 `b2c836c52cdf063180b9ee76f67ac42946101b79ac457f3494035a67c090d961`.
Git HEAD: `c6dba2cf4610876db7d24361eb13fc9ea3ceca0c`; the working tree contains prior accepted source changes. Runtime source identity is pinned separately:

| Source | SHA256 |
| --- | --- |
| `src/mc_photogrammetry/cli.py` | `e2e584fed754e48757ed5b3916fbd365da0e9f1d47f8e65ffc9a5de118472377` |
| `src/mc_photogrammetry/pipeline/image_derived_session.py` | `e48414bf9eea6a7da7a7cc7d8c19865e0494953ac8796d9f3cd2e57e81b6248e` |
| `src/mc_photogrammetry/pipeline/image_derived_entry.py` | `1767b416061b20b1e60cd5c2d3d5b0814f232a3bc56d01328fc294818b20b6fc` |
| `src/mc_photogrammetry/pipeline/image_only.py` | `ca28afd0e9dc36d2b827a28807b736d635374acb2787506c2d2f3d07c929c93c` |
| `src/mc_photogrammetry/pipeline/image_derived_connections.py` | `b46d51a17f45a292666ef089199dca120cc11b91125d028e5114d0c6a010aedd` |
| `src/mc_photogrammetry/pipeline/image_derived_pose_graph.py` | `b80b75034ff133ab8dd25ea328081863c97abd7969c9d03e35b751e01447a74e` |
| `src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py` | `c894ef4cae49795e9b17f7cc10f2840fd853c00f3e049a151f0daa4bf02f4703` |
| `src/mc_photogrammetry/pipeline/image_derived_production.py` | `6f722046536fcb0040f689caa3c6c1ea7ffaef3cf37ee4d0c064b709cc31875e` |
| `src/mc_photogrammetry/export/validation.py` | `9e53d5ca8388b42eac19d8eee47e80fd97dbc162d8a0ba73c7282841fef331cb` |

## Preserved artifacts and records

All six files below are under `exports/diagnostics/post-camera-admission-normal-smoke-2026-10-07/`. Stdout/stderr/project remained unchanged during recovery.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `project.mcscan` | 10257 | `b829d311f5490bd7c20274e169082b31dbe2065fb88cb271f79968d63270acdc` |
| `run.initial.json` | 6952 | `606fc3298e72912cb538a210a76c247f5bd4de938db26b6b6a400bc0422878c7` |
| `run.json` | 11913 | `30f6b57da1424efad36cbb5b5ee1baa3a312e1cc98b98faea8ba8a023a9ba3f8` |
| `stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `stdout.txt` | 346 | `e6964163f73452013b105100e381f9bbb57780bd1dae7487f0fd7ce2fda51c26` |
| `supervisor-terminal.txt` | 1371 | `dc78c9e06b6b7eb0252c0e3fa2f75ca40eb8aaf8732d27510bd1703d1f20fba9` |

Result record: `agent swarm instantiation template/runs/RESULT-post-camera-admission-normal-smoke-2026-10-07.json`, SHA256 `8895713b9db99a49ffecfa22a76f8d7718cfaf96e15e2484b24d9a04f21e61c6`. It includes the exact supervisor script, retained heartbeat evidence, command/source/input bindings and unknown status fields. Packet status is `incomplete_telemetry_failure`, invocation consumed. Task status is ready for lead review, with this outcome disclosed.

## Limits and stop

Approximately nine Sol agent-minutes including preflight/run/recovery/reporting, below the 30 aggregate agent-minute task cap. No source or tests were edited; no environment installation, network upload/download, model/checkpoint/C1-pixel/truth/sidecar/JAR/sealed-fixture read, Minecraft/Prism/UI/client/capture, mouse/input/audio action or additional run occurred. Data is four unused train-split views from the same canal-town build as C1; it is not build-disjoint or holdout.

This preserves evidence that the camera stage admitted these four inputs and a later geometry gate refused them. It does not establish usable reconstruction, accuracy, generalization, C2 readiness, release acceptance or project completion. The telemetry failure leaves smoke acceptance incomplete. STOP; the one-shot is consumed and no retry is authorized.

