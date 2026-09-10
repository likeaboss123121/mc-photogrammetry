# Planned data pipeline and long-term appearance scope

Binding user clarification, 2026-09-09. Status: **PLANNED**, not implemented.
The immediate objective remains a vanilla-texture reconstruction proof of concept.
Continue the current foundational research and correctness assignments; this document
does not initiate scraping, training, dependency installation or a new implementation phase.

## Long-term product scope

- Reconstruct builds from screenshots and footage containing shaders and arbitrary resource packs.
- Accept a user-provided known resource pack or Minecraft JAR as an appearance asset input.
  Ordinary asset selection does not require developer mode. Camera coordinates, directions,
  bounds and other explicit geometry constraints still require advanced/developer opt-in.
- With a known pack, use its block mappings, textures and relevant models to help identify
  underlying block states. Keep appearance interpretation separate from exported state identity.
- With an unknown pack, support a closest-vanilla visual approximation with explicit uncertainty.
  Distinguish that result from recovery of the original block identity; a pack may make different
  blocks look identical or make one block resemble an unrelated vanilla block.
- Shader support includes appearance changes and temporal effects in footage. Preserve the
  distinction between observed surfaces and ambiguous/hidden geometry under those effects.
- Shaders, resource packs and expanded footage reconstruction are long-term requirements,
  not prerequisites for the initial vanilla proof of concept or reasons to loosen its gates.

## Planned generated-data pipeline

The user proposes using existing community maps/schematics and procedurally generated worlds
as a large source of known geometry. Dataset scale is an opportunity, not a verified inventory
or a substitute for data quality. Acquisition and rendering automation remain future work.

1. Import selected builds or generate worlds, preserving source/version/provenance and allowed
   usage information. Deduplicate builds and derived variants before splitting the dataset.
2. Capture the authoritative block states and geometry for the scene, with a litematic reference.
   Retain the surrounding scene needed to reproduce occlusion, lighting and shader effects.
3. Generate screenshots or camera trajectories varying relative position, pitch, yaw, FOV,
   gamma, lighting, viewpoint overlap, occlusion and distance. Later vary shaders and resource packs.
4. Save capture settings, asset versions/hashes, camera calibration and exact state labels in a
   separate training/evaluation record. None is silently supplied to production reconstruction.
5. Derive visible block/face labels for each frame and the union over selected frames. Keep
   the full scene reference as well as visible-area subsets; visibility is not a rectangular crop.
   Fix a visibility policy for transparency, cutouts and shader effects before evaluating those cases.
6. Split by source build/world family before rendering variants, so the same structure cannot
   leak across training and evaluation via another camera, shader or texture pack. Reserve
   unseen builds and appearance combinations, with fixed vanilla holdouts retained throughout.
7. Scale capture and learning only after ingestion, labels, coordinate transforms, asset/state
   mapping, evaluation, storage limits and reproducible execution pass independent checks.

## Preparation while the foundation is being built

- Keep appearance assets configurable and avoid hard-coding vanilla colour as block identity.
- Keep data provenance, camera labels, visibility labels and production inputs separate.
- Preserve exact states alongside appearance approximations and unknown-property alternatives.
- Test a small, manually audited capture batch before automated bulk acquisition or training.
- Actual Minecraft captures should be part of evaluation, so a recognizer is not validated only
  against images from the same approximate renderer whose errors it might learn.
- Choose the learning method after reliable data and baseline measurements exist. Reinforcement
  learning is not mandated by this scope clarification.

Current milestones, owners and acceptance decisions remain in [AGENT_HANDOFF.md](../AGENT_HANDOFF.md).
