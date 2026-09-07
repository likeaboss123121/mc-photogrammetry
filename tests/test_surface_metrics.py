"""Surface F1: does the reconstruction recover the cells a camera could see?

``coordinate_f1`` scores placed cells against every reference block, buried ones
included, so a stage that emits the region's bounding-box shell scores well on any
build that fills its own box. That is how a geometry stage which never worked hid
behind a 94.70% figure. These tests pin the honest measure and the failure it was
hiding.

"Exposed" means air beside the block that a camera could reach. Neither reading of
that on its own survives contact with the fixtures: air-inside-the-region misses
the boundary of an open build, and air-anywhere credits a reconstruction for the
outside of a mass whose photographs were all taken from a sealed room within it.
Flooding from the camera cells settles both.
"""
from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume
from mc_photogrammetry.export.compare import compare_litematics
from mc_photogrammetry.export.litematic import LitematicExporter
from mc_photogrammetry.reconstruction.occupancy import bounding_shell

SIDE = 7
CAVITY = range(2, 5)
STONE = BlockState("minecraft:stone")
STEPS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _cells(predicate):
    return {
        (x, y, z)
        for x in range(SIDE)
        for y in range(SIDE)
        for z in range(SIDE)
        if predicate((x, y, z))
    }


def _hollow():
    """A solid region with a cavity in the middle - a cave, in miniature."""
    return _cells(lambda c: not all(v in CAVITY for v in c))


def _sparse():
    """A build that does not fill its box - a small mass in an empty region."""
    return _cells(lambda c: all(v in CAVITY for v in c))


def _exposed(solid):
    """Blocks touching air inside the region: the no-camera fallback."""
    inside = lambda c: all(0 <= v < SIDE for v in c)
    return {
        c for c in solid
        if any(inside(n) and n not in solid
               for n in [(c[0] + s[0], c[1] + s[1], c[2] + s[2]) for s in STEPS])
    }


def _write(tmp_path, name, cells):
    volume = BlockVolume()
    for cell in cells:
        volume.set(cell, BlockObservation(STONE))
    path = tmp_path / f"{name}.litematic"
    LitematicExporter().export(volume, path, name=name)
    return path


def test_recovering_the_true_surface_scores_perfectly(tmp_path):
    solid = _hollow()
    reference = _write(tmp_path, "reference", solid)
    candidate = _write(tmp_path, "candidate", _exposed(solid))
    comparison = compare_litematics(reference, candidate)
    assert comparison.reference_exposed == len(_exposed(solid))
    assert comparison.surface_recall == 1.0
    assert comparison.surface_precision == 1.0
    assert comparison.surface_f1 == 1.0


def test_the_bounding_shell_scores_nothing_on_a_build_that_leaves_room(tmp_path):
    # The shell can only pass for an answer where the build fills its own box.
    # Where it does not - which is most fixtures - the shell and the visible
    # surface share almost nothing, and the honest measure says so.
    solid = _sparse()
    reference = _write(tmp_path, "reference", solid)
    shell = bounding_shell((SIDE, SIDE, SIDE))
    assert not (shell & solid)
    candidate = _write(tmp_path, "shell", shell)
    comparison = compare_litematics(reference, candidate)
    # No translation brings more than one wall of the shell onto the build.
    assert comparison.surface_f1 < 0.2


def test_a_camera_sealed_inside_a_mass_sees_only_its_own_room(tmp_path):
    # Template 4 in miniature: a nearly solid block, photographed from a pocket
    # within it. The outside of the mass is air, but no photograph contains it, so
    # crediting a reconstruction for reaching it would be crediting a guess.
    room = (3, 3, 3)
    solid = _cells(lambda c: c != room)
    reference = _write(tmp_path, "solid", solid)
    comparison = compare_litematics(reference, reference, camera_cells=[room])
    assert comparison.reference_exposed == len(STEPS)


def test_without_cameras_the_region_edge_alone_does_not_expose(tmp_path):
    # Nothing says which side of the boundary the photographer stood on, so the
    # measure stays with the conservative reading rather than inventing one.
    solid = _cells(lambda c: True)
    reference = _write(tmp_path, "solid", solid)
    comparison = compare_litematics(reference, reference)
    assert comparison.reference_exposed == 0


def test_a_buried_block_is_never_exposed(tmp_path):
    solid = _cells(lambda c: True)
    reference = _write(tmp_path, "solid", solid)
    corner = (0, 0, 0)
    comparison = compare_litematics(reference, reference, camera_cells=[corner])
    assert (3, 3, 3) not in _exposed(solid)
    assert comparison.reference_exposed < len(solid)


def test_surface_and_coordinate_scores_disagree_by_design(tmp_path):
    # A candidate made only of cells no camera could see scores on coordinates and
    # far less on surface, which is the disagreement the measure exists for.
    solid = _hollow()
    reference = _write(tmp_path, "reference", solid)
    candidate = _write(tmp_path, "buried", solid - _exposed(solid))
    comparison = compare_litematics(reference, candidate)
    assert comparison.coordinate_f1 > comparison.surface_f1
