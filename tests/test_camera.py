import pytest

from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    camera_projection_matrix,
    camera_space_points,
    load_camera_manifest,
    project_points,
    project_clipped_polygon,
    unproject_pixel,
)
from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume
from mc_photogrammetry.reconstruction.multiview import visible_blocks_for_camera
from mc_photogrammetry.reconstruction.multiview import reconstruct_visible_surfaces
from mc_photogrammetry.reconstruction.scene import load_scene_transform


def test_camera_projection_round_trips_a_world_point():
    camera = CameraModel(position=(0.0, 0.0, 0.0), yaw_degrees=0.0, pitch_degrees=0.0)
    point = (2.0, 1.0, 8.0)

    pixel = project_points([point], camera, width=1920, height=1080)[0]
    assert pixel is not None
    recovered = unproject_pixel(pixel[0], pixel[1], camera, width=1920, height=1080, depth=pixel[2])

    assert recovered == pytest.approx(point)


@pytest.mark.parametrize("width,height", [(1920, 1080), (1080, 1920), (2560, 1369)])
def test_sfm_projection_matrix_matches_the_shared_camera_projection(width, height):
    camera = CameraModel((2.0, 4.0, -3.0), yaw_degrees=-37.0, pitch_degrees=22.0)
    points = [(5.0, 5.0, 11.0), (-2.0, 1.0, 8.0)]
    matrix = camera_projection_matrix(camera, width=width, height=height)

    for point, direct in zip(points, project_points(points, camera, width=width, height=height)):
        assert direct is not None
        projected = matrix @ (*point, 1.0)
        assert projected[0] / projected[2] == pytest.approx(direct[0])
        assert projected[1] / projected[2] == pytest.approx(direct[1])
        assert projected[2] == pytest.approx(direct[2])


def test_camera_space_uses_positive_depth_for_a_point_in_front_of_the_camera():
    camera = CameraModel((0.0, 0.0, 0.0), 0.0, 0.0, eye_height=0.0)
    view = camera_space_points([(0.0, 0.0, 5.0)], camera)
    assert tuple(view[0]) == pytest.approx((0.0, 0.0, 5.0))


def test_clipped_polygon_keeps_the_front_part_of_a_near_plane_crossing_face():
    camera = CameraModel((0.0, 0.0, 0.0), 0.0, 0.0, eye_height=0.0)
    polygon = project_clipped_polygon(
        [(-1.0, -1.0, -1.0), (1.0, -1.0, 2.0), (1.0, 1.0, 2.0), (-1.0, 1.0, -1.0)],
        camera, width=256, height=256,
    )
    assert polygon.shape[0] >= 3
    assert float(polygon[:, 2].min()) >= 1e-3 - 1e-12


def test_camera_rejects_invalid_fov():
    with pytest.raises(ValueError, match="fov"):
        CameraModel((0.0, 0.0, 0.0), 0.0, 0.0, fov_degrees=180.0)


def test_camera_manifest_uses_player_feet_and_eye_height():
    camera = CameraModel.from_dict(
        {"position": [1, 2, 3], "yaw_degrees": 0, "pitch_degrees": 0, "fov_degrees": 110}
    )

    assert camera.eye_height == pytest.approx(1.62)


def test_camera_manifest_loads_template_poses():
    cameras = load_camera_manifest("templates/cameras.json")

    assert len(cameras) == 9
    assert cameras["2026-09-02_12.17.15.png"].fov_degrees == 110.0
    assert cameras["2026-09-02_12.17.38.png"].yaw_degrees == -179.6


def test_minecraft_yaw_minus_90_looks_toward_positive_x():
    camera = CameraModel((0.0, 0.0, 0.0), yaw_degrees=-90.0, pitch_degrees=0.0)
    projected = project_points([(8.0, 0.0, 0.0)], camera, width=100, height=100)[0]

    assert projected is not None
    assert projected[0] == pytest.approx(50.0)


def test_screen_right_is_west_when_facing_south():
    """Minecraft's yaw 0 faces +Z, which puts +X (east) on the left of frame.

    Screen right is forward x up = (-cos, 0, -sin). Dropping that sign mirrors
    every render left to right, which still looks like a plausible room and so
    survives block-level checks while silently sampling the wrong blocks.
    """
    camera = CameraModel((0.0, 0.0, 0.0), yaw_degrees=0.0, pitch_degrees=0.0, eye_height=0.0)

    east, west = project_points(
        [(4.0, 0.0, 8.0), (-4.0, 0.0, 8.0)], camera, width=100, height=100
    )

    assert east is not None and west is not None
    assert east[0] < 50.0 < west[0]


def test_yaw_minus_90_faces_east_and_puts_south_on_the_right():
    camera = CameraModel((0.0, 0.0, 0.0), yaw_degrees=-90.0, pitch_degrees=0.0, eye_height=0.0)

    ahead, south = project_points(
        [(8.0, 0.0, 0.0), (8.0, 0.0, 4.0)], camera, width=100, height=100
    )

    assert ahead is not None and south is not None
    assert ahead[0] == pytest.approx(50.0)
    # Facing east, right is south: the two rotations agree on one handedness.
    assert south[0] > 50.0


def test_calibrated_projection_keeps_frontmost_block_per_pixel():
    volume = BlockVolume()
    volume.set((0, 0, 5), BlockObservation(BlockState("minecraft:stone")))
    volume.set((0, 0, 8), BlockObservation(BlockState("minecraft:dirt")))
    camera = CameraModel((0.0, 0.0, 0.0), 0.0, 0.0, fov_degrees=110.0, eye_height=0.0)

    visible = visible_blocks_for_camera(
        volume, camera, width=100, height=100, frame_id="frame"
    )

    assert len(visible) == 1
    assert visible[0].coordinate == (0, 0, 5)


def test_template_scene_transform_maps_local_litematic_bounds():
    transform = load_scene_transform("templates/scene.json")

    assert transform.world_origin == (-236, 45, 248)
    assert transform.size == (161, 47, 180)
    assert transform.to_world((160, 46, 179)) == (-76, 91, 427)
    assert transform.to_local((-76, 91, 427)) == (160, 46, 179)


def test_template_cameras_see_the_reference_build():
    from mc_photogrammetry.export.compare import load_litematic

    reference = load_litematic("templates/Photogrammetry Template.litematic")
    transform = load_scene_transform("templates/scene.json")
    cameras = load_camera_manifest("templates/cameras.json")
    world_points = [transform.to_world(coordinate) for coordinate in reference.blocks]

    for camera in cameras.values():
        projected = project_points(world_points, camera, width=2560, height=1369)
        visible = [
            point
            for point in projected
            if point is not None and 0 <= point[0] < 2560 and 0 <= point[1] < 1369
        ]
        assert len(visible) > 10_000


def test_compact_template_cameras_see_the_reference_build():
    from mc_photogrammetry.export.compare import load_litematic

    reference = load_litematic("templates 2/Template 2.litematic")
    transform = load_scene_transform("templates 2/scene.json")
    cameras = load_camera_manifest("templates 2/cameras.json")
    world_points = [transform.to_world(coordinate) for coordinate in reference.blocks]

    assert reference.size == (9, 12, 17)
    assert len(reference.palette) == 27
    for camera in cameras.values():
        projected = project_points(world_points, camera, width=2560, height=1369)
        visible = [
            point
            for point in projected
            if point is not None and 0 <= point[0] < 2560 and 0 <= point[1] < 1369
        ]
        assert len(visible) > 100
