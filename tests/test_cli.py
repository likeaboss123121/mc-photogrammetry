

def test_the_parser_builds_and_every_command_is_reachable():
    """A smoke test for the argument parser.

    `ADJACENCY_STRENGTH` was made the CLI default without importing it, which left
    every invocation raising `NameError` from `build_parser`. Nothing caught it,
    because the tests exercised the command bodies and never the parser itself.
    """
    from mc_photogrammetry.cli import build_parser

    parser = build_parser()
    names: set[str] = set()
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            names.update(choices)
    assert {"compare", "validate", "export"} <= names


def test_compare_accepts_the_attribution_flags():
    from mc_photogrammetry.cli import build_parser

    args = build_parser().parse_args(
        ["compare", "a.litematic", "b.litematic", "--attribute",
         "--cameras", "c.json", "--scene", "s.json", "--minecraft-jar", "v.jar"]
    )
    assert args.attribute is True
    assert args.cameras.name == "c.json"


def test_the_reconstruction_command_defaults_to_the_measured_adjacency():
    # 46: the library, the CLI and the benchmark script each defaulted to a
    # different value, and the CLI's was the worst of the three.
    from mc_photogrammetry.cli import build_parser
    from mc_photogrammetry.reconstruction.raycast import ADJACENCY_STRENGTH

    args = build_parser().parse_args([
        "reconstruct-visible-multiview", "images",
        "--cameras", "c.json", "--scene", "s.json",
    ])
    assert args.adjacency_strength == ADJACENCY_STRENGTH


def test_visible_multiview_exposes_opt_in_partial_shape_fitting():
    from mc_photogrammetry.cli import build_parser

    args = build_parser().parse_args([
        "reconstruct-visible-multiview", "images",
        "--cameras", "c.json", "--scene", "s.json", "--fit-partial-shapes",
    ])
    assert args.fit_partial_shapes is True
