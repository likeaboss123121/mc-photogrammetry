"""Closed native boundaries. Import is stdlib-only and performs no I/O.

Real services activate only in a privately authenticated, job-contained worker.
The synthetic task supplies fake services and never activates the real loader.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import ntpath
import os
from pathlib import Path
import re
import stat
import struct
import zlib

MANIFEST_CAP = 262144
DIAGNOSTIC_CAP = 1048576
IMAGE_CAP = 8388608
SMALL_CAP = 65536
LIMITS = dict(wall_seconds=180, committed_bytes=2147483648, log_bytes=65536,
    comparisons=1048576, tuples=4096, determinants=32768, footprint_predicates=16800,
    footprint_operands=77414400, extraction_windows=4352, extraction_pixels=10027008,
    gradient_sites=7676928, derivation_elements=1392640, distance_matrix_bytes=8388608,
    component_contributions=335544320, evidence_bytes=16777216, model_bytes=10092544,
    publication_bytes=20381696, aggregate_bytes=37748736, small_bytes=65536)
LABELS = ("row176", "row197", "shared-right")
ENDPOINTS = (("image-0000", 2), ("image-0000", 3), ("image-0001", 1))
SOURCE_PINS = (
    ("scripts/evaluation/spatial_context_descriptor.py", "066147b78b35ff4a9da84a433463394b6fb711431fe902135e34c6c8be056c42"),
    ("scripts/evaluation/spatial_context_evidence.py", "825d93f8660f34c09a022ea229d9173e83b984e5186b002494848e4c5efa046d"))
HISTORICAL_PATHS = (
    "src/mc_photogrammetry/reconstruction/lattice_planes.py",
    "src/mc_photogrammetry/pipeline/image_derived_signatures.py",
    "src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py",
    "src/mc_photogrammetry/pipeline/image_derived_endpoint_selection.py",
    "src/mc_photogrammetry/pipeline/image_derived_joint_graph.py",
    "scripts/evaluation/image_point_candidate_roster.py",
    "scripts/evaluation/diagnose_image_geometry_failure.py")
MASK_LIMITS = dict(mask_pixels=4000000, aggregate_pixels=12000000,
                   aggregate_packed_bytes=524288, bundle_bytes=1048576)
SOURCE_KEYS = ("frame_id", "endpoint", "region_id", "encoded_sha256", "encoded_length",
    "decoded_shape", "decoded_dtype", "decoded_rgb_sha256", "ingress_sha256",
    "packed_mask_sha256", "full_native_bool_sha256", "usable_mask_sha256", "usable_helper_sha256")
ENVIRONMENT_KEYS = ("SystemRoot", "TEMP", "TMP", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
SEED_PINS = {
    "scripts.evaluation.probe_spatial_context_ambiguity": "c6d591fe7828b85e89c62efc8137efb77aa07d7a4281346755c44032cbac884b",
    "scripts.evaluation.spatial_context_descriptor": SOURCE_PINS[0][1],
    "scripts.evaluation.spatial_context_evidence": SOURCE_PINS[1][1],
    "scripts.evaluation.diagnose_image_geometry_failure": "5c3bdb9ed5dc34b021f3e60c92d0b7cd5ffa2aba8aea35225354213107d0d2af",
    "mc_photogrammetry.reconstruction.lattice_planes": "c416bea087eb1c9228fa0b17f78fb8177b49e711612e4322af989aa15e65995b",
    "mc_photogrammetry.input.image": "43c318a45bc6965b6a9d3d8f43fc2375052cbe5ad2b13b07a7b7b4313f0bf732",
    "mc_photogrammetry.input.media": "8f872b165283fb65a01e7470062cb51040feeee46ecb25ed26cdb0b67b7c2f89",
    "mc_photogrammetry.pipeline.image_derived_session": "e48414bf9eea6a7da7a7cc7d8c19865e0494953ac8796d9f3cd2e57e81b6248e",
    "mc_photogrammetry.pipeline.correspondence": "192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1",
    "mc_photogrammetry.vision.gui_mask": "1d8a6d9ed9537a2e5dd98c794aa9c45b07c748090e1a17f9db3d230e5e74cda9",
    "mc_photogrammetry.pipeline.image_derived_joint_graph": "60f4f55bc887bef50c60427a00fca54f5ad854a1785ebd5b1319c4b823df25ab",
    "mc_photogrammetry.pipeline.image_derived_endpoint_selection": "f8065946eaedb95cdbb8d2bfceace0246bf1369d2f8683f2fd555aa1904a7c0a",
    "mc_photogrammetry.pipeline.image_derived_entry": "1767b416061b20b1e60cd5c2d3d5b0814f232a3bc56d01328fc294818b20b6fc",
}


# Exact 83-node recursive ten-seed closure, including package initializers.
REPOSITORY_PINS = {'mc_photogrammetry': 'e54953216fa986cd12def3b31c30d8e200c26378057e006117f33f49a81b192f', 'mc_photogrammetry.blocks': 'd791f35c103aee8a9561f039a1f2a8493e21014692e9329fe84628404a332069', 'mc_photogrammetry.blocks.appearance': '2380f1676e713d340432dfa28f2146b924206157bbb41f79f8c39ed897c621b8', 'mc_photogrammetry.blocks.model_atlas': 'f53e8878c55fc902d76ef79816e92f0a3e70e2a20eb26458f1e81ae697403c54', 'mc_photogrammetry.blocks.textures': 'f735ae866d949bdcf6f24c90b2a604b4c2b8a0b2d5661561aa9f4ff47c37cc45', 'mc_photogrammetry.domain': 'f223d7149f7c3a3f320f7b33705820f77288f9fd256a4c284c520cf20f357d35', 'mc_photogrammetry.domain.models': '6a87d543661cc62ada9a154bcdaa089ca844f41691ff3b53207ad89e3f05a478', 'mc_photogrammetry.export': '259fac03dcbb676b1f518aa515d7f38b26efd499071c0f29b0db232fb07e1b3e', 'mc_photogrammetry.export.attribution': '4e053555f215c7305fa384775e5186be2f93d6d03c0bc9b4cb210180a6a47f96', 'mc_photogrammetry.export.compare': '6d1cc9ef3bc7d5e7dc08ef2a7fb269cb4bf7e3ee0f4cc2e01114f47625d2e738', 'mc_photogrammetry.export.litematic': '1917451e46ccb451dcbf2e5194161edfbea0862dc55e62489fcdeeb14f3db794', 'mc_photogrammetry.export.validation': '9e53d5ca8388b42eac19d8eee47e80fd97dbc162d8a0ba73c7282841fef331cb', 'mc_photogrammetry.input': '14ee634d9fab59deffdc5c48b800fd1619773909a26213ae487f8c043855e64f', 'mc_photogrammetry.input.frames': 'ed9b166d29887ef849d3d601fdef89306fccac4d40cae4d39a48a4a6ff290d8f', 'mc_photogrammetry.input.image': '43c318a45bc6965b6a9d3d8f43fc2375052cbe5ad2b13b07a7b7b4313f0bf732', 'mc_photogrammetry.input.media': '8f872b165283fb65a01e7470062cb51040feeee46ecb25ed26cdb0b67b7c2f89', 'mc_photogrammetry.input.video': '8265ada8f8d9153d54b94365fb50a643a8c892edbb0ab21176fe9af01d629d04', 'mc_photogrammetry.pipeline': '28dd13121ddce9923020ab7493c416e4146520e4f734f8588d02389008312f32', 'mc_photogrammetry.pipeline.candidate_support_provenance': 'a8bd3102652720513ed85e2d3b7e2c81a79a01a7621c41238b09f6e741604562', 'mc_photogrammetry.pipeline.candidate_support_provenance_v2': 'f5cee3cfe9d0247069637992fb29d1f9a8533abc9383af841cfc9a726313ccd2', 'mc_photogrammetry.pipeline.connected_camera': '3b5df735084daddccb7f25511af1b8549c0300d02e4841ade39e39461b8275a6', 'mc_photogrammetry.pipeline.contracts': 'db24e730bfed52ec983ae387a26f1b612fd94abc3e7608053080915c899ed5cc', 'mc_photogrammetry.pipeline.correspondence': '192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1', 'mc_photogrammetry.pipeline.frame_ingress': '3411b4067281b19a6ac2e530ece5dfc2e53e0f63b978e83e09ffadb27f715f0a', 'mc_photogrammetry.pipeline.image_derived_connections': 'b46d51a17f45a292666ef089199dca120cc11b91125d028e5114d0c6a010aedd', 'mc_photogrammetry.pipeline.image_derived_endpoint_selection': 'f8065946eaedb95cdbb8d2bfceace0246bf1369d2f8683f2fd555aa1904a7c0a', 'mc_photogrammetry.pipeline.image_derived_entry': '1767b416061b20b1e60cd5c2d3d5b0814f232a3bc56d01328fc294818b20b6fc', 'mc_photogrammetry.pipeline.image_derived_graph': '7549c7177a82f8ef452ae9fa4581f6a443d290a5e3ae9af9403de5335eb3a6b6', 'mc_photogrammetry.pipeline.image_derived_graph_attribution': '085ff09c2747add670362db810789776b6459a9cf9e7260b1c2f52e1d0c563f0', 'mc_photogrammetry.pipeline.image_derived_graph_v3': '34ad9aaac65ebc42d0a87223a9da75a4b0a243d70a96a00cc81119bff00b3546', 'mc_photogrammetry.pipeline.image_derived_joint_graph': '60f4f55bc887bef50c60427a00fca54f5ad854a1785ebd5b1319c4b823df25ab', 'mc_photogrammetry.pipeline.image_derived_lattice_measure': 'fb061bbfaa14bfb969e4349f1a01c080ee338a558d3cd6521425eb1d8231ed8e', 'mc_photogrammetry.pipeline.image_derived_pose_graph': 'b80b75034ff133ab8dd25ea328081863c97abd7969c9d03e35b751e01447a74e', 'mc_photogrammetry.pipeline.image_derived_production_record': '5b70f1598fb7338da5167fe961199eda7f12ee189228acd7ad08a96fec28cd2f', 'mc_photogrammetry.pipeline.image_derived_session': 'e48414bf9eea6a7da7a7cc7d8c19865e0494953ac8796d9f3cd2e57e81b6248e', 'mc_photogrammetry.pipeline.image_derived_shift_domain_policy': '40453154b8b5da2a28d91fd004a37a78d2bb033829a45084ddc528e453c3e4e4', 'mc_photogrammetry.pipeline.image_derived_signatures': '7e287aecb6c1e6bb111c5d5a6c33abc876b6083ce48abdf3c4fff1306819e43f', 'mc_photogrammetry.pipeline.image_only': 'ca28afd0e9dc36d2b827a28807b736d635374acb2787506c2d2f3d07c929c93c', 'mc_photogrammetry.pipeline.joint_matching': '9303f56ab194600c58a27d97321fa2f2e4323e4bb78276a2df0372e8ed8ae4e3', 'mc_photogrammetry.project': '3cf716d6d4b09e1a26420b9557b07fe26325eac9a1d326494cb72f20decd70be', 'mc_photogrammetry.project.store': 'b646084f746df0e7c213d5723f01f156992b23997400476796d7ee0187002b41', 'mc_photogrammetry.reconstruction': '7f80f67bfe5ce8f93a4cf140f093e5d23458b6b194814a2c0a8d2973f99c5d09', 'mc_photogrammetry.reconstruction.appearance_scoring': '84f767fb781bebb08c39cb45fa0ec41990655b3562dcfa090ff8f4defda9ed02', 'mc_photogrammetry.reconstruction.appearance_state_proposals': '39ecfc70415472a7770a76fe492cc251d42233a43a557ecbf60305ae8846876a', 'mc_photogrammetry.reconstruction.boundary_candidates': '95e550a44bd65a19044b2a35a99d9a503b389d066085dccb464b0f18fa89216c', 'mc_photogrammetry.reconstruction.calibration': '2ee1690a817a59f8b6ffd7858b65ce9d36bb01a6506577261b267c79e8912bfc', 'mc_photogrammetry.reconstruction.camera': '0515c5bd82f01d3dc174ced6cb54405adef0a755b94c577dae513fc53c4fe680', 'mc_photogrammetry.reconstruction.endpoint_hypothesis': 'f197d8242148ebd3b49b2a5f01fa2bab117731ef01f51a36423da3ca8ee2f08c', 'mc_photogrammetry.reconstruction.facade': 'c324946e27c604f62f3397af7d8f331febfdd43fa8fa8a757a0277e64344a6cf', 'mc_photogrammetry.reconstruction.face_observation': '03710aaec58d1e2fffb769174db34f826c7d8dd830bedb8d9074bd231bda8c1e', 'mc_photogrammetry.reconstruction.face_sampling': '14be0f406364c2175db84aa4909825a2d059511b6d39c0c33debf4b3d2ecba6e', 'mc_photogrammetry.reconstruction.grid': '401888823cc50db2112f8b78ab5a91f3ca35010bf65bfda51ecbec61f2e0a266', 'mc_photogrammetry.reconstruction.lattice': '76a0c0f6a0bc057fbbc75216b7dbcdfef637d69f7e484d92ba373c3a6eb45c9b', 'mc_photogrammetry.reconstruction.lattice_planes': 'c416bea087eb1c9228fa0b17f78fb8177b49e711612e4322af989aa15e65995b', 'mc_photogrammetry.reconstruction.lighting': '7e99b691f258d6fca055b4620309e069c0be700b8ca20b8f8d6fd6d8c9f5c148', 'mc_photogrammetry.reconstruction.multiview': 'acfc0e4c4e6b45102a3911c19b6e4d6ec86c455428eea1a2ad19af8c5b287d8a', 'mc_photogrammetry.reconstruction.occupancy': '81b32acfc4cf370f303a67d83dcab9fff2a246d7a5d0733c80b973f212fba59b', 'mc_photogrammetry.reconstruction.plane_axis_support': '18598afb2096b504464411e3a2cbf6a2e960cb2601c4dd2fb8f77d4b46c91f3f', 'mc_photogrammetry.reconstruction.plane_endpoint_provenance': '3cb6aacea4d9e6873d0b10edd4ea7539b1646a1e75475cef2006f81caa3316e7', 'mc_photogrammetry.reconstruction.plane_geometry_provenance': '1d51ffb706be2826af7bd9393bec8537f2bb9499937d71baec49c100338c8605', 'mc_photogrammetry.reconstruction.plane_graph': '6868ac42c657c2ca270e9d91a36c4f7958e2ad4ad9014b391bff415ce4a646a8', 'mc_photogrammetry.reconstruction.plane_phase_anchor': '36d8d24541a856d97476ae440ee37fb43173672cfff18340538ca92b56bdeefc', 'mc_photogrammetry.reconstruction.plane_placement': '743fd95a18651a342642ced43701803cda379a2d392604557dc6069ba3909880', 'mc_photogrammetry.reconstruction.plane_region_support': '45a5b02947e3f95ee5d160334b5172e3a547b16c5893b9c757f964839d8896e3', 'mc_photogrammetry.reconstruction.plane_relative_translation': '6b0d971e1badc4ae25001850670d248d3bd19f342cc4605c0513b3cc758ce113', 'mc_photogrammetry.reconstruction.plane_residue': 'a3bc4365aeefb0773ff020685825c1d4e2e21650aaa0515dfd03c9f1280d87a4', 'mc_photogrammetry.reconstruction.plane_shift_domain': 'dabef9f32c8b4591dbdb92bee109fbd1a4d7b32b75cd6def6a1b89cf2cfec5df', 'mc_photogrammetry.reconstruction.plane_vector_domain': 'fd60bda08d8304eb378dc7eb568a6ea6365031e69ccd9bd4c3590e01f92411f0', 'mc_photogrammetry.reconstruction.plane_volume': '872b6e5317f7cf0f2c729ccb57f0ad8c52368290397facb0eb34b37eed25a7bd', 'mc_photogrammetry.reconstruction.raycast': '44864cbeb56ae1d171acb9cb4c95256c9c456896fe4127f676010a550eb88ceb', 'mc_photogrammetry.reconstruction.region_growth': 'be486f11c46f31087656e9b2ca5a382688e95f8bef20db25c7bab03d095aa214', 'mc_photogrammetry.reconstruction.residual': '44a0d05afac68d11822bb8dbf1ed5e50fe97ae7e2557f1eb74d7680ad846791b', 'mc_photogrammetry.reconstruction.scene': 'cd72ead7f553c0157bbdf80b3651e5c6d31faf379af068f3ae235a5e4e47e53c', 'mc_photogrammetry.reconstruction.sfm': '256cf5a882bb1b535838c638c16b2e4290bd802531e72cc3db1bf7bab459a181', 'mc_photogrammetry.reconstruction.shape_fit': 'cb08402f5b0230a615fab70b57982b539b3f361c9269f4197554a479b8ad0b97', 'mc_photogrammetry.reconstruction.state_promotion': '65e361ebdc0efaa91a38ff15ddf2780f49791b536b3caa73a41575b8fe5a9824', 'mc_photogrammetry.reconstruction.stereo': '7dfef297603b317075134d6f60c1e345ae796acb3e92a18dc04f6d172eab9c38', 'mc_photogrammetry.reconstruction.supported': '4c94564d3842f1c3a03883f6f95a78a4ac08260150eb9fc12251bf708e35f29c', 'mc_photogrammetry.reconstruction.surfaces': '888ebf1ab837baccd845b85256fb2cc7b2cf518352c371eac58fbc48264ec11c', 'mc_photogrammetry.reconstruction.texture_calibration': '15b987e645463a84b0e021d3f2181f89bb1d6db83fa4225a9f07f4bfec117d4a', 'mc_photogrammetry.vision': '10a37a6abd7ba1f3ac3e58194a08500a9b01795206e62866d5e7f566871b0ce6', 'mc_photogrammetry.vision.gui_mask': '1d8a6d9ed9537a2e5dd98c794aa9c45b07c748090e1a17f9db3d230e5e74cda9', 'scripts.evaluation.diagnose_image_geometry_failure': '5c3bdb9ed5dc34b021f3e60c92d0b7cd5ffa2aba8aea35225354213107d0d2af'}


class Refusal(ValueError):
    pass


def require(ok, code):
    if not ok:
        raise Refusal(code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def integer(value, ceiling=2 ** 63 - 1, minimum=0):
    require(type(value) is int and minimum <= value <= ceiling, "integer_bound")


def text(value, maximum=128):
    require(type(value) is str and 0 < len(value) <= maximum and value.isascii(), "text_bound")


def digest(value):
    require(type(value) is str and len(value) == 64
            and re.fullmatch("[0-9a-f]{64}", value) is not None, "digest")


def keys(value, expected):
    require(type(value) is dict and len(value) == len(expected)
            and all(type(k) is str and len(k) <= 256 and k in expected for k in value), "keys")


def sequence(value, size):
    require(type(value) is list and len(value) == size, "cardinality")


def windows_path(value):
    text(value, 1024)
    require(len(value) >= 3 and value[0].isascii() and value[0].isalpha()
            and value[1:3] == ":\\" and value.count(":") == 1
            and "/" not in value and "%" not in value and "\x00" not in value
            and ntpath.normpath(value) == value, "absolute_path")
    require(all(part not in ("", ".", "..") and not part.endswith((".", " "))
                for part in value[3:].split("\\")), "path_component")
    reserved = {"con", "prn", "aux", "nul", *("com"+str(i) for i in range(1,10)), *("lpt"+str(i) for i in range(1,10))}
    require(all(part.split(".")[0].casefold() not in reserved for part in value[3:].split("\\")), "device_path")
    return ntpath.normcase(value)


def bound_object(value, *, cap, depth=24, items=262144, strings=1024, packed=False):
    """No serialization/copy until every container/scalar has been capped."""
    stack = [(value, 0, "")]
    count = lower = 0
    while stack:
        item, level, field = stack.pop()
        count += 1
        require(count <= items and level <= depth, "object_structure")
        if type(item) is dict:
            require(len(item) <= min(256, items - count - len(stack)), "object_fields")
            for key, child in item.items():
                text(key, 256)
                lower += len(key) + 3
                stack.append((child, level + 1, key))
        elif type(item) is list:
            require(len(item) <= min(8704, items - count - len(stack)), "object_list")
            stack.extend((child, level + 1, field) for child in item)
        elif type(item) is str:
            maximum = 699052 if packed and field == "packed_mask_base64" else strings
            text(item, maximum)
            lower += len(item) + 2
        elif type(item) is int:
            require(item.bit_length() <= 64, "integer_width")
            lower += 21
        elif type(item) is float:
            require(math.isfinite(item), "nonfinite")
            lower += 24
        else:
            require(item is None or type(item) is bool, "object_type")
            lower += 5
        require(lower + count <= cap, "object_bytes")


def canonical(value, cap=MANIFEST_CAP):
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False,
                         sort_keys=True, separators=(",", ":")).encode("ascii")
    require(len(encoded) <= cap, "canonical_bytes")
    return encoded


def parse(data, *, cap, depth=24, items=262144, strings=1024, packed=False):
    require(type(data) is bytes and 0 < len(data) <= cap and data.isascii(), "json_bytes")
    # Lexical depth/integer/string guards precede json.loads allocations.
    level = tokens = pos = 0
    while pos < len(data):
        char = data[pos]
        if char == 34:
            start = pos
            pos += 1
            while pos < len(data):
                if data[pos] == 92:
                    pos += 2
                    continue
                if data[pos] == 34:
                    break
                pos += 1
            require(pos < len(data), "json_string")
            require(pos - start <= (699052 if packed else strings * 6) + 2, "json_string")
            tokens += 1
        elif char in (123, 91):
            level += 1
            tokens += 1
            require(level <= depth, "json_depth")
        elif char in (125, 93):
            level -= 1
            require(level >= 0, "json_depth")
        elif char in b"-0123456789":
            start = pos
            while pos + 1 < len(data) and data[pos + 1] in b"-+0123456789.eE":
                pos += 1
            require(pos - start <= 32, "json_number")
            tokens += 1
        elif char in (116, 102, 110):
            tokens += 1
        require(tokens <= items, "json_tokens")
        pos += 1
    require(level == 0, "json_depth")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate_key")
            result[key] = value
        return result
    def constant(_):
        raise Refusal("nonfinite")
    try:
        value = json.loads(data, object_pairs_hook=unique, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, Refusal):
            raise
        raise Refusal("json_parse") from None
    bound_object(value, cap=cap, depth=depth, items=items, strings=strings, packed=packed)
    return value


def sealed(value, field, cap=MANIFEST_CAP):
    digest(value[field])
    require(value[field] == sha(canonical({k: v for k, v in value.items() if k != field}, cap)), "self_seal")


def source(record):
    keys(record, SOURCE_KEYS)
    text(record["frame_id"])
    sequence(record["endpoint"], 2)
    require(record["endpoint"][0] == record["frame_id"], "endpoint")
    integer(record["endpoint"][1], 65535)
    sequence(record["decoded_shape"], 3)
    for dimension in record["decoded_shape"]:
        integer(dimension, 1600, 1)
    require(record["decoded_shape"] == [900, 1600, 3] and record["decoded_dtype"] == "|u1", "native_shape")
    integer(record["encoded_length"], IMAGE_CAP, 1)
    for key in SOURCE_KEYS:
        if key.endswith("sha256") or key == "region_id":
            digest(record[key])


def template(record):
    keys(record, ("bindings", "runtime", "metadata", "row176", "diagnostic_sha256", "diagnostic_length",
                  "label_associations", "feasibility_sha256", "dependency_pins", "limits"))
    bindings = record["bindings"]
    keys(bindings, ("sources", "source_pins", "diagnostic_outer_sha256", "diagnostic_nested_sha256",
                   "contract_sha256", "source_sha256", "runtime_sha256"))
    sequence(bindings["sources"], 2)
    require(bindings["source_pins"] == [list(row) for row in SOURCE_PINS], "source_pins")
    require(record["limits"] == LIMITS, "limits")
    for item in [record["row176"], *bindings["sources"]]:
        source(item)
    require([tuple(s["endpoint"]) for s in [record["row176"], *bindings["sources"]]] == list(ENDPOINTS), "endpoints")
    for key in ("diagnostic_sha256", "feasibility_sha256"):
        digest(record[key])
    integer(record["diagnostic_length"], DIAGNOSTIC_CAP, 1)
    # Close every remaining nested shape without loading the coordinator.
    keys(record["dependency_pins"], ("descriptor", "evidence"))
    require(record["dependency_pins"] == dict(descriptor=SOURCE_PINS[0][1], evidence=SOURCE_PINS[1][1]), "dependency_pins")
    runtime_keys = ("python", "numpy", "platform", "implementation", "python_sha256", "numpy_sha256", "platform_sha256", "implementation_sha256")
    require(type(record["runtime"]) is dict and set(record["runtime"]) <= set(runtime_keys), "runtime")
    for key, value in record["runtime"].items():
        text(value)
        if key.endswith("sha256"):
            digest(value)
    metadata = record["metadata"]
    keys(metadata, ("expected_supply", "feasibility_identities", "reservation", "dependency_pins", "source_pins",
        "contract_sha256", "source_sha256", "source_inventory", "label_associations", "inventory_sha256",
        "diagnostic_payload_sha256", "diagnostic_outer_sha256", "diagnostic_nested_sha256", "runtime_sha256"))
    sequence(metadata["expected_supply"], 3)
    for count in metadata["expected_supply"]:
        integer(count, 4352)
    require(metadata["expected_supply"] == [0, 37, 2570] and metadata["feasibility_identities"] == list(LABELS), "supply")
    require(metadata["source_inventory"] == [record["row176"], *bindings["sources"]], "inventory")
    require(all(record["row176"][key] == bindings["sources"][0][key] for key in
                ("encoded_sha256", "encoded_length", "decoded_rgb_sha256", "ingress_sha256", "usable_mask_sha256", "usable_helper_sha256")), "shared_left_identity")
    associations = [dict(label=label, frame_id=s["frame_id"], endpoint=s["endpoint"], region_id=s["region_id"])
                    for label, s in zip(LABELS, metadata["source_inventory"])]
    require(record["label_associations"] == metadata["label_associations"] == associations, "associations")
    keys(metadata["reservation"], ("plans", "grid_positions", "boolean_operands", "extractions", "extraction_windows",
        "extraction_pixels", "gradient_sites", "derivation_elements", "comparisons_per_arm", "comparisons_total",
        "distance_matrix_bytes", "component_contributions", "tuples_per_arm", "determinants_per_arm", "tuples_total",
        "determinants_total", "model_bytes", "publication_bytes", "wall_seconds", "committed_bytes", "log_bytes"))
    for value in metadata["reservation"].values():
        integer(value)
    expected = dict(plans=3, grid_positions=16800, boolean_operands=77414400, extractions=2,
        extraction_windows=2607, extraction_pixels=6006528, gradient_sites=4598748,
        derivation_elements=834240, comparisons_per_arm=95090, comparisons_total=190180,
        distance_matrix_bytes=760720, component_contributions=30428800, tuples_per_arm=4096,
        determinants_per_arm=32768, tuples_total=8192, determinants_total=65536,
        model_bytes=10092544, publication_bytes=20381696, wall_seconds=180,
        committed_bytes=2147483648, log_bytes=65536)
    require(metadata["reservation"] == expected, "reservation")
    require(bindings["contract_sha256"] == "c14d8ce72fc7131ea6b17b7c26df735a9475d2bec5eacd9b6e04bcf0f9391bcb"
            and bindings["source_sha256"] == SOURCE_PINS[0][1]
            and record["feasibility_sha256"] == "29e4fb299158377fd791b0f24185dce8c6b6f55df9650c691942a863d58f46fb", "accepted_contract")
    require(metadata["dependency_pins"] == record["dependency_pins"]
            and metadata["source_pins"] == bindings["source_pins"], "metadata_pins")
    for key in ("contract_sha256", "source_sha256", "runtime_sha256", "diagnostic_outer_sha256", "diagnostic_nested_sha256"):
        require(metadata[key] == bindings[key], "metadata_binding")
    require(metadata["diagnostic_payload_sha256"] == record["diagnostic_sha256"], "metadata_diagnostic")
    require(bindings["runtime_sha256"] == sha(canonical(record["runtime"])), "runtime_seal")
    require(metadata["inventory_sha256"] == sha(canonical(dict(sources=metadata["source_inventory"], label_associations=associations))), "inventory_seal")
    for key in metadata:
        if key.endswith("sha256"):
            digest(metadata[key])
    for key in bindings:
        if key.endswith("sha256"):
            digest(bindings[key])


class Manifest:
    def __init__(self, data, expected_file_sha256):
        digest(expected_file_sha256)
        require(type(data) is bytes and len(data) <= MANIFEST_CAP and sha(data) == expected_file_sha256, "manifest_external")
        record = parse(data, cap=MANIFEST_CAP)
        keys(record, ("schema", "attempt_id", "request_template", "files", "provenance", "runtime", "launch", "artifacts", "limits", "manifest_sha256"))
        require(record["schema"] == "spatial-context-native-manifest-v1", "manifest_schema")
        text(record["attempt_id"])
        require(re.fullmatch("[A-Za-z0-9_-]+", record["attempt_id"]) is not None, "attempt_id")
        template(record["request_template"])
        require(record["limits"] == LIMITS, "limits")
        files = record["files"]
        keys(files, ("diagnostic", "frames"))
        paths = []
        def file_pin(pin, cap):
            keys(pin, ("path", "bytes", "sha256"))
            paths.append(windows_path(pin["path"]))
            integer(pin["bytes"], cap, 1)
            digest(pin["sha256"])
        file_pin(files["diagnostic"], DIAGNOSTIC_CAP)
        request = record["request_template"]
        require((files["diagnostic"]["bytes"], files["diagnostic"]["sha256"])
                == (request["diagnostic_length"], request["diagnostic_sha256"]), "diagnostic_file_join")
        sequence(files["frames"], 2)
        for pin, src in zip(files["frames"], request["bindings"]["sources"]):
            keys(pin, ("frame_id", "path", "bytes", "sha256", "png_width", "png_height", "exif_orientation"))
            file_pin({k: pin[k] for k in ("path", "bytes", "sha256")}, IMAGE_CAP)
            require((pin["frame_id"], pin["bytes"], pin["sha256"]) == (src["frame_id"], src["encoded_length"], src["encoded_sha256"]), "frame_file_join")
            for key in ("png_width", "png_height"):
                integer(pin[key], 1600, 1)
            require(pin["png_width"] * pin["png_height"] == 1440000, "png_pixels")
            integer(pin["exif_orientation"], 8, 1)
        provenance = record["provenance"]
        keys(provenance, ("inventory_receipt", "diagnostic_outer_sha256", "diagnostic_nested_sha256", "mask_context_sha256", "mask_code_pins", "mask_limits", "mask_inventory"))
        file_pin(provenance["inventory_receipt"], MANIFEST_CAP)
        keys(provenance["mask_code_pins"], HISTORICAL_PATHS)
        for value in provenance["mask_code_pins"].values():
            digest(value)
        keys(provenance["mask_limits"], tuple(MASK_LIMITS))
        require(provenance["mask_limits"] == MASK_LIMITS, "historical_limits")
        for key in ("diagnostic_outer_sha256", "diagnostic_nested_sha256", "mask_context_sha256"):
            digest(provenance[key])
        require(all(provenance[k] == request["bindings"][k] == request["metadata"][k]
                    for k in ("diagnostic_outer_sha256", "diagnostic_nested_sha256")), "diagnostic_seal_join")
        sequence(provenance["mask_inventory"], 3)
        for row, label, src in zip(provenance["mask_inventory"], LABELS, request["metadata"]["source_inventory"]):
            keys(row, ("label", "endpoint", "region_id", "reading_binding_sha256", "ingress_record_sha256", "header_packed_sha256", "full_bool_sha256", "polygon_vertices"))
            require((row["label"], row["endpoint"], row["region_id"], row["full_bool_sha256"])
                    == (label, src["endpoint"], src["region_id"], src["full_native_bool_sha256"]), "mask_inventory")
            integer(row["polygon_vertices"], 64, 3)
            for key in ("region_id", "reading_binding_sha256", "ingress_record_sha256", "header_packed_sha256", "full_bool_sha256"):
                digest(row[key])
        runtime = record["runtime"]
        keys(runtime, ("attestation", "python", "modules"))
        file_pin(runtime["attestation"], MANIFEST_CAP)
        keys(runtime["python"], ("path", "sha256", "version"))
        windows_path(runtime["python"]["path"])
        paths.append(windows_path(runtime["python"]["path"]))
        digest(runtime["python"]["sha256"])
        text(runtime["python"]["version"])
        require(type(runtime["modules"]) is list and 0 < len(runtime["modules"]) <= 256, "module_count")
        identities = []
        origins = []
        for module in runtime["modules"]:
            keys(module, ("module", "path", "sha256", "kind"))
            text(module["module"], 256)
            require(re.fullmatch("[A-Za-z_][A-Za-z0-9_.]*", module["module"]) is not None, "module_name")
            origins.append(windows_path(module["path"]))
            digest(module["sha256"])
            require(module["kind"] in ("repository", "stdlib", "extension", "package"), "module_kind")
            identities.append(module["module"])
        require(identities == sorted(set(identities)) and len(origins) == len(set(origins)), "module_identity")
        paths.extend(origins)
        module_pins = {row["module"]: row["sha256"] for row in runtime["modules"]}
        require(all(module_pins.get(name) == pin for name, pin in {**REPOSITORY_PINS, **SEED_PINS}.items()), "seed_pins")
        require(all(name in module_pins for name in ("scripts.evaluation.spatial_context_native_io",
                    "scripts.evaluation.run_spatial_context_native")), "adapter_pins_missing")
        launch = record["launch"]
        keys(launch, ("cwd", "argv", "environment"))
        windows_path(launch["cwd"])
        require(type(launch["argv"]) is list and 1 <= len(launch["argv"]) <= 16, "argv")
        for value in launch["argv"]:
            text(value, 1024)
        require(launch["argv"][0] == runtime["python"]["path"], "python_argv")
        keys(launch["environment"], ENVIRONMENT_KEYS)
        for value in launch["environment"].values():
            text(value, 1024)
        require(all(launch["environment"][k] == "1" for k in ENVIRONMENT_KEYS[3:]), "environment_settings")
        for key in ("SystemRoot", "TEMP", "TMP"):
            windows_path(launch["environment"][key])
        artifacts = record["artifacts"]
        keys(artifacts, ("output_directory", "authorization_path", "consumed_path", "evidence", "worker_candidate", "completion", "failure", "log"))
        output = artifacts["output_directory"]
        windows_path(output)
        paths.append(windows_path(output))
        for key in ("authorization_path", "consumed_path"):
            paths.append(windows_path(artifacts[key]))
        for key, basename in (("evidence", "evidence.json"), ("worker_candidate", "worker-staged-candidate.json"),
                              ("completion", "completion.json"), ("failure", "failure.json"), ("log", "worker.log")):
            require(artifacts[key] == ntpath.join(output, basename), "artifact_name")
            paths.append(windows_path(artifacts[key]))
        require(len(paths) == len(set(paths)), "path_alias")
        for key in ("TEMP", "TMP"):
            require(ntpath.commonpath((output, launch["environment"][key])) == output, "attempt_temp")
        sealed(record, "manifest_sha256")
        require(canonical(record) + b"\n" == data, "manifest_canonical")
        self.record, self.file_sha256 = record, expected_file_sha256
        self.wire = data

    def check(self):
        bound_object(self.record, cap=MANIFEST_CAP)
        require(canonical(self.record) + b"\n" == self.wire, "manifest_mutation")


def identity(info):
    require(stat.S_ISREG(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400, "file_type")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def no_reparse(path):
    current = Path(path)
    for part in (current, *current.parents):
        info = part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400, "reparse")


def read_fixed(pin, cap):
    integer(pin["bytes"], cap, 1)
    no_reparse(pin["path"])
    before = identity(Path(pin["path"]).lstat())
    require(before[2] == pin["bytes"], "file_size")
    with Path(pin["path"]).open("rb") as stream:
        require(identity(os.fstat(stream.fileno())) == before, "file_handle_identity")
        data = stream.read(pin["bytes"] + 1)
        require(identity(os.fstat(stream.fileno())) == before, "file_mutation")
    require(identity(Path(pin["path"]).lstat()) == before, "file_path_mutation")
    require(len(data) == pin["bytes"] and sha(data) == pin["sha256"], "file_digest")
    return data


class FixedReaders:
    def __init__(self, manifest, grant, reader=read_fixed):
        require(grant.active, "worker_grant")
        self.manifest, self.grant, self.reader, self.used = manifest, grant, reader, set()

    def read_payload(self, kind, key, limit):
        require(self.grant.active, "worker_grant")
        self.manifest.check()
        files = self.manifest.record["files"]
        if kind == "diagnostic" and key == "sealed_source":
            pin, cap = files["diagnostic"], DIAGNOSTIC_CAP
        else:
            require(kind == "encoded", "input_kind")
            matches = [p for p in files["frames"] if p["frame_id"] == key]
            require(len(matches) == 1, "input_key")
            pin, cap = matches[0], IMAGE_CAP
        integer(limit, pin["bytes"] + 1, 1)
        require(limit == pin["bytes"] + 1, "read_limit")
        require(pin["path"] not in self.used, "duplicate_read")
        self.used.add(pin["path"])
        return self.reader(pin, cap)


def polygon(value):
    require(type(value) is list and 3 <= len(value) <= 64, "polygon_vertices")
    for vertex in value:
        sequence(vertex, 2)
        require(all(type(v) in (int, float) and math.isfinite(v) for v in vertex), "polygon_vertex")


def diagnostic_shapes(record):
    """All three masks and witness polygons close before native delegates."""
    require(record.get("schema") == "mc-photogrammetry-image-geometry-failure-diagnostic-v2", "diagnostic_schema")
    sequence(record["ingress"], 2)
    for ingress, name in zip(record["ingress"], ("image-0000", "image-0001")):
        ingress_shape(ingress, name)
    witness = record["joint_endpoint_selection"]["first_witness_provenance"]
    sequence(witness["readings"], 3)
    sequence(witness["rows"], 2)
    for reading in witness["readings"]:
        polygon(reading["polygon"])
    bundle = record["source_region_masks"]
    keys(bundle, ("schema", "producer", "status", "source_context", "code_pins", "limits", "entries", "record_sha256"))
    sequence(bundle["entries"], 3)
    total = packed = 0
    for entry in bundle["entries"]:
        keys(entry, ("endpoint", "reading_index", "reading_binding_sha256", "canonical_polygon", "ingress_sha256", "row_refs", "mask", "full_native_bool_c_sha256"))
        polygon(entry["canonical_polygon"])
        require(type(entry["row_refs"]) is list and len(entry["row_refs"]) <= 2, "row_refs")
        mask = entry["mask"]
        keys(mask, ("source_region_id", "axis", "sources", "image_shape", "bbox_xyxy", "crop_shape", "encoding", "bit_order", "packed_mask_base64", "mask_sha256", "pixel_count"))
        require(mask["image_shape"] == [900, 1600], "mask_shape")
        sequence(mask["crop_shape"], 2)
        sequence(mask["bbox_xyxy"], 4)
        for value in [*mask["crop_shape"], *mask["bbox_xyxy"]]:
            integer(value, 1600)
        h, w = mask["crop_shape"]
        x0, y0, x1, y1 = mask["bbox_xyxy"]
        require(0 < h <= 900 and 0 < w <= 1600 and h == y1-y0 and w == x1-x0
                and y1 <= 900 and x1 <= 1600, "mask_crop")
        total += 1440000
        length = (h*w+7)//8
        packed += length
        require(packed <= 524288 and total <= 4320000, "mask_budget")
        require(type(mask["packed_mask_base64"]) is str
                and len(mask["packed_mask_base64"]) == 4*((length+2)//3), "packed_length")
        require(type(mask["sources"]) is list and 1 <= len(mask["sources"]) <= 32, "mask_sources")
        for value in mask["sources"]:
            text(value)
        require(mask["sources"] == sorted(set(mask["sources"])), "mask_sources")
        for value in (mask["source_region_id"], mask["mask_sha256"], entry["reading_binding_sha256"], entry["ingress_sha256"], entry["full_native_bool_c_sha256"]):
            digest(value)


def ingress_shape(record, frame_id):
    keys(record, ("frame_id", "source_label", "byte_sha256", "byte_length", "decoded_shape",
                  "decoded_dtype", "decoded_pixel_sha256", "transform", "schema", "decoder"))
    require(record["frame_id"] == frame_id and record["schema"] == "mc-photogrammetry-frame-ingress-provenance-v1"
            and record["decoder"] == "input.image.load_image:exif_transpose+convert('RGB')", "ingress_schema")
    text(record["source_label"], 1024)
    integer(record["byte_length"], IMAGE_CAP, 1)
    digest(record["byte_sha256"])
    digest(record["decoded_pixel_sha256"])
    sequence(record["decoded_shape"], 3)
    require(all(type(n) is int for n in record["decoded_shape"])
            and record["decoded_shape"] == [900,1600,3] and record["decoded_dtype"] == "|u1", "ingress_layout")
    transform = record["transform"]
    keys(transform, ("source_size", "crop_origin", "crop_size", "output_size"))
    for field, value in transform.items():
        if value is not None:
            sequence(value, 2)
            require(all(type(n) is int and 0 <= n <= 1600 for n in value), "ingress_transform_type")
    require(transform["source_size"] == [1600,900] and transform["crop_origin"] == [0,0]
            and transform["crop_size"] in (None,[1600,900])
            and transform["output_size"] in (None,[1600,900]), "ingress_transform")


class DiagnosticAdapter:
    def __init__(self, manifest, services):
        self.manifest, self.services, self.entries, self.used = manifest, services, None, set()

    def inspect(self, data):
        self.manifest.check()
        pin = self.manifest.record["files"]["diagnostic"]
        require(type(data) is bytes and len(data) == pin["bytes"] and sha(data) == pin["sha256"], "diagnostic_external")
        record = parse(data, cap=DIAGNOSTIC_CAP, packed=True)
        diagnostic_shapes(record)
        d = self.services.diagnostic
        require(record["record_sha256"] == d._digest({k:v for k,v in record.items() if k != "record_sha256"}), "outer_seal")
        bundle = record["source_region_masks"]
        d._bounded_bytes(bundle, limit=DIAGNOSTIC_CAP)
        d._preflight_mask_entries(bundle["entries"])
        context = d._mask_source_context(record)
        provenance = self.manifest.record["provenance"]
        require(bundle["schema"] == "mc-photogrammetry-source-region-mask-bundle-v1"
                and bundle["producer"] == "native-plane-region-mask-export-1"
                and bundle["status"] == "complete", "bundle_schema")
        require(bundle["code_pins"] == provenance["mask_code_pins"]
                and bundle["limits"] == provenance["mask_limits"] == d._mask_limits(), "historical_provenance")
        require(bundle["record_sha256"] == d._digest({k:v for k,v in bundle.items() if k != "record_sha256"}, limit=DIAGNOSTIC_CAP), "nested_seal")
        require(context == bundle["source_context"] and d._digest(context) == provenance["mask_context_sha256"], "mask_context")
        d._validate_mask_entry_joins(bundle["entries"], context)
        require(record["record_sha256"] == provenance["diagnostic_outer_sha256"]
                and bundle["record_sha256"] == provenance["diagnostic_nested_sha256"], "external_seals")
        sources = self.manifest.record["request_template"]["metadata"]["source_inventory"]
        sequence(context["ingress"], 2)
        ingress_records = {}
        for ingress, frame in zip(context["ingress"], self.manifest.record["files"]["frames"]):
            ingress_shape(ingress, frame["frame_id"])
            source_record = next(src for src in sources if src["frame_id"] == frame["frame_id"])
            require((ingress["byte_length"], ingress["byte_sha256"], ingress["decoded_shape"],
                     ingress["decoded_dtype"], ingress["decoded_pixel_sha256"])
                    == (source_record["encoded_length"], source_record["encoded_sha256"], source_record["decoded_shape"],
                        source_record["decoded_dtype"], source_record["ingress_sha256"]), "ingress_rgb_source_join")
            ingress_records[frame["frame_id"]] = d._digest(ingress)
        for entry, inventory, src in zip(bundle["entries"], provenance["mask_inventory"], sources):
            mask = entry["mask"]
            require(entry["endpoint"] == src["endpoint"] and mask["source_region_id"] == src["region_id"]
                    and entry["full_native_bool_c_sha256"] == src["full_native_bool_sha256"]
                    and entry["ingress_sha256"] == ingress_records[src["frame_id"]], "mask_rgb_source_join")
            require((entry["endpoint"], mask["source_region_id"], entry["reading_binding_sha256"],
                     entry["ingress_sha256"], mask["mask_sha256"], entry["full_native_bool_c_sha256"], len(entry["canonical_polygon"]))
                    == (inventory["endpoint"], inventory["region_id"], inventory["reading_binding_sha256"],
                        inventory["ingress_record_sha256"], inventory["header_packed_sha256"], inventory["full_bool_sha256"], inventory["polygon_vertices"]), "external_inventory")
        self.entries = {entry["mask"]["source_region_id"]: entry for entry in bundle["entries"]}
        require(len(self.entries) == 3, "duplicate_region")
        self.payload = data
        return self.services.driver.DiagnosticInspection(sha(data), record["record_sha256"], bundle["record_sha256"], tuple(sources))

    def decode_mask(self, data, region):
        self.manifest.check()
        require(self.entries is not None and data is self.payload and region in self.entries, "mask_request")
        require(region not in self.used, "duplicate_mask")
        self.used.add(region)
        result = self.services.diagnostic._decode_source_mask(self.entries[region])
        require(result.shape == (900,1600) and result.dtype.str == "|b1" and result.flags.c_contiguous
                and sha(result.tobytes(order="C")) == self.entries[region]["full_native_bool_c_sha256"], "mask_identity")
        result.setflags(write=False)
        return result


def exif_orientation(payload):
    """Inspect only bounded TIFF IFD0; never hand unbounded metadata to Pillow."""
    require(8 <= len(payload) <= 4096 and payload[:2] in (b"II", b"MM"), "png_exif")
    endian = "<" if payload[:2] == b"II" else ">"
    require(struct.unpack_from(endian+"H", payload, 2)[0] == 42, "png_exif")
    offset = struct.unpack_from(endian+"I", payload, 4)[0]
    require(8 <= offset <= len(payload)-2, "png_exif_offset")
    count = struct.unpack_from(endian+"H", payload, offset)[0]
    require(count <= 64 and offset+2+12*count+4 <= len(payload), "png_exif_entries")
    orientation = 1
    found = False
    for index in range(count):
        tag, kind, size, value = struct.unpack_from(endian+"HHII", payload, offset+2+12*index)
        widths = {1:1,2:1,3:2,4:4,5:8,7:1,9:4,10:8}
        require(kind in widths and size <= 4096 and size*widths[kind] <= 4096, "png_exif_value")
        if size*widths[kind] > 4:
            require(8 <= value <= len(payload)-size*widths[kind], "png_exif_value")
        if tag == 274:
            require(not found and kind == 3 and size == 1, "png_exif_orientation")
            orientation = struct.unpack_from(endian+"H", payload, offset+2+12*index+8)[0]
            integer(orientation, 8, 1)
            found = True
    # Reject additional linked IFDs rather than leaving a decoder traversal uncapped.
    require(struct.unpack_from(endian+"I", payload, offset+2+12*count)[0] == 0, "png_exif_chain")
    return orientation


def png_preflight(data, pin):
    require(type(data) is bytes and 33 <= len(data) <= IMAGE_CAP and data[:8] == b"\x89PNG\r\n\x1a\n", "png_signature")
    pos = 8
    count = 0
    seen_data = ended = False
    ancillary = set()
    orientation = 1
    while pos < len(data):
        require(len(data)-pos >= 12, "png_chunk")
        length = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos+4:pos+8]
        require(length <= IMAGE_CAP and length <= len(data)-pos-12, "png_chunk")
        payload = data[pos+8:pos+8+length]
        require(zlib.crc32(kind+payload) & 0xffffffff == struct.unpack_from(">I", data, pos+8+length)[0], "png_crc")
        count += 1
        require(count <= 65536, "png_chunks")
        if count == 1:
            require(kind == b"IHDR" and length == 13, "png_ihdr")
            w,h,depth,color,compression,filters,interlace = struct.unpack(">IIBBBBB",payload)
            require((w,h) == (pin["png_width"],pin["png_height"]) and w <= 1600 and h <= 1600 and w*h == 1440000, "png_dimensions")
            legal = {0:(1,2,4,8,16),2:(8,16),3:(1,2,4,8),4:(8,16),6:(8,16)}
            require(color in legal and depth in legal[color] and compression == filters == 0 and interlace in (0,1), "png_encoding")
        elif kind == b"IDAT":
            require(not ended, "png_order")
            seen_data = True
        elif kind == b"IEND":
            require(length == 0 and seen_data and pos+12 == len(data), "png_iend")
            ended = True
        else:
            # A small closed ancillary policy excludes compressed profiles/text,
            # APNG and arbitrary decoder-triggering decompression metadata.
            require(kind in (b"PLTE", b"tRNS", b"gAMA", b"cHRM", b"sRGB", b"pHYs", b"eXIf"), "png_ancillary")
            require(length <= (4096 if kind == b"eXIf" else 1024), "png_ancillary_size")
            require(kind not in ancillary and not seen_data, "png_ancillary_order")
            ancillary.add(kind)
            if kind == b"eXIf":
                orientation = exif_orientation(payload)
        pos += 12+length
    require(ended and count >= 3, "png_iend")
    require(orientation == pin["exif_orientation"], "png_orientation_pin")


class RGBAdapter:
    def __init__(self, manifest, services):
        self.manifest, self.services, self.used, self.pixels, self.usable = manifest, services, set(), {}, {}

    def decode_rgb(self, data, frame_id):
        self.manifest.check()
        matches = [pin for pin in self.manifest.record["files"]["frames"] if pin["frame_id"] == frame_id]
        require(len(matches) == 1 and frame_id not in self.used, "rgb_once")
        pin = matches[0]
        require(type(data) is bytes and len(data) == pin["bytes"] and sha(data) == pin["sha256"], "rgb_encoded")
        png_preflight(data, pin)
        self.used.add(frame_id)
        pixels = self.services.load_image_bytes(data, suffix=".png")
        src = next(s for s in self.manifest.record["request_template"]["bindings"]["sources"] if s["frame_id"] == frame_id)
        require(pixels.shape == (900,1600,3) and pixels.dtype.str == "|u1" and pixels.flags.c_contiguous, "rgb_layout")
        require(sha(pixels.tobytes(order="C")) == src["decoded_rgb_sha256"]
                and self.services.image_sha256(pixels) == src["ingress_sha256"], "rgb_identity")
        pixels.setflags(write=False)
        self.pixels[frame_id] = pixels
        return pixels

    def ingress_digest(self, pixels):
        require(any(pixels is value for value in self.pixels.values()), "rgb_object")
        return self.services.image_sha256(pixels)

    def usable_mask(self, pixels, helper):
        self.manifest.check()
        names = [name for name,value in self.pixels.items() if value is pixels]
        require(len(names) == 1 and names[0] not in self.usable, "usable_once")
        name = names[0]
        src = next(s for s in self.manifest.record["request_template"]["bindings"]["sources"] if s["frame_id"] == name)
        require(helper == src["usable_helper_sha256"], "usable_helper")
        mask = self.services.usable_mask_for(name, pixels.shape, None)
        require(mask.shape == (900,1600) and mask.dtype.str == "|b1" and mask.flags.c_contiguous
                and sha(mask.tobytes(order="C")) == src["usable_mask_sha256"], "usable_identity")
        mask.setflags(write=False)
        self.usable[name] = mask
        return mask


class VerifiedSourceLoader:
    """Execute immutable verified bytes. Never reopen a path or consult bytecode."""
    def __init__(self, name, path, data, expected_sha256, on_loaded=None):
        require(type(data) is bytes and 0 < len(data) <= 64*1024*1024
                and path.endswith(".py") and sha(data) == expected_sha256, "source_buffer_pin")
        self.name, self.path, self.data = name, path, data
        self.on_loaded = on_loaded

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        require(module.__name__ == self.name, "source_module_name")
        module.__file__ = self.path
        exec(compile(self.data, self.path, "exec", dont_inherit=True), module.__dict__)
        if self.on_loaded is not None:
            self.on_loaded(self, module)


class VerifiedSourceFinder:
    """Closed memory registry; an injected cache keeps authored controls isolated."""
    def __init__(self, records, buffers, cache, bootstrap=()):
        require(len(records) <= 256 and set(records) == set(buffers), "source_registry")
        self.records, self.buffers, self.cache = records, buffers, cache
        require(all(name in ("scripts.evaluation.spatial_context_native_io", "scripts.evaluation.run_spatial_context_native")
                    for name in bootstrap), "bootstrap_name")
        self.baseline = {name:cache[name] for name in records if name in cache}
        self.bootstrap = {name:cache[name] for name in bootstrap}
        self.session, self.issued = {}, {}
        for name, pin in records.items():
            require(name not in cache or name in bootstrap, "preloaded_service_module")
            require(pin["path"].endswith(".py") and sha(buffers[name]) == pin["sha256"], "source_registry_pin")

    def attest(self, loader, module):
        name = loader.name
        require(self.issued.get(name) is loader and self.cache.get(name) is module
                and module.__name__ == name and module.__file__ == self.records[name]["path"]
                and module.__spec__.loader is loader and module.__loader__ is loader
                and module.__spec__.origin == loader.path,
                "session_load_identity")
        self.session[name] = (module, loader, module.__spec__, loader.path, sha(loader.data))

    def reuse(self, name):
        require(name in self.records and name in self.cache, "session_module_missing")
        module = self.cache[name]
        if name in self.bootstrap:
            require(module is self.bootstrap[name] and module is self.baseline[name]
                    and windows_path(str(Path(module.__file__))) == windows_path(self.records[name]["path"]),
                    "bootstrap_identity")
            return module
        require(name in self.session, "unattested_service_module")
        observed, loader, spec, path, source_sha = self.session[name]
        require(module is observed and module.__spec__ is spec and spec.loader is loader and module.__loader__ is loader
                and spec.origin == path and module.__file__ == path
                and source_sha == self.records[name]["sha256"] and self.issued.get(name) is loader,
                "session_module_substitution")
        return module

    def load(self, name, importer):
        require(name in self.records, "module_pin_missing")
        if name in self.cache:
            return self.reuse(name)
        module = importer(name)
        require(self.reuse(name) is module, "session_import_result")
        return module

    def find_spec(self, fullname, path=None, target=None):
        from importlib import machinery, util
        if fullname in self.records:
            if fullname in self.cache:
                return self.reuse(fullname).__spec__
            pin = self.records[fullname]
            loader = VerifiedSourceLoader(fullname, pin["path"], self.buffers[fullname], pin["sha256"], self.attest)
            self.issued[fullname] = loader
            package = Path(pin["path"]).name == "__init__.py"
            spec = util.spec_from_loader(fullname, loader, origin=pin["path"], is_package=package)
            if package:
                spec.submodule_search_locations = [str(Path(pin["path"]).parent)]
            return spec
        children = [row for name,row in self.records.items() if name.startswith(fullname+".")]
        if children:
            directories = set()
            for row in children:
                directory = Path(row["path"]).parent
                for _ in range(len(row["module"][len(fullname)+1:].split("."))-1):
                    directory = directory.parent
                directories.add(str(directory))
            require(len(directories) == 1, "namespace_origin")
            spec = machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = list(directories)
            return spec
        spec = machinery.BuiltinImporter.find_spec(fullname)
        if spec is None:
            spec = machinery.FrozenImporter.find_spec(fullname)
        require(spec is not None, "undeclared_module_origin")
        return spec


def lazy_services(manifest, grant):
    """Inactive until separately approved native/bootstrap support exists."""
    require(grant.active and grant.job_enforced, "loader_grant")
    manifest.check()
    import sys
    modules = {r["module"]:r for r in manifest.record["runtime"]["modules"]}
    # Extension/native loading cannot use this source-only loader. A future
    # reviewed bootstrap must attest and supply a separate exact native loader.
    require(all(row["kind"] != "extension" and row["path"].endswith(".py") for row in modules.values()), "native_bootstrap_required")
    buffers = {name:read_fixed(dict(path=pin["path"], bytes=Path(pin["path"]).stat().st_size,
                        sha256=pin["sha256"]), 64*1024*1024) for name,pin in modules.items()}
    bootstrap = ("scripts.evaluation.spatial_context_native_io", "scripts.evaluation.run_spatial_context_native")
    for name in bootstrap:
        cached = sys.modules.get(name)
        require(cached is not None and getattr(cached, "__file__", None) is not None
                and windows_path(str(Path(cached.__file__))) == windows_path(modules[name]["path"]), "bootstrap_origin")
    finder = VerifiedSourceFinder(modules, buffers, sys.modules, bootstrap)
    sys.meta_path[:] = [finder]
    def load(name):
        return finder.load(name, importlib.import_module)
    from types import SimpleNamespace
    driver = load("scripts.evaluation.probe_spatial_context_ambiguity")
    return SimpleNamespace(driver=driver, diagnostic=load("scripts.evaluation.diagnose_image_geometry_failure"),
        load_image_bytes=load("mc_photogrammetry.input.image").load_image_bytes,
        image_sha256=load("mc_photogrammetry.pipeline.image_derived_session").image_sha256,
        usable_mask_for=load("mc_photogrammetry.pipeline.correspondence").usable_mask_for)
