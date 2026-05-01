from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LAYOUT_LINE_FROM_RIGHT_IN = 22.5
REAL_DOOR_BOTTOM_WIDTH_IN = 42.5
NAPA_HORIZONTAL_OFFSET_IN = 0.5
RIGHT_NAPA_EXTRA_LEFT_SHIFT_IN = 0.0

DECAL_LAYOUT = {
    "napaL": {
        "size_in": [20.5, 15.15],
        "anchor_bottom_center_in": [NAPA_HORIZONTAL_OFFSET_IN, 16.0],
        "image": BASE_DIR / "decals" / "napaL.png",
    },
    "serialnumber": {
        "size_in": [20.0, 2.1],
        "anchor_bottom_center_in": [0.0, 1.5],
        "image": BASE_DIR / "decals" / "serialnumber.png",
    },
    "usdot": {
        "size_in": [12.25, 2.1],
        "anchor_bottom_center_in": [0.0, 4.5],
        "image": BASE_DIR / "decals" / "usdot.png",
    },
    "safetyfirst": {
        "size_in": [12.0, 6.0],
        "anchor_bottom_center_in": [0.0, 8.0],
        "image": BASE_DIR / "decals" / "safetyfirst.png",
    },
}


def _mirror_anchor(anchor_bottom_center_in):
    return [-float(anchor_bottom_center_in[0]), float(anchor_bottom_center_in[1])]


def _right_side_anchor(name, anchor_bottom_center_in):
    mirrored = _mirror_anchor(anchor_bottom_center_in)
    if name == "napaL":
        mirrored[0] -= float(RIGHT_NAPA_EXTRA_LEFT_SHIFT_IN)
    return mirrored


def get_decal_layout(door_side="left"):
    side = str(door_side).lower()
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported door side: {door_side}")

    if side == "left":
        return {
            name: {
                "size_in": list(spec["size_in"]),
                "anchor_bottom_center_in": list(spec["anchor_bottom_center_in"]),
                "image": Path(spec["image"]),
            }
            for name, spec in DECAL_LAYOUT.items()
        }

    mirrored = {}
    for name, spec in DECAL_LAYOUT.items():
        mirrored_name = "napaR" if name == "napaL" else name
        mirrored_image = BASE_DIR / "decals" / "napaR.png" if name == "napaL" else Path(spec["image"])
        mirrored[mirrored_name] = {
            "size_in": list(spec["size_in"]),
            "anchor_bottom_center_in": _right_side_anchor(name, spec["anchor_bottom_center_in"]),
            "image": mirrored_image,
        }
    return mirrored
