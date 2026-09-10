"""Resolve X-Fi backbone files without mixing released and trained weights."""

from __future__ import annotations

import hashlib
from pathlib import Path


FILENAMES = {
    "released": {
        "mmwave": Path("mmWave/mmwave_ResNet18.pt"),
        "wifi": Path("WIFI/wifi_ResNet18.pt"),
        "rfid": Path("RFID/RFID_ResNet18.pt"),
    },
    "ours": {
        "mmwave": Path("mmWave/mmwave_ResNet18.pt"),
        "wifi": Path("WIFI/wifi_ResNet18.pt"),
        "rfid": Path("RFID/rfid_ResNet18.pt"),
    },
}

RELEASED_SHA256 = {
    "mmwave": "03ae1027da2baf2210bfda363497de2913c9ff044423f2158e03a6b4041a6d56",
    "wifi": "3cb3f45d45775fd0c99f0f8650b9bd621d79eb801e7286b44ff908a74fd4c4e9",
    "rfid": "0b5803413483a6f67db88481f63c5997c35c0305465e21d8287182490dd2bd3b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_backbones(
    backbone_root: str | Path,
    source: str,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Resolve and verify one complete, explicitly declared backbone set."""
    if source not in FILENAMES:
        raise ValueError(f"backbone source must be one of {tuple(FILENAMES)}, got {source!r}")

    root = Path(backbone_root).resolve()
    paths = {name: root / relative for name, relative in FILENAMES[source].items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{source} backbone set is incomplete under {root}; missing: {missing}"
        )
    digests = {name: sha256(path) for name, path in paths.items()}

    if source == "released":
        mismatches = {
            name: {"expected": RELEASED_SHA256[name], "actual": digest}
            for name, digest in digests.items()
            if digest != RELEASED_SHA256[name]
        }
        if mismatches:
            raise ValueError(
                "--backbone-source released was declared, but checkpoint hashes "
                f"do not match the released files: {mismatches}"
            )
    else:
        copied_release = [
            name for name, digest in digests.items() if digest == RELEASED_SHA256[name]
        ]
        if copied_release:
            raise ValueError(
                "--backbone-source ours was declared, but these checkpoints are "
                f"released weights: {copied_release}"
            )
        missing_metadata = [
            str(path.with_suffix(".json"))
            for path in paths.values()
            if not path.with_suffix(".json").is_file()
        ]
        if missing_metadata:
            raise FileNotFoundError(
                "trained backbones require their provenance metadata; missing: "
                f"{missing_metadata}"
            )

    return paths, digests
