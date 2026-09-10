#!/usr/bin/env python3
"""Build the XRF55 evaluation splits from the raw release layout.

The protocols follow Sec. 5.3 of the XRF55 paper (https://doi.org/10.1145/3643543):

* trial_split: Part 1 trials 01-14 train, 15-20 test; all four scenes. Holds out
  repetitions only, so the same people and actions appear on both sides.
* trial_split_validation: Part 1 trials 01-10 train, 11-14 validate.
* unseen_subjects: Scene 1 people 01-21 train, 22-30 test; all 20 trials.
* unseen_subjects_validation: Scene 1 people 01-16 train, 17-21 validate.
* unseen_scenes: Scene 1 train, Scene 2 validate, Scenes 3 and 4 test.
* unseen_scene_unseen_subject: Scene 1 minus everyone appearing in another scene
  train, Scene 2 validate, Scenes 3 and 4 test; room and people both unseen.

The scene protocols carry their own validation split, so tuning happens under the
same kind of shift the test measures rather than on a slice of training data.

Expected source layout, where ``raw_root`` holds both released archives:

    {raw_root}/part1/Scene{1..4}/Scene{1..4}/{RFID,WiFi,mmWave}/*.npy
    {raw_root}/part2/Scene1_part2/{RFID,WiFi,mmWave}/*.npy

Set that root with ``--raw-root`` or by exporting ``XRF55_RAW_ROOT``; with neither
set it falls back to ``DEFAULT_RAW_ROOT`` below.

Part 1 covers Scene 1 subjects 01-11 plus Scenes 2-4; Part 2 extends Scene 1 to
subjects 12-30; Scenes 2-4 draw on the same global subject numbering and add
subject 31. Files are named ``{subject}_{action}_{repetition}.npy`` with
1-based identifiers, so the classification label is ``action - 1``.

Import ``build_split`` to resolve a protocol into file lists in process. Run the
script to report a split, or drop the ``--dry-run`` flag to materialise the
symlink tree that dataset classes expecting a pre-split root consume:

    {dst}/{protocol}/{train_data,test_data}/{modality}/Scene{N}/Scene{N}/*.npy

Scene 1 links merge Part 1 and Part 2 into one directory; their subject ranges are
disjoint, so no filename collides.

Only the standard library is used, so the split can be prepared without the
training stack installed.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# Shared cluster copy of the release. Override per machine with --raw-root or
# by exporting XRF55_RAW_ROOT.
DEFAULT_RAW_ROOT = Path("/mnt/weka/rmkrtchyan/ws/data/XRF55")

MODALITIES = ("RFID", "WiFi", "mmWave")
ALL_SCENES = ("Scene1", "Scene2", "Scene3", "Scene4")
# Subject identifiers are global rather than scene-local. Scene 1 covers 01-30
# (Part 1 holds 01-11, Part 2 extends it to 12-30) while Scenes 2-4 each reuse
# three of those identifiers or add subject 31, so one person can appear in two
# scenes. Only the scene-transfer protocol is affected, and by design.
SCENE1_SUBJECTS = range(1, 31)
SUBJECT_IDS = range(1, 32)

# Subject roster of each released scene. Scenes 2-4 re-record people already in
# Scene 1, except subject 31, who appears only in Scene 2. This is what makes a
# joint scene-and-subject shift possible at all: excluding a scene's people from
# training leaves that scene both an unseen room and unseen people.
SCENE_SUBJECTS: dict[str, frozenset[int]] = {
    "Scene1": frozenset(range(1, 31)),
    "Scene2": frozenset({5, 24, 31}),
    "Scene3": frozenset({6, 7, 23}),
    "Scene4": frozenset({3, 4, 13}),
}
ACTIONS = range(1, 56)
REPETITIONS = range(1, 21)

SCENE_ROOTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "Scene1": (("part1", "Scene1", "Scene1"), ("part2", "Scene1_part2")),
    "Scene2": (("part1", "Scene2", "Scene2"),),
    "Scene3": (("part1", "Scene3", "Scene3"),),
    "Scene4": (("part1", "Scene4", "Scene4"),),
}

TRAIN_SPLIT = "train"


@dataclass(frozen=True)
class Selector:
    """Which raw files belong to one split."""

    scenes: tuple[str, ...]
    subjects: frozenset[int] | None = None
    repetitions: frozenset[int] | None = None
    part1_only: bool = False


@dataclass(frozen=True)
class Protocol:
    """A training selector, plus evaluation selectors split by role.

    ``validation`` names the evaluation splits that exist for tuning. Everything
    else in ``evaluations`` is a test split, to be read once and never used to
    choose between configurations.
    """

    name: str
    description: str
    train: Selector
    evaluations: dict[str, Selector]
    validation: frozenset[str] = frozenset()
    expected: tuple[int, dict[str, int]] | None = None

    def __post_init__(self) -> None:
        unknown = self.validation - set(self.evaluations)
        if unknown:
            raise ValueError(f"{self.name}: no such evaluation split {sorted(unknown)}")

    def splits(self) -> list[tuple[str, Selector]]:
        return [(TRAIN_SPLIT, self.train), *self.evaluations.items()]

    def validation_splits(self) -> tuple[str, ...]:
        return tuple(n for n in self.evaluations if n in self.validation)

    def test_splits(self) -> tuple[str, ...]:
        return tuple(n for n in self.evaluations if n not in self.validation)

    def role(self, split: str) -> str:
        if split == TRAIN_SPLIT:
            return "train"
        return "validation" if split in self.validation else "test"

    def directory_name(self, split: str) -> str:
        """Map a logical split name onto its symlink-tree directory."""
        if split == TRAIN_SPLIT:
            return "train_data"
        role = self.role(split)
        prefix = "validation_data" if role == "validation" else "test_data"
        siblings = (
            self.validation_splits() if role == "validation" else self.test_splits()
        )
        return prefix if len(siblings) == 1 else f"{prefix}_{split}"


PROTOCOL_NAMES = (
    "trial_split",
    "trial_split_validation",
    "unseen_subjects",
    "unseen_subjects_validation",
    "unseen_scenes",
    "unseen_scenes_two_rooms",
    "unseen_scene_unseen_subject",
)

ALL_TRIALS = frozenset(REPETITIONS)


def subjects_outside_scene1() -> frozenset[int]:
    """People who appear in Scenes 2-4 and so cannot stay in Scene-1 training.

    Subject 31 is in the set but not in Scene 1, so dropping the intersection
    leaves 22 people rather than 21.
    """
    elsewhere: set[int] = set()
    for scene in ("Scene2", "Scene3", "Scene4"):
        elsewhere |= set(SCENE_SUBJECTS[scene])
    return frozenset(elsewhere)


def build_protocol(name: str) -> Protocol:
    """Return the selectors for one protocol.

    Sizes are per modality and follow from the release: 55 actions and 20
    repetitions per person, Scene 1 covering people 01-30, and Scenes 2-4 covering
    three people each.
    """
    if name == "trial_split":
        return Protocol(
            name=name,
            description=(
                "Part 1 trials 01-14 train, trials 15-20 test, all four scenes. "
                "Holds out repetitions only: the same people and actions appear on "
                "both sides."
            ),
            train=Selector(ALL_SCENES, repetitions=frozenset(range(1, 15)),
                           part1_only=True),
            evaluations={
                "part1_trials_15_20": Selector(
                    ALL_SCENES,
                    repetitions=frozenset(range(15, 21)),
                    part1_only=True,
                )
            },
            expected=(15400, {"part1_trials_15_20": 6600}),
        )
    if name == "trial_split_validation":
        return Protocol(
            name=name,
            description="Part 1 trials 01-10 train, trials 11-14 validate.",
            train=Selector(ALL_SCENES, repetitions=frozenset(range(1, 11)),
                           part1_only=True),
            evaluations={
                "validation_trials_11_14": Selector(
                    ALL_SCENES,
                    repetitions=frozenset(range(11, 15)),
                    part1_only=True,
                )
            },
            validation=frozenset({"validation_trials_11_14"}),
            expected=(11000, {"validation_trials_11_14": 4400}),
        )
    if name == "unseen_subjects":
        return Protocol(
            name=name,
            description=(
                "Scene 1 people 01-21 train, 22-30 test, all 20 trials. Holds out "
                "people within one room."
            ),
            train=Selector(("Scene1",), subjects=frozenset(range(1, 22))),
            evaluations={
                "subjects_22_30": Selector(
                    ("Scene1",), subjects=frozenset(range(22, 31))
                )
            },
            expected=(23100, {"subjects_22_30": 9900}),
        )
    if name == "unseen_subjects_validation":
        return Protocol(
            name=name,
            description=(
                "Scene 1 people 01-16 train, 17-21 validate. Tuning partner for "
                "unseen_subjects, which keeps its full 01-21 training set."
            ),
            train=Selector(("Scene1",), subjects=frozenset(range(1, 17))),
            evaluations={
                "validation_subjects_17_21": Selector(
                    ("Scene1",), subjects=frozenset(range(17, 22))
                )
            },
            validation=frozenset({"validation_subjects_17_21"}),
            expected=(17600, {"validation_subjects_17_21": 5500}),
        )
    if name == "unseen_scenes":
        return Protocol(
            name=name,
            description=(
                "Scene 1 train, Scene 2 validate, Scenes 3 and 4 test. Holds out "
                "rooms; most of the people are seen in training."
            ),
            train=Selector(("Scene1",), subjects=frozenset(SCENE1_SUBJECTS),
                           repetitions=ALL_TRIALS),
            evaluations={
                # Scene 2 is restricted to people the model trained on. Subject 31
                # appears in no other scene, so leaving it in would make validation
                # a mixture of room shift and room-and-person shift while every
                # test person is one the model has seen. Validation has to be the
                # same kind of shift as the test, or tuning selects for the wrong
                # thing. It costs one person: 3,300 samples become 2,200.
                "Scene2": Selector(("Scene2",),
                                   subjects=frozenset(SCENE1_SUBJECTS)),
                "Scene3": Selector(("Scene3",)),
                "Scene4": Selector(("Scene4",)),
            },
            validation=frozenset({"Scene2"}),
            expected=(33000, {"Scene2": 2200, "Scene3": 3300, "Scene4": 3300}),
        )
    if name == "unseen_scenes_two_rooms":
        return Protocol(
            name=name,
            description=(
                "Scenes 1 and 2 train, Scenes 3 and 4 test. The two-room variant "
                "of unseen_scenes, for measuring whether a second training room "
                "improves transfer to a third. It has no validation split: Scene 2 "
                "is the room it moves into training, so selection cannot happen "
                "here. Run it only at a configuration already frozen elsewhere."
            ),
            train=Selector(("Scene1", "Scene2"), repetitions=ALL_TRIALS),
            evaluations={
                "Scene3": Selector(("Scene3",)),
                "Scene4": Selector(("Scene4",)),
            },
            expected=(36300, {"Scene3": 3300, "Scene4": 3300}),
        )
    if name == "unseen_scene_unseen_subject":
        kept = frozenset(set(SCENE1_SUBJECTS) - subjects_outside_scene1())
        return Protocol(
            name=name,
            description=(
                "Scene 1 minus everyone who appears in another scene train, Scene 2 "
                "validate, Scenes 3 and 4 test. Room and people are both unseen."
            ),
            train=Selector(("Scene1",), subjects=kept, repetitions=ALL_TRIALS),
            evaluations={scene: Selector((scene,)) for scene in
                         ("Scene2", "Scene3", "Scene4")},
            validation=frozenset({"Scene2"}),
            expected=(24200, {"Scene2": 3300, "Scene3": 3300, "Scene4": 3300}),
        )
    raise ValueError(f"unknown protocol: {name}")


@dataclass(frozen=True)
class Sample:
    """One raw file with its decoded identity."""

    path: Path
    modality: str
    scene: str
    subject: int
    action: int
    repetition: int

    @property
    def label(self) -> int:
        return self.action - 1

    @property
    def identity(self) -> tuple[str, int, int, int]:
        return (self.scene, self.subject, self.action, self.repetition)


def scene_roots(raw_root: Path, scene: str, part1_only: bool = False) -> list[Path]:
    """Return the raw directories holding one scene, Part 2 included."""
    roots = [raw_root.joinpath(*parts) for parts in SCENE_ROOTS[scene]]
    if scene == "Scene1" and part1_only:
        roots = roots[:1]
    missing = [str(path) for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing XRF55 scene roots: {missing}")
    return roots


def collect(raw_root: Path, modality: str, selector: Selector) -> list[Sample]:
    """Enumerate every file of one modality that the selector accepts."""
    samples: list[Sample] = []
    identities: set[tuple[str, int, int, int]] = set()

    for scene in selector.scenes:
        for root in scene_roots(raw_root, scene, selector.part1_only):
            directory = root / modality
            if not directory.is_dir():
                raise FileNotFoundError(f"missing modality directory: {directory}")
            for path in sorted(directory.glob("*.npy")):
                parts = path.stem.split("_")
                if len(parts) != 3:
                    raise ValueError(f"unexpected XRF55 filename: {path}")
                try:
                    subject, action, repetition = (int(part) for part in parts)
                except ValueError as error:
                    raise ValueError(f"unexpected XRF55 filename: {path}") from error
                if subject not in SUBJECT_IDS:
                    raise ValueError(f"subject out of range 1..31: {path}")
                if action not in ACTIONS:
                    raise ValueError(f"action out of range 1..55: {path}")
                if repetition not in REPETITIONS:
                    raise ValueError(f"repetition out of range 1..20: {path}")
                if selector.subjects is not None and subject not in selector.subjects:
                    continue
                if (
                    selector.repetitions is not None
                    and repetition not in selector.repetitions
                ):
                    continue
                sample = Sample(
                    path=path,
                    modality=modality,
                    scene=scene,
                    subject=subject,
                    action=action,
                    repetition=repetition,
                )
                if sample.identity in identities:
                    raise ValueError(f"duplicate XRF55 sample: {sample.identity}")
                identities.add(sample.identity)
                samples.append(sample)

    if not samples:
        raise ValueError(
            f"no samples for modality={modality}, scenes={list(selector.scenes)}, "
            f"subjects={selector.subjects}, repetitions={selector.repetitions}"
        )
    samples.sort(key=lambda sample: sample.identity)
    return samples


def check_alignment(per_modality: dict[str, list[Sample]], split: str) -> None:
    """Fail when the modalities of one split do not cover the same samples."""
    reference_modality, reference_samples = next(iter(per_modality.items()))
    reference = {sample.identity for sample in reference_samples}
    for modality, samples in per_modality.items():
        identities = {sample.identity for sample in samples}
        if identities == reference:
            continue
        missing = sorted(reference - identities)[:5]
        extra = sorted(identities - reference)[:5]
        raise ValueError(
            f"{split}: {modality} is not aligned with {reference_modality}; "
            f"missing examples {missing}, unexpected examples {extra}"
        )


def build_split(
    raw_root: Path, protocol: Protocol, modalities: tuple[str, ...]
) -> dict[str, dict[str, list[Sample]]]:
    """Resolve a protocol into ``{split: {modality: samples}}``."""
    split_samples: dict[str, dict[str, list[Sample]]] = {}
    for split, selector in protocol.splits():
        per_modality = {
            modality: collect(raw_root, modality, selector) for modality in modalities
        }
        check_alignment(per_modality, split)
        split_samples[split] = per_modality
    return split_samples


def verify_counts(
    split_samples: dict[str, dict[str, list[Sample]]], protocol: Protocol
) -> list[str]:
    """Compare per-modality counts against the published protocol sizes."""
    if protocol.expected is None:
        return []
    train_expected, evaluation_expected = protocol.expected
    expected = {TRAIN_SPLIT: train_expected, **evaluation_expected}
    problems = []
    for split, per_modality in split_samples.items():
        for modality, samples in per_modality.items():
            if len(samples) != expected[split]:
                problems.append(
                    f"{split}/{modality}: found {len(samples):,} samples, "
                    f"expected {expected[split]:,}"
                )
    return problems


def check_link_modalities(modalities: tuple[str, ...]) -> None:
    """Reject a link tree that its own consumers cannot read.

    The pre-split loaders find one modality by globbing and derive the others by
    substituting the modality directory into the path. A tree missing a modality
    therefore fails when the first batch loads, not when it is built, so refuse to
    build one. Filtering modalities is still fine for the import path, which
    returns file lists rather than a directory layout.
    """
    missing = [modality for modality in MODALITIES if modality not in modalities]
    if missing:
        raise ValueError(
            "a link tree needs every modality: loaders derive sibling paths by "
            f"substitution and would fail on the missing {', '.join(missing)}. "
            "Drop --modalities to emit all three, or use --dry-run, or import "
            "build_split for a single-modality file list."
        )


def create_link(source: Path, destination: Path, *, force: bool) -> str:
    """Create one absolute symbolic link and report what happened."""
    if destination.is_symlink():
        if destination.resolve(strict=False) == source:
            return "existing"
        if not force:
            raise FileExistsError(
                f"{destination} already points elsewhere; pass --force to replace it"
            )
        destination.unlink()
    elif destination.exists():
        if not force:
            raise FileExistsError(
                f"{destination} already exists; pass --force to replace it"
            )
        destination.unlink()

    destination.symlink_to(source)
    return "created"


def emit_links(
    split_samples: dict[str, dict[str, list[Sample]]],
    protocol: Protocol,
    link_root: Path,
    *,
    force: bool,
) -> tuple[int, int]:
    """Materialise the symlink tree and return (created, already correct)."""
    created = 0
    existing = 0
    for split, per_modality in split_samples.items():
        split_directory = protocol.directory_name(split)
        for modality, samples in per_modality.items():
            for sample in samples:
                # The doubled scene directory mirrors the release layout, which
                # the pre-split loaders glob verbatim.
                destination_directory = (
                    link_root / split_directory / modality / sample.scene / sample.scene
                )
                destination_directory.mkdir(parents=True, exist_ok=True)
                status = create_link(
                    sample.path.resolve(),
                    destination_directory / sample.path.name,
                    force=force,
                )
                created += status == "created"
                existing += status == "existing"
    return created, existing


def report(
    split_samples: dict[str, dict[str, list[Sample]]], protocol: Protocol
) -> None:
    """Print per-scene counts followed by the per-split totals."""
    per_scene: dict[tuple[str, str, str], int] = defaultdict(int)
    for split, per_modality in split_samples.items():
        for modality, samples in per_modality.items():
            for sample in samples:
                per_scene[(split, modality, sample.scene)] += 1

    print(f"{'Split':<24} {'Modality':<10} {'Scene':<8} {'Samples':>10}")
    print("-" * 56)
    for key in sorted(per_scene):
        split, modality, scene = key
        print(f"{split:<24} {modality:<10} {scene:<8} {per_scene[key]:>10,}")

    print("-" * 56)
    for split, per_modality in split_samples.items():
        modality, samples = next(iter(per_modality.items()))
        directory = protocol.directory_name(split)
        print(
            f"{split:<24} {len(samples):>10,} samples per modality "
            f"-> {directory}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an XRF55 evaluation split from the raw release layout."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(os.environ.get("XRF55_RAW_ROOT") or DEFAULT_RAW_ROOT),
        help=(
            "Dataset root holding part1/ and part2/. Defaults to $XRF55_RAW_ROOT "
            f"when set, otherwise {DEFAULT_RAW_ROOT}."
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=PROTOCOL_NAMES,
        default="trial_split",
        help="Split protocol to build (default: trial_split).",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=MODALITIES,
        default=list(MODALITIES),
        help="Modalities to include (default: all three).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "XRF55",
        help="Output root; the tree is written to a {protocol} subdirectory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the source and report counts without writing anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing files or links at destination paths.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Warn instead of failing when counts differ from the paper.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.is_dir():
        raise SystemExit(
            f"source directory does not exist: {raw_root}\n"
            "pass --raw-root or export XRF55_RAW_ROOT to point at the release"
        )

    protocol = build_protocol(args.protocol)
    modalities = tuple(
        modality for modality in MODALITIES if modality in set(args.modalities)
    )
    if not args.dry_run:
        try:
            check_link_modalities(modalities)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    split_samples = build_split(raw_root, protocol, modalities)

    problems = verify_counts(split_samples, protocol)
    if problems:
        message = "unexpected split sizes:\n" + "\n".join(f"  {p}" for p in problems)
        if not args.allow_count_mismatch:
            raise SystemExit(message + "\npass --allow-count-mismatch to continue")
        print(f"Warning: {message}")

    # Nesting by protocol keeps two splits from merging into one train_data.
    link_root = Path(args.dst).expanduser().resolve() / protocol.name
    print(f"Protocol: {protocol.name}")
    print(f"  {protocol.description}")
    print(f"Source: {raw_root}")
    print(f"Destination: {link_root}")
    print(f"Modalities: {', '.join(modalities)}")
    print()
    report(split_samples, protocol)
    print()

    if args.dry_run:
        print("Dry run complete; nothing was written.")
        return

    created, existing = emit_links(split_samples, protocol, link_root, force=args.force)
    print(f"Links: {created:,} created, {existing:,} already correct.")


if __name__ == "__main__":
    main()
