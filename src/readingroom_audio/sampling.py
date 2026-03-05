"""Stratified sample selection from 161 events for audio benchmark.

Classifies events into 8 series groups and 5 content types, then selects
samples using proportional stratified sampling with era/format balancing.
"""

import json
import math
import random
from pathlib import Path

# ── Series grouping (15 series → 8 groups) ──────────────────────────

SERIES_GROUP_MAP = {
    "Other": "other",
    "Readrink (Book Club)": "readrink",
    "Book Club": "readrink",
    "Screening & Talk": "screening",
    "Filmvirus": "screening",
    "Talk Series": "talk_series",
    "Artist Talk": "talk_series",
    "This is Not Fiction": "talk_series",
    "Solidarities": "talk_series",
    "Definitions Series": "definitions",
    "Right Here, Right Now": "definitions",
    "Sleepover": "sleepover",
    "Night School": "night_school",
    "re:reading": "re_reading",
    "Reading Group": "re_reading",
}

# Target samples per group (proportional to population)
GROUP_TARGETS = {
    "other": 12,
    "readrink": 11,
    "screening": 6,
    "talk_series": 3,
    "definitions": 2,
    "sleepover": 2,
    "night_school": 2,
    "re_reading": 2,
}

# Pipeline recommendation per content type (used by batch --auto-pipeline)
FORMAT_PIPELINE_MAP: dict[str, str] = {
    "lecture": "hybrid_demucs_df",
    "panel": "hybrid_demucs_df",
    "book_club": "hybrid_demucs_df",
    "screening": "deepfilter_12dB",
    "performance": "hybrid_demucs_remix",
}


def load_all_events(events_dir: Path) -> list[dict]:
    """Load all event JSONs and enrich with derived fields."""
    events = []
    for f in sorted(events_dir.glob("*.json")):
        with open(f) as fh:
            event = json.load(fh)
        event["_file"] = f.name
        event["_era"] = _classify_era(event.get("event_date", ""))
        event["_format_group"] = _classify_format(event)
        event["_series_group"] = SERIES_GROUP_MAP.get(
            event.get("series", "Other"), "other"
        )
        event["_speaker_count"] = len(
            event.get("baseline_extraction", {}).get("speakers_raw", [])
        )
        events.append(event)
    return events


def classify_event(event: dict) -> dict:
    """Return classification labels for an event."""
    return {
        "series_group": event.get("_series_group", "other"),
        "format_group": event.get("_format_group", "talk"),
        "era": event.get("_era", "middle"),
        "multi_part": event.get("video_count", 1) > 1,
        "duration_group": _classify_duration(event.get("total_duration_seconds", 0)),
    }


def stratified_sample(
    events: list[dict],
    target_n: int = 40,
    seed: int = 42,
) -> list[dict]:
    """Select ~target_n events via proportional stratified sampling.

    Within each series group, prefers coverage of all 3 eras and
    format variety. Returns manifest entries ready for benchmark.
    """
    rng = random.Random(seed)

    # Group events by series_group
    groups: dict[str, list[dict]] = {}
    for event in events:
        sg = event["_series_group"]
        groups.setdefault(sg, []).append(event)

    # Compute per-group targets proportional to population
    total = len(events)
    targets = {}
    for sg, members in groups.items():
        targets[sg] = max(1, math.ceil(len(members) / total * target_n))

    # Adjust to hit target_n (trim from largest group if over)
    while sum(targets.values()) > target_n:
        largest = max(targets, key=targets.get)
        targets[largest] -= 1
    while sum(targets.values()) < target_n:
        largest = max(groups, key=lambda g: len(groups[g]))
        targets[largest] += 1

    selected = []
    for sg, members in sorted(groups.items()):
        n = targets.get(sg, 1)
        picks = _balanced_select(members, n, rng)
        selected.extend(picks)

    return [_make_manifest_entry(e) for e in selected]


def select_representative_video(event: dict) -> dict:
    """Pick a single representative video from an event.

    For multi-part events, pick the longest video.
    For single-video events, use as-is.
    """
    videos = event.get("videos", [])
    if not videos:
        return {}
    if len(videos) == 1:
        return videos[0]
    return max(videos, key=lambda v: v.get("duration_seconds", 0))


def _classify_era(date_str: str) -> str:
    """Classify event date into era bucket."""
    if not date_str:
        return "middle"
    try:
        year = int(date_str[:4])
    except (ValueError, IndexError):
        return "middle"
    if year <= 2013:
        return "early"
    elif year <= 2016:
        return "middle"
    else:
        return "late"


def _classify_format(event: dict) -> str:
    """Classify event into format group from baseline_extraction.formats.

    Order matters: explicit "lecture" check fires before "performance" so that
    events like "Lecture-Performance" (E059) classify as lecture (speech-dominant).
    """
    formats = event.get("baseline_extraction", {}).get("formats", [])
    formats_lower = " ".join(formats).lower()
    if "screening" in formats_lower or "film" in formats_lower:
        return "screening"
    if "book" in formats_lower or "reading" in formats_lower:
        return "book_club"
    if "panel" in formats_lower or "discussion" in formats_lower:
        return "panel"
    if "lecture" in formats_lower:
        return "lecture"
    if "performance" in formats_lower or "การแสดง" in formats_lower:
        return "performance"
    return "lecture"


def _classify_duration(seconds: int) -> str:
    """Classify total event duration."""
    if seconds < 1800:
        return "short"
    elif seconds < 5400:
        return "medium"
    else:
        return "long"


def _balanced_select(
    members: list[dict], n: int, rng: random.Random
) -> list[dict]:
    """Select n members balancing across eras and formats."""
    if n >= len(members):
        return list(members)

    # Bucket by era
    by_era: dict[str, list[dict]] = {}
    for m in members:
        by_era.setdefault(m["_era"], []).append(m)

    # Shuffle within each era bucket
    for era_list in by_era.values():
        rng.shuffle(era_list)

    # Round-robin across eras until we have n
    selected = []
    era_order = ["early", "middle", "late"]
    era_iters = {era: iter(by_era.get(era, [])) for era in era_order}

    while len(selected) < n:
        picked_this_round = False
        for era in era_order:
            if len(selected) >= n:
                break
            try:
                candidate = next(era_iters[era])
                selected.append(candidate)
                picked_this_round = True
            except StopIteration:
                continue
        if not picked_this_round:
            break

    return selected


def _make_manifest_entry(event: dict) -> dict:
    """Build a benchmark manifest entry from an event."""
    video = select_representative_video(event)
    video_id = video.get("id", "")
    event_num = event.get("event_number", 0)
    segment_id = f"E{event_num:03d}_{video_id}"

    return {
        "segment_id": segment_id,
        "event_number": event_num,
        "event_key": event.get("key", ""),
        "event_date": event.get("event_date", ""),
        "video_id": video_id,
        "video_duration_seconds": video.get("duration_seconds", 0),
        "strata": classify_event(event),
        "segment": {},
        "baseline_scores": {},
        "status": {
            "downloaded": False,
            "segment_extracted": False,
            "baseline_scored": False,
        },
    }
