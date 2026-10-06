"""Deduplication utilities for mirror-effect bounding boxes."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, List


def deduplicate_mirror_bboxes(input_data: Dict) -> Dict:
    """
    Remove duplicate bounding boxes caused by mirror reflections.

    When the same actor has multiple bboxes in the same frame (different track_ids),
    keep only the track with the highest average ident_score across all frames.

    Only applies when an actor has >1 bbox in at least one frame.
    Returns a new dict with filtered persons (does not mutate the input).
    """
    frames = input_data.get("frames", [])
    if not frames:
        return input_data

    first_person = next(
        (p for f in frames for p in f.get("persons", [])), None
    )
    if first_person is None or "ident_score" not in first_person or "track_id" not in first_person:
        return input_data

    actors_with_duplicates = set()
    for frame in frames:
        actor_counts: Dict[str, int] = defaultdict(int)
        for person in frame.get("persons", []):
            actor = person.get("actor")
            if actor:
                actor_counts[actor] += 1
        for actor, count in actor_counts.items():
            if count > 1:
                actors_with_duplicates.add(actor)

    if not actors_with_duplicates:
        return input_data

    actor_track_scores: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for frame in frames:
        for person in frame.get("persons", []):
            actor = person.get("actor")
            if actor in actors_with_duplicates:
                track_id = person.get("track_id")
                score = person.get("ident_score", 0.0)
                if track_id is not None:
                    actor_track_scores[actor][track_id].append(score)

    best_track_per_actor: Dict[str, int] = {}
    for actor, tracks in actor_track_scores.items():
        best_track = max(tracks.keys(), key=lambda tid: sum(tracks[tid]) / len(tracks[tid]))
        best_track_per_actor[actor] = best_track

    new_data = copy.deepcopy(input_data)
    total_removed = 0
    for frame in new_data["frames"]:
        filtered_persons = []
        for person in frame.get("persons", []):
            actor = person.get("actor")
            track_id = person.get("track_id")
            if actor in best_track_per_actor:
                if track_id == best_track_per_actor[actor]:
                    filtered_persons.append(person)
                else:
                    total_removed += 1
            else:
                filtered_persons.append(person)
        frame["persons"] = filtered_persons

    if total_removed > 0:
        deduped_actors = ", ".join(
            f"{actor} (kept track {tid})"
            for actor, tid in best_track_per_actor.items()
        )
        print(f"    Mirror dedup: removed {total_removed} duplicate bbox(es) [{deduped_actors}]")

    return new_data
