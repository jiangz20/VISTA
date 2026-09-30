"""Collective visual experience compilation and teacher-view projection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .hindsight import outcome_status, sanitize_agent_claim


ROOT_ARTIFACT_ID = "ROOT"
SCHEMA_VERSION = "collective_visual_experience_v3"


def _intersection_area(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _iou(a, b) -> float:
    intersection = _intersection_area(a, b)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union else 0.0


def _overlap_coefficient(a, b) -> float:
    denominator = min(_area(a), _area(b))
    return _intersection_area(a, b) / denominator if denominator else 0.0


def _contains(a, b) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _hamming(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def _id_number(value: str) -> int:
    digits = "".join(character for character in str(value) if character.isdigit())
    return int(digits or 0)


@dataclass(frozen=True, order=True)
class ExperienceRelation:
    relation_type: str
    source: str
    target: str


@dataclass(frozen=True)
class OutcomeRecord:
    trajectory_id: str
    status: str
    trajectory_valid: bool
    crop_count: int
    failure_type: str | None


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    trajectory_id: str
    step: int
    kind: str
    tool_name: str | None
    source_artifact_id: str | None
    produced_artifact_id: str | None
    bbox_2d: tuple[float, float, float, float] | None
    label: str
    stated_intent: str
    final_answer: str | None
    visible_artifact_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    decision_ids: tuple[str, ...]
    outcome: OutcomeRecord | None


@dataclass(frozen=True)
class ArtifactOccurrence:
    occurrence_id: str
    trajectory_id: str
    decision_id: str
    raw_artifact_id: str
    cluster_id: str
    parent_artifact_id: str
    root_bbox: tuple[float, float, float, float]
    sha256: str
    phash: int


@dataclass(frozen=True)
class ArtifactCluster:
    cluster_id: str
    representative_occurrence_id: str
    occurrence_ids: tuple[str, ...]
    root_bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ArtifactOutcomeAssociation:
    artifact_id: str
    association_type: str
    successful_trajectory_ids: tuple[str, ...]
    failed_trajectory_ids: tuple[str, ...]
    unverified_trajectory_ids: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosticRecord:
    diagnostic_type: str
    target_trajectory_id: str
    artifact_ids: tuple[str, ...]


@dataclass
class CollectiveExperience:
    task_id: str
    policy_version: int
    trajectories: tuple[TrajectoryRecord, ...]
    decisions: tuple[DecisionRecord, ...]
    occurrences: tuple[ArtifactOccurrence, ...]
    artifacts: tuple[ArtifactCluster, ...]
    relations: tuple[ExperienceRelation, ...]
    associations: tuple[ArtifactOutcomeAssociation, ...]
    diagnostics: tuple[DiagnosticRecord, ...]
    artifact_images: dict[str, Any] = field(default_factory=dict, repr=False)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self, include_images: bool = False) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "policy_version": self.policy_version,
            "trajectories": [asdict(row) for row in self.trajectories],
            "decisions": [asdict(row) for row in self.decisions],
            "occurrences": [asdict(row) for row in self.occurrences],
            "artifacts": [asdict(row) for row in self.artifacts],
            "relations": [asdict(row) for row in self.relations],
            "associations": [asdict(row) for row in self.associations],
            "diagnostics": [asdict(row) for row in self.diagnostics],
        }
        if include_images:
            data["artifact_images"] = self.artifact_images
        return data

    def dumps(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def artifact_by_id(self) -> dict[str, ArtifactCluster]:
        return {artifact.cluster_id: artifact for artifact in self.artifacts}

    def occurrence_by_id(self) -> dict[str, ArtifactOccurrence]:
        return {occurrence.occurrence_id: occurrence for occurrence in self.occurrences}

    def decision_by_id(self) -> dict[str, DecisionRecord]:
        return {decision.decision_id: decision for decision in self.decisions}


@dataclass(frozen=True)
class ExperienceView:
    artifact_ids: tuple[str, ...]
    decision_ids: tuple[str, ...]
    artifact_budget: int
    selection_policy: str = "collective_coverage"
    target_trajectory_id: str | None = None


@dataclass(frozen=True)
class DecisionAlignment:
    target_trajectory_id: str
    target_step: int
    target_decision_id: str | None
    student_visible_artifact_ids: tuple[str, ...]
    peer_only_artifact_ids: tuple[str, ...]
    self_future_artifact_ids: tuple[str, ...]


class CollectiveExperienceCompiler:
    """Compile complete grouped rollout experience without teacher-budget pruning."""

    def __init__(
        self,
        near_duplicate_iou: float = 0.90,
        phash_distance: int = 4,
        semantic_deduper: Any | None = None,
        max_intent_chars: int = 240,
    ):
        self.near_duplicate_iou = float(near_duplicate_iou)
        self.phash_distance = int(phash_distance)
        self.semantic_deduper = semantic_deduper
        self.max_intent_chars = int(max_intent_chars)
        if not 0.0 <= self.near_duplicate_iou <= 1.0:
            raise ValueError("near_duplicate_iou must be in [0, 1]")
        if self.phash_distance < 0:
            raise ValueError("phash_distance must be non-negative")
        if self.max_intent_chars < 0:
            raise ValueError("max_intent_chars must be non-negative")

    @staticmethod
    def _collect_outcomes(events: list[dict[str, Any]]) -> dict[str, OutcomeRecord]:
        outcomes: dict[str, OutcomeRecord] = {}
        for event in events:
            trajectory_id = str(event.get("trajectory_id", ""))
            raw = event.get("trajectory_outcome")
            if not trajectory_id or not isinstance(raw, Mapping):
                continue
            record = OutcomeRecord(
                trajectory_id=trajectory_id,
                status=outcome_status(raw),
                trajectory_valid=bool(raw.get("trajectory_valid", False)),
                crop_count=int(raw.get("crop_count", 0)),
                failure_type=(str(raw["failure_type"]) if raw.get("failure_type") is not None else None),
            )
            previous = outcomes.get(trajectory_id)
            if previous is not None and previous != record:
                raise ValueError(f"inconsistent trajectory outcomes for {trajectory_id!r}")
            outcomes[trajectory_id] = record
        return outcomes

    def compile(
        self,
        task_id: str,
        policy_version: int,
        events: Iterable[dict[str, Any]],
    ) -> CollectiveExperience:
        all_events = [dict(event) for event in events if event]
        all_events.sort(
            key=lambda event: (
                str(event.get("trajectory_id", "")),
                int(event.get("step", -1)),
                str(event.get("kind", "")),
            )
        )
        outcomes = self._collect_outcomes(all_events)
        crop_events = [event for event in all_events if event.get("kind") == "crop"]

        cluster_rows: list[dict[str, Any]] = []
        raw_to_cluster: dict[tuple[str, str], str] = {}
        raw_event_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        representative_images: dict[str, Any] = {}
        occurrence_rows: list[dict[str, Any]] = []

        for event in crop_events:
            trajectory_id = str(event["trajectory_id"])
            step = int(event["step"])
            raw_artifact = dict(event["artifact"])
            raw_id = str(raw_artifact["artifact_id"])
            key = (trajectory_id, raw_id)
            if key in raw_event_by_key:
                raise ValueError(f"duplicate raw artifact occurrence {key!r}")
            raw_event_by_key[key] = event
            root_bbox = tuple(float(value) for value in raw_artifact["root_bbox"])
            match = None
            for candidate in cluster_rows:
                exact = (
                    candidate["sha256"] == str(raw_artifact["sha256"]) and tuple(candidate["root_bbox"]) == root_bbox
                )
                spatially_near = _iou(candidate["root_bbox"], root_bbox) >= self.near_duplicate_iou
                visually_near = _hamming(candidate["phash"], int(raw_artifact["phash"])) <= self.phash_distance
                if not visually_near and self.semantic_deduper is not None and spatially_near:
                    visually_near = bool(
                        self.semantic_deduper.are_similar(candidate["artifact_image"], event["artifact_image"])
                    )
                near = spatially_near or visually_near
                if exact or near:
                    match = candidate
                    break
            if match is None:
                cluster_id = f"V{len(cluster_rows) + 1}"
                match = {
                    "cluster_id": cluster_id,
                    "root_bbox": root_bbox,
                    "sha256": str(raw_artifact["sha256"]),
                    "phash": int(raw_artifact["phash"]),
                    "artifact_image": event["artifact_image"],
                    "occurrence_ids": [],
                }
                cluster_rows.append(match)
                representative_images[cluster_id] = event["artifact_image"]
            cluster_id = str(match["cluster_id"])
            raw_to_cluster[key] = cluster_id
            occurrence_id = f"O:{trajectory_id}:{raw_id}"
            match["occurrence_ids"].append(occurrence_id)
            occurrence_rows.append(
                {
                    "occurrence_id": occurrence_id,
                    "trajectory_id": trajectory_id,
                    "decision_id": f"D:{trajectory_id}:{step}",
                    "raw_artifact_id": raw_id,
                    "cluster_id": cluster_id,
                    "parent_raw_artifact_id": raw_artifact.get("parent_artifact_id"),
                    "root_bbox": root_bbox,
                    "sha256": str(raw_artifact["sha256"]),
                    "phash": int(raw_artifact["phash"]),
                }
            )

        occurrences: list[ArtifactOccurrence] = []
        for row in occurrence_rows:
            parent_raw_id = row.pop("parent_raw_artifact_id")
            if parent_raw_id in {None, "A0"}:
                parent_artifact_id = ROOT_ARTIFACT_ID
            else:
                parent_key = (row["trajectory_id"], str(parent_raw_id))
                if parent_key not in raw_to_cluster:
                    raise ValueError(f"missing parent occurrence for {row['occurrence_id']!r}: {parent_key!r}")
                parent_artifact_id = raw_to_cluster[parent_key]
            occurrences.append(
                ArtifactOccurrence(
                    parent_artifact_id=parent_artifact_id,
                    **row,
                )
            )

        artifacts = tuple(
            ArtifactCluster(
                cluster_id=str(row["cluster_id"]),
                representative_occurrence_id=str(row["occurrence_ids"][0]),
                occurrence_ids=tuple(sorted(row["occurrence_ids"])),
                root_bbox=tuple(row["root_bbox"]),
            )
            for row in cluster_rows
        )
        trajectory_raw_history: dict[str, list[tuple[int, str]]] = {}
        for event in crop_events:
            trajectory_raw_history.setdefault(str(event["trajectory_id"]), []).append(
                (int(event["step"]), str(event["artifact"]["artifact_id"]))
            )
        for rows in trajectory_raw_history.values():
            rows.sort()

        decisions: list[DecisionRecord] = []
        for event in all_events:
            kind = str(event.get("kind", ""))
            if kind not in {"crop", "final", "premature_answer"}:
                continue
            trajectory_id = str(event.get("trajectory_id", ""))
            if not trajectory_id:
                continue
            step = int(event.get("step", -1))
            visible_raw = event.get("visible_artifact_ids")
            if visible_raw is None:
                visible_raw = [ROOT_ARTIFACT_ID]
                visible_raw.extend(
                    raw_id for prior_step, raw_id in trajectory_raw_history.get(trajectory_id, []) if prior_step < step
                )
            visible: list[str] = []
            for raw_id in visible_raw:
                raw_id = str(raw_id)
                canonical_id = (
                    ROOT_ARTIFACT_ID
                    if raw_id in {ROOT_ARTIFACT_ID, "A0"}
                    else raw_to_cluster.get((trajectory_id, raw_id))
                )
                if canonical_id and canonical_id not in visible:
                    visible.append(canonical_id)

            action = dict(event.get("action") or {})
            source_raw_id = action.get("source_artifact_id")
            source_artifact_id = None
            if source_raw_id is not None:
                source_artifact_id = (
                    ROOT_ARTIFACT_ID
                    if str(source_raw_id) == "A0"
                    else raw_to_cluster.get((trajectory_id, str(source_raw_id)))
                )
            produced_artifact_id = None
            if kind == "crop":
                produced_artifact_id = raw_to_cluster[(trajectory_id, str(event["artifact"]["artifact_id"]))]
            bbox = action.get("bbox_2d")
            decisions.append(
                DecisionRecord(
                    decision_id=f"D:{trajectory_id}:{step}",
                    trajectory_id=trajectory_id,
                    step=step,
                    kind=kind,
                    tool_name=("image_zoom_in_tool" if kind == "crop" else None),
                    source_artifact_id=source_artifact_id,
                    produced_artifact_id=produced_artifact_id,
                    bbox_2d=(tuple(float(value) for value in bbox) if bbox is not None else None),
                    label=sanitize_agent_claim(action.get("label", ""), self.max_intent_chars),
                    stated_intent=sanitize_agent_claim(action.get("rationale", ""), self.max_intent_chars),
                    # Keep the schema field for compatibility, but never copy
                    # generated answers into collective teacher experience.
                    # Raw events retain them for reward/outcome verification.
                    final_answer=None,
                    visible_artifact_ids=tuple(visible),
                )
            )

        decisions.sort(key=lambda row: (row.trajectory_id, row.step, row.decision_id))
        decision_ids_by_trajectory: dict[str, list[str]] = {}
        for decision in decisions:
            decision_ids_by_trajectory.setdefault(decision.trajectory_id, []).append(decision.decision_id)
        trajectory_ids = set(decision_ids_by_trajectory) | set(outcomes)
        trajectories = tuple(
            TrajectoryRecord(
                trajectory_id=trajectory_id,
                decision_ids=tuple(decision_ids_by_trajectory.get(trajectory_id, [])),
                outcome=outcomes.get(trajectory_id),
            )
            for trajectory_id in sorted(trajectory_ids)
        )

        relations: set[ExperienceRelation] = set()
        decision_by_id = {decision.decision_id: decision for decision in decisions}
        for trajectory in trajectories:
            ordered = [decision_by_id[decision_id] for decision_id in trajectory.decision_ids]
            for previous, current in zip(ordered, ordered[1:]):
                relations.add(ExperienceRelation("next", previous.decision_id, current.decision_id))
            for decision in ordered:
                relations.add(ExperienceRelation("belongs_to", decision.decision_id, trajectory.trajectory_id))
                if decision.source_artifact_id is not None:
                    relations.add(ExperienceRelation("uses", decision.decision_id, decision.source_artifact_id))
                if decision.produced_artifact_id is not None:
                    relations.add(
                        ExperienceRelation(
                            "produces",
                            decision.decision_id,
                            decision.produced_artifact_id,
                        )
                    )
        for occurrence in occurrences:
            relations.add(ExperienceRelation("member_of", occurrence.occurrence_id, occurrence.cluster_id))
            relations.add(
                ExperienceRelation(
                    "derived_from",
                    occurrence.cluster_id,
                    occurrence.parent_artifact_id,
                )
            )
        for index, left in enumerate(artifacts):
            for right in artifacts[index + 1 :]:
                if _contains(left.root_bbox, right.root_bbox):
                    relations.add(ExperienceRelation("contains", left.cluster_id, right.cluster_id))
                elif _contains(right.root_bbox, left.root_bbox):
                    relations.add(ExperienceRelation("contains", right.cluster_id, left.cluster_id))
                elif _iou(left.root_bbox, right.root_bbox) > 0:
                    relations.add(ExperienceRelation("overlaps", left.cluster_id, right.cluster_id))

        occurrence_by_id = {occurrence.occurrence_id: occurrence for occurrence in occurrences}
        associations: list[ArtifactOutcomeAssociation] = []
        observed_by_trajectory: dict[str, set[str]] = {}
        for artifact in artifacts:
            source_trajectories = {
                occurrence_by_id[occurrence_id].trajectory_id for occurrence_id in artifact.occurrence_ids
            }
            successful = sorted(
                trajectory_id
                for trajectory_id in source_trajectories
                if outcomes.get(trajectory_id) and outcomes[trajectory_id].status == "success"
            )
            failed = sorted(
                trajectory_id
                for trajectory_id in source_trajectories
                if outcomes.get(trajectory_id) and outcomes[trajectory_id].status == "failure"
            )
            unverified = sorted(source_trajectories.difference(successful).difference(failed))
            if successful and failed:
                association_type = "mixed_outcome_observation"
            elif successful:
                association_type = "observed_in_success"
            elif failed:
                association_type = "observed_in_failure"
            else:
                association_type = "unverified"
            associations.append(
                ArtifactOutcomeAssociation(
                    artifact_id=artifact.cluster_id,
                    association_type=association_type,
                    successful_trajectory_ids=tuple(successful),
                    failed_trajectory_ids=tuple(failed),
                    unverified_trajectory_ids=tuple(unverified),
                )
            )
            for trajectory_id in source_trajectories:
                observed_by_trajectory.setdefault(trajectory_id, set()).add(artifact.cluster_id)

        success_artifact_ids = {
            association.artifact_id for association in associations if association.successful_trajectory_ids
        }
        diagnostics: list[DiagnosticRecord] = []
        if success_artifact_ids:
            for trajectory_id, outcome in sorted(outcomes.items()):
                if outcome.status != "failure":
                    continue
                missing = tuple(
                    sorted(
                        success_artifact_ids.difference(observed_by_trajectory.get(trajectory_id, set())),
                        key=_id_number,
                    )
                )
                if missing:
                    diagnostics.append(
                        DiagnosticRecord(
                            diagnostic_type="coverage_gap_candidate",
                            target_trajectory_id=trajectory_id,
                            artifact_ids=missing,
                        )
                    )

        return CollectiveExperience(
            task_id=str(task_id),
            policy_version=int(policy_version),
            trajectories=trajectories,
            decisions=tuple(decisions),
            occurrences=tuple(sorted(occurrences, key=lambda row: row.occurrence_id)),
            artifacts=tuple(sorted(artifacts, key=lambda row: _id_number(row.cluster_id))),
            relations=tuple(sorted(relations)),
            associations=tuple(sorted(associations, key=lambda row: _id_number(row.artifact_id))),
            diagnostics=tuple(diagnostics),
            artifact_images=representative_images,
        )


class ExperienceViewProjector:
    """Project a dependency-closed teacher view from full experience.

    The default view is conditioned on the current rollout decision when one is
    available. This is deliberately trajectory/decision-level conditioning:
    it produces one view per sampled response and does not rebuild a view for
    every token or run an additional teacher forward pass.
    """

    def __init__(self, artifact_budget: int, min_visual_novelty: float = 0.0):
        self.artifact_budget = int(artifact_budget)
        self.min_visual_novelty = float(min_visual_novelty)
        if self.artifact_budget < 1:
            raise ValueError("artifact_budget must be at least 1")
        if not 0.0 <= self.min_visual_novelty <= 1.0:
            raise ValueError("min_visual_novelty must be in [0, 1]")

    @staticmethod
    def _parent_map(experience: CollectiveExperience) -> dict[str, set[str]]:
        parents: dict[str, set[str]] = {}
        for relation in experience.relations:
            if relation.relation_type == "derived_from":
                parents.setdefault(relation.source, set()).add(relation.target)
        return parents

    @staticmethod
    def _closure(artifact_id: str, parents: Mapping[str, set[str]]) -> set[str]:
        closure = {artifact_id}
        stack = [artifact_id]
        while stack:
            current = stack.pop()
            for parent in parents.get(current, set()):
                if parent == ROOT_ARTIFACT_ID or parent in closure:
                    continue
                closure.add(parent)
                stack.append(parent)
        return closure

    @staticmethod
    def _decision_context(
        experience: CollectiveExperience,
        event: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(event, Mapping):
            return None
        trajectory_id = str(event.get("trajectory_id", ""))
        if not trajectory_id:
            return None
        try:
            step = int(event.get("step", -1))
        except (TypeError, ValueError):
            step = -1
        decision = experience.decision_by_id().get(f"D:{trajectory_id}:{step}")
        if decision is None:
            return None
        occurrence_by_id = experience.occurrence_by_id()
        association_by_id = {association.artifact_id: association for association in experience.associations}
        peer_trajectories: dict[str, set[str]] = {}
        future_self: set[str] = set()
        for artifact in experience.artifacts:
            source_trajectories = {
                occurrence_by_id[occurrence_id].trajectory_id for occurrence_id in artifact.occurrence_ids
            }
            peer_trajectories[artifact.cluster_id] = source_trajectories.difference({trajectory_id})
            if trajectory_id in source_trajectories and not peer_trajectories[artifact.cluster_id]:
                if any(
                    experience.decision_by_id()[occurrence_by_id[occurrence_id].decision_id].step > step
                    for occurrence_id in artifact.occurrence_ids
                ):
                    future_self.add(artifact.cluster_id)
        return {
            "trajectory_id": trajectory_id,
            "step": step,
            "visible": set(decision.visible_artifact_ids),
            "peer_trajectories": peer_trajectories,
            "future_self": future_self,
            "association_by_id": association_by_id,
        }

    def project(
        self,
        experience: CollectiveExperience,
        event: Mapping[str, Any] | None = None,
    ) -> ExperienceView:
        """Select a view, optionally conditioned on one rollout decision.

        With a target event, visible student artifacts and peer artifacts that
        occurred in successful trajectories receive priority. Future-only
        artifacts from the target rollout are deprioritized, while dependency
        closure is still preserved. Without an event, the original shared
        collective-coverage policy is used as a deterministic fallback.
        """
        decision_context = self._decision_context(experience, event)
        if len(experience.artifacts) <= self.artifact_budget:
            selected = {artifact.cluster_id for artifact in experience.artifacts}
        else:
            artifact_by_id = experience.artifact_by_id()
            occurrence_by_id = experience.occurrence_by_id()
            association_by_id = {association.artifact_id: association for association in experience.associations}
            parents = self._parent_map(experience)
            selected: set[str] = set()
            covered_trajectories: set[str] = set()
            covered_outcomes: set[str] = set()
            while len(selected) < self.artifact_budget:
                candidates = []
                for artifact in experience.artifacts:
                    if artifact.cluster_id in selected:
                        continue
                    closure = self._closure(artifact.cluster_id, parents)
                    additions = closure.difference(selected)
                    if len(selected) + len(additions) > self.artifact_budget:
                        continue
                    source_trajectories = {
                        occurrence_by_id[occurrence_id].trajectory_id for occurrence_id in artifact.occurrence_ids
                    }
                    association = association_by_id[artifact.cluster_id]
                    outcome_types = {
                        "success" if association.successful_trajectory_ids else "",
                        "failure" if association.failed_trajectory_ids else "",
                        "unverified" if association.unverified_trajectory_ids else "",
                    }.difference({""})
                    if selected:
                        similarities = []
                        for selected_id in selected:
                            spatial_similarity = _iou(
                                artifact.root_bbox,
                                artifact_by_id[selected_id].root_bbox,
                            )
                            similarities.append(spatial_similarity)
                        novelty = 1.0 - max(similarities)
                    else:
                        novelty = 1.0
                    if selected and novelty < self.min_visual_novelty:
                        continue
                    source_gain = len(source_trajectories.difference(covered_trajectories)) / max(
                        1, len(source_trajectories)
                    )
                    outcome_gain = len(outcome_types.difference(covered_outcomes)) / 3.0
                    rare_bonus = 1.0 / max(1, len(source_trajectories))
                    if decision_context is None:
                        score = (3.0 * novelty + 2.0 * source_gain + outcome_gain + 0.5 * rare_bonus) / max(
                            1, len(additions)
                        )
                    else:
                        peer_trajectories = decision_context["peer_trajectories"][artifact.cluster_id]
                        successful_peer_trajectories = peer_trajectories.intersection(
                            association.successful_trajectory_ids
                        )
                        visible_bonus = float(artifact.cluster_id in decision_context["visible"])
                        peer_bonus = len(peer_trajectories) / max(1, len(source_trajectories))
                        successful_peer_bonus = len(successful_peer_trajectories) / max(1, len(peer_trajectories))
                        future_penalty = float(
                            artifact.cluster_id in decision_context["future_self"]
                            and artifact.cluster_id not in decision_context["visible"]
                        )
                        score = (
                            6.0 * visible_bonus
                            + 4.0 * peer_bonus
                            + 2.0 * successful_peer_bonus
                            + 1.5 * novelty
                            + 0.5 * source_gain
                            + 0.5 * rare_bonus
                            - 2.0 * future_penalty
                        ) / max(1, len(additions))
                    candidates.append(
                        (
                            score,
                            novelty,
                            -len(additions),
                            -_id_number(artifact.cluster_id),
                            artifact.cluster_id,
                            additions,
                            source_trajectories,
                            outcome_types,
                        )
                    )
                if not candidates:
                    break
                chosen = max(candidates)
                selected.update(chosen[5])
                covered_trajectories.update(chosen[6])
                covered_outcomes.update(chosen[7])

        selected_decisions = {
            decision.decision_id for decision in experience.decisions if decision.produced_artifact_id in selected
        }
        return ExperienceView(
            artifact_ids=tuple(sorted(selected, key=_id_number)),
            decision_ids=tuple(sorted(selected_decisions)),
            artifact_budget=self.artifact_budget,
            selection_policy=("decision_conditioned" if decision_context is not None else "collective_coverage"),
            target_trajectory_id=(decision_context["trajectory_id"] if decision_context is not None else None),
        )


class DecisionAligner:
    """Describe how one unchanged student decision relates to a shared view."""

    def align(
        self,
        experience: CollectiveExperience,
        view: ExperienceView,
        event: Mapping[str, Any] | None,
    ) -> DecisionAlignment:
        if not isinstance(event, Mapping):
            return DecisionAlignment("", -1, None, (), (), ())
        trajectory_id = str(event.get("trajectory_id", ""))
        step = int(event.get("step", -1))
        decision_id = f"D:{trajectory_id}:{step}" if trajectory_id else None
        decision = experience.decision_by_id().get(str(decision_id))
        selected = set(view.artifact_ids)
        visible = selected.intersection(decision.visible_artifact_ids) if decision is not None else set()
        occurrence_by_id = experience.occurrence_by_id()
        peer: set[str] = set()
        self_future: set[str] = set()
        for artifact in experience.artifacts:
            if artifact.cluster_id not in selected or artifact.cluster_id in visible:
                continue
            occurrences = [occurrence_by_id[occurrence_id] for occurrence_id in artifact.occurrence_ids]
            if any(occurrence.trajectory_id != trajectory_id for occurrence in occurrences):
                peer.add(artifact.cluster_id)
            elif any(
                occurrence.trajectory_id == trajectory_id
                and experience.decision_by_id()[occurrence.decision_id].step > step
                for occurrence in occurrences
            ):
                self_future.add(artifact.cluster_id)
        return DecisionAlignment(
            target_trajectory_id=trajectory_id,
            target_step=step,
            target_decision_id=(decision.decision_id if decision else None),
            student_visible_artifact_ids=tuple(sorted(visible, key=_id_number)),
            peer_only_artifact_ids=tuple(sorted(peer, key=_id_number)),
            self_future_artifact_ids=tuple(sorted(self_future.difference(peer), key=_id_number)),
        )


class ExperienceSerializer:
    """Render teacher-only visual context without generated final answers."""

    def __init__(
        self,
        include_outcomes: bool = True,
        compact_ids: bool = True,
        compact_graph: bool = True,
    ):
        self.include_outcomes = bool(include_outcomes)
        self.compact_ids = bool(compact_ids)
        self.compact_graph = bool(compact_graph)

    def trajectory_aliases(self, experience: CollectiveExperience) -> dict[str, str]:
        if not self.compact_ids:
            return {}
        return {
            trajectory.trajectory_id: f"T{index}" for index, trajectory in enumerate(experience.trajectories, start=1)
        }

    @staticmethod
    def _alias_trajectory(trajectory_id: str, aliases: Mapping[str, str]) -> str:
        return aliases.get(trajectory_id, trajectory_id)

    @classmethod
    def _alias_node(cls, node_id: str, aliases: Mapping[str, str]) -> str:
        if node_id in aliases:
            return aliases[node_id]
        if node_id.startswith("D:"):
            _, trajectory_id, step = node_id.rsplit(":", 2)
            return f"D:{cls._alias_trajectory(trajectory_id, aliases)}:{step}"
        return node_id

    def render_view(self, experience: CollectiveExperience, view: ExperienceView) -> tuple[str, list[Any]]:
        selected_artifacts = set(view.artifact_ids)
        selected_decisions = set(view.decision_ids)
        trajectory_aliases = self.trajectory_aliases(experience)
        occurrence_by_id = experience.occurrence_by_id()
        association_by_id = {association.artifact_id: association for association in experience.associations}
        lines = [
            "[COLLECTIVE_VISUAL_EXPERIENCE]",
            f"task={json.dumps(experience.task_id, ensure_ascii=False)} policy_version={experience.policy_version}",
            (
                "outcome_semantics=categorical trajectory hindsight; association is not causal evidence"
                if self.include_outcomes
                else "outcome_semantics=hidden"
            ),
        ]
        for trajectory in experience.trajectories:
            trajectory_id = json.dumps(
                self._alias_trajectory(trajectory.trajectory_id, trajectory_aliases),
                ensure_ascii=False,
            )
            if self.compact_graph:
                if self.include_outcomes and trajectory.outcome is not None:
                    outcome = trajectory.outcome
                    failure = f" failure={outcome.failure_type}" if outcome.failure_type else ""
                    lines.append(
                        f"trajectory={trajectory_id} outcome={outcome.status} "
                        f"valid={int(outcome.trajectory_valid)} crops={outcome.crop_count} "
                        f"decisions={len(trajectory.decision_ids)}{failure}"
                    )
                else:
                    lines.append(f"trajectory={trajectory_id} decisions={len(trajectory.decision_ids)}")
            elif self.include_outcomes and trajectory.outcome is not None:
                lines.append(
                    "trajectory={trajectory} outcome={status} valid={valid} "
                    "failure_type={failure_type} crop_count={crop_count}".format(
                        trajectory=trajectory_id,
                        status=trajectory.outcome.status,
                        valid=str(trajectory.outcome.trajectory_valid).lower(),
                        failure_type=trajectory.outcome.failure_type or "none",
                        crop_count=trajectory.outcome.crop_count,
                    )
                )
            else:
                lines.append("trajectory=" + trajectory_id)
        for decision in experience.decisions:
            if decision.decision_id not in selected_decisions:
                continue
            decision_id = self._alias_node(decision.decision_id, trajectory_aliases)
            source = (
                self._alias_node(decision.source_artifact_id, trajectory_aliases)
                if decision.source_artifact_id
                else "none"
            )
            produces = (
                self._alias_node(decision.produced_artifact_id, trajectory_aliases)
                if decision.produced_artifact_id
                else "none"
            )
            if self.compact_graph:
                lines.append(
                    "decision={decision} kind={kind} tool={tool} src={source} "
                    "produces={produces} bbox={bbox} label={label} intent={intent}".format(
                        decision=json.dumps(decision_id, ensure_ascii=False),
                        kind=decision.kind,
                        tool=decision.tool_name or "none",
                        source=source,
                        produces=produces,
                        bbox=list(decision.bbox_2d) if decision.bbox_2d else [],
                        label=json.dumps(decision.label, ensure_ascii=False),
                        intent=json.dumps(decision.stated_intent, ensure_ascii=False),
                    )
                )
            else:
                lines.append(
                    "decision={decision} trajectory={trajectory} step={step} "
                    "tool={tool} source={source} produces={produces} bbox={bbox} "
                    "label={label} intent={intent}".format(
                        decision=json.dumps(decision_id, ensure_ascii=False),
                        trajectory=json.dumps(
                            self._alias_trajectory(decision.trajectory_id, trajectory_aliases),
                            ensure_ascii=False,
                        ),
                        step=decision.step,
                        tool=decision.tool_name or "none",
                        source=source,
                        produces=produces,
                        bbox=list(decision.bbox_2d) if decision.bbox_2d else [],
                        label=json.dumps(decision.label, ensure_ascii=False),
                        intent=json.dumps(decision.stated_intent, ensure_ascii=False),
                    )
                )
        images: list[Any] = []
        for artifact in experience.artifacts:
            if artifact.cluster_id not in selected_artifacts:
                continue
            source_trajectories = sorted(
                {
                    self._alias_trajectory(
                        occurrence_by_id[occurrence_id].trajectory_id,
                        trajectory_aliases,
                    )
                    for occurrence_id in artifact.occurrence_ids
                }
            )
            association = association_by_id[artifact.cluster_id]
            success = [
                self._alias_trajectory(value, trajectory_aliases) for value in association.successful_trajectory_ids
            ]
            failure = [self._alias_trajectory(value, trajectory_aliases) for value in association.failed_trajectory_ids]
            unverified = [
                self._alias_trajectory(value, trajectory_aliases) for value in association.unverified_trajectory_ids
            ]
            if self.compact_graph:
                outcome_text = (
                    f" outcome={association.association_type} "
                    f"ok={json.dumps(success, ensure_ascii=False, separators=(',', ':'))} "
                    f"fail={json.dumps(failure, ensure_ascii=False, separators=(',', ':'))} "
                    f"unknown={json.dumps(unverified, ensure_ascii=False, separators=(',', ':'))}"
                    if self.include_outcomes
                    else ""
                )
                lines.append(
                    f"artifact={artifact.cluster_id} bbox={list(artifact.root_bbox)} "
                    f"seen={json.dumps(source_trajectories, ensure_ascii=False, separators=(',', ':'))}"
                    f"{outcome_text} <image>"
                )
            else:
                outcome_text = (
                    " association={association} success={success} failure={failure} unverified={unverified}".format(
                        association=association.association_type,
                        success=success,
                        failure=failure,
                        unverified=unverified,
                    )
                    if self.include_outcomes
                    else ""
                )
                lines.append(
                    f"artifact={artifact.cluster_id} root_bbox={list(artifact.root_bbox)} "
                    f"observed_by={source_trajectories}{outcome_text} <image>"
                )
            images.append(experience.artifact_images[artifact.cluster_id])
        visible_node_ids = (
            selected_artifacts
            | selected_decisions
            | {trajectory.trajectory_id for trajectory in experience.trajectories}
            | {ROOT_ARTIFACT_ID}
        )
        visible_relations: dict[str, list[str]] = {}
        for relation in experience.relations:
            if relation.relation_type == "member_of":
                continue
            if relation.source not in visible_node_ids or relation.target not in visible_node_ids:
                continue
            source = self._alias_node(relation.source, trajectory_aliases)
            target = self._alias_node(relation.target, trajectory_aliases)
            if self.compact_graph:
                visible_relations.setdefault(relation.relation_type, []).append(f"{source}>{target}")
            else:
                lines.append(f"relation={relation.relation_type}({source},{target})")
        if self.compact_graph:
            for relation_type in sorted(visible_relations):
                lines.append(f"relations={relation_type}:" + ";".join(visible_relations[relation_type]))
        if self.include_outcomes:
            for diagnostic in experience.diagnostics:
                visible_artifacts = tuple(
                    artifact_id for artifact_id in diagnostic.artifact_ids if artifact_id in selected_artifacts
                )
                if visible_artifacts:
                    lines.append(
                        "diagnostic={kind} target={target} artifacts={artifacts}".format(
                            kind=diagnostic.diagnostic_type,
                            target=json.dumps(
                                self._alias_trajectory(
                                    diagnostic.target_trajectory_id,
                                    trajectory_aliases,
                                ),
                                ensure_ascii=False,
                            ),
                            artifacts=list(visible_artifacts),
                        )
                    )
        lines.append("[/COLLECTIVE_VISUAL_EXPERIENCE]")
        return "\n".join(lines), images

    @classmethod
    def render_alignment(
        cls,
        alignment: DecisionAlignment,
        trajectory_aliases: Mapping[str, str] | None = None,
    ) -> str:
        aliases = trajectory_aliases or {}
        return "\n".join(
            [
                "[DECISION_ALIGNMENT]",
                "target_trajectory={trajectory} target_step={step} target_decision={decision}".format(
                    trajectory=json.dumps(
                        cls._alias_trajectory(alignment.target_trajectory_id, aliases),
                        ensure_ascii=False,
                    ),
                    step=alignment.target_step,
                    decision=(
                        cls._alias_node(alignment.target_decision_id, aliases)
                        if alignment.target_decision_id
                        else "unknown"
                    ),
                ),
                "student_visible=" + json.dumps(list(alignment.student_visible_artifact_ids)),
                "peer_only=" + json.dumps(list(alignment.peer_only_artifact_ids)),
                "self_future=" + json.dumps(list(alignment.self_future_artifact_ids)),
                "[/DECISION_ALIGNMENT]",
            ]
        )
