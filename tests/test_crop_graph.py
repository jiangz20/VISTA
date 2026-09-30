import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from vista.crop import CropAction, CropRuntime, parse_action, parse_final_answer
from vista.experience import (
    CollectiveExperienceCompiler,
    DecisionAligner,
    ExperienceSerializer,
    ExperienceViewProjector,
)
from vista.hindsight import annotate_trajectory_events
from vista.prompts import (
    COLLECTIVE_EXPERIENCE_TEACHER_SYSTEM_INSTRUCTION,
    append_teacher_context,
)


def make_image(value=None):
    if value is None:
        values = np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3) % 255
    else:
        values = np.full((256, 256, 3), value, dtype=np.uint8)
    return Image.fromarray(values.astype(np.uint8))


def crop_event(runtime, trajectory_id, step, source, bbox, *, label="test"):
    visible = tuple(runtime.artifact_order)
    action = CropAction(source, bbox, label)
    artifact = runtime.crop(action)
    return {
        "kind": "crop",
        "task_id": "task",
        "trajectory_id": trajectory_id,
        "step": step,
        "policy_version": 7,
        "visible_artifact_ids": visible,
        "action": {
            "source_artifact_id": source,
            "bbox_2d": bbox,
            "label": label,
            "rationale": "inspect the stated region",
        },
        "artifact": artifact.metadata(),
        "artifact_image": artifact.image,
    }


def final_event(trajectory_id, step, answer_correct, visible, answer=None):
    return {
        "kind": "final",
        "task_id": "task",
        "trajectory_id": trajectory_id,
        "step": step,
        "visible_artifact_ids": tuple(visible),
        "final_answer": answer or ("A" if answer_correct else "B"),
        "answer_correct": answer_correct,
    }


def annotate(events, *, episode_return, trajectory_valid=True):
    rows = [{"experience_event": event} for event in events]
    annotate_trajectory_events(
        rows,
        episode_return=episode_return,
        trajectory_valid=trajectory_valid,
    )
    return [row["experience_event"] for row in rows]


def test_parse_active_visual_actions_and_nested_root_coordinates():
    native = '<tool_call>{"name":"crop","arguments":{"source_artifact_id":"A1","bbox_2d":[0,1,900,999]}}</tool_call>'
    assert parse_action(native).source_artifact_id == "A1"
    qwen = '<tool_call>{"name":"image_zoom_in_tool","arguments":{"img_idx":1,"bbox_2d":[0,0,500,500]}}</tool_call>'
    assert parse_action(qwen, ["A0", "A3"]).source_artifact_id == "A3"
    with pytest.raises(ValueError):
        parse_action('<tool_call>{"name":"crop","arguments":{"bbox_2d":[0,1]}}</tool_call>')
    assert parse_final_answer(r"$\boxed{\text{C}}$") == "C"

    runtime = CropRuntime(make_image(), min_dimension=8)
    first = runtime.crop(CropAction("A0", (100, 200, 900, 800)))
    second = runtime.crop(CropAction(first.artifact_id, (250, 250, 750, 750)))
    assert second.parent_artifact_id == "A1"
    assert second.root_bbox == pytest.approx((300, 350, 700, 650))


def test_complete_experience_is_deterministic_and_preserves_occurrences():
    runtime_a = CropRuntime(make_image(), min_dimension=8)
    runtime_b = CropRuntime(make_image(), min_dimension=8)
    event_a = crop_event(runtime_a, "T0", 0, "A0", (0, 0, 500, 500))
    event_b = crop_event(runtime_b, "T1", 0, "A0", (0, 0, 500, 500))
    compiler = CollectiveExperienceCompiler()

    first = compiler.compile("task", 7, [event_b, event_a])
    second = compiler.compile("task", 7, [event_a, event_b])

    assert first.dumps() == second.dumps()
    assert len(first.artifacts) == 1
    assert len(first.occurrences) == 2
    assert first.artifacts[0].occurrence_ids == ("O:T0:A1", "O:T1:A1")
    payload = first.dumps()
    assert "ground_truth" not in payload
    assert "episode_return" not in payload
    assert "answer_correct" not in payload


def test_near_duplicate_threshold_uses_iou_or_phash():
    image = make_image(127)
    event_a = crop_event(CropRuntime(image, min_dimension=8), "T0", 0, "A0", (0, 0, 500, 500))
    event_b = crop_event(CropRuntime(image, min_dimension=8), "T1", 0, "A0", (13, 13, 513, 513))
    # Force the pHash criterion to fail; the default IoU criterion should still
    # merge these crops under the OR rule.
    event_b["artifact"]["phash"] = event_a["artifact"]["phash"] ^ 0b11111

    assert len(CollectiveExperienceCompiler().compile("task", 1, [event_a, event_b]).artifacts) == 1
    strict = CollectiveExperienceCompiler(near_duplicate_iou=0.98).compile("task", 1, [event_a, event_b])
    assert len(strict.artifacts) == 2


def test_phash_near_duplicate_does_not_require_iou():
    image = make_image(127)
    event_a = crop_event(CropRuntime(image, min_dimension=8), "T0", 0, "A0", (0, 0, 100, 100))
    event_b = crop_event(CropRuntime(image, min_dimension=8), "T1", 0, "A0", (900, 900, 1000, 1000))

    assert len(CollectiveExperienceCompiler().compile("task", 1, [event_a, event_b]).artifacts) == 1


def test_optional_semantic_deduper_only_handles_high_iou_candidates():
    image_a = make_image(11)
    image_b = make_image(222)
    event_a = crop_event(CropRuntime(image_a, min_dimension=8), "T0", 0, "A0", (0, 0, 500, 500))
    event_b = crop_event(CropRuntime(image_b, min_dimension=8), "T1", 0, "A0", (0, 0, 500, 500))
    event_b["artifact"]["phash"] = event_a["artifact"]["phash"] ^ 1

    class FakeDeduper:
        def __init__(self):
            self.calls = 0

        def are_similar(self, left, right):
            self.calls += 1
            return True

    deduper = FakeDeduper()
    experience = CollectiveExperienceCompiler(phash_distance=0, semantic_deduper=deduper).compile(
        "task", 1, [event_a, event_b]
    )
    assert len(experience.artifacts) == 1
    assert deduper.calls == 1


def test_view_budget_does_not_prune_materialized_experience():
    image = make_image(127)
    events = [
        crop_event(CropRuntime(image, min_dimension=8), "T0", 0, "A0", (0, 0, 400, 400)),
        crop_event(CropRuntime(image, min_dimension=8), "T1", 0, "A0", (300, 300, 700, 700)),
        crop_event(CropRuntime(image, min_dimension=8), "T2", 0, "A0", (600, 600, 1000, 1000)),
    ]
    # Keep these disjoint regions as distinct visual artifacts under the OR
    # dedup rule by making the pHash criterion fail as well.
    events[1]["artifact"]["phash"] = events[0]["artifact"]["phash"] ^ 0b11111
    events[2]["artifact"]["phash"] = events[0]["artifact"]["phash"] ^ 0b1111111111
    experience = CollectiveExperienceCompiler().compile("task", 1, events)
    view = ExperienceViewProjector(artifact_budget=2).project(experience)

    assert len(experience.artifacts) == 3
    assert len(view.artifact_ids) == 2
    assert set(view.artifact_ids) <= {artifact.cluster_id for artifact in experience.artifacts}


def test_decision_conditioned_projection_prioritizes_peer_evidence():
    target_runtime = CropRuntime(make_image(11), min_dimension=8)
    peer_runtime = CropRuntime(make_image(22), min_dimension=8)
    other_runtime = CropRuntime(make_image(33), min_dimension=8)
    target_event = crop_event(target_runtime, "T-0-target", 0, "A0", (0, 0, 300, 300))
    peer_event = crop_event(peer_runtime, "T-1-peer", 0, "A0", (350, 350, 650, 650))
    other_event = crop_event(other_runtime, "T-2-other", 0, "A0", (700, 700, 1000, 1000))
    # Keep the three spatially disjoint regions distinct under the OR dedup rule.
    peer_event["artifact"]["phash"] = target_event["artifact"]["phash"] ^ 0b11111
    other_event["artifact"]["phash"] = target_event["artifact"]["phash"] ^ 0b1111111111
    experience = CollectiveExperienceCompiler().compile("task", 1, [target_event, peer_event, other_event])

    shared = ExperienceViewProjector(artifact_budget=1).project(experience)
    conditioned = ExperienceViewProjector(artifact_budget=1).project(experience, target_event)

    assert shared.selection_policy == "collective_coverage"
    assert conditioned.selection_policy == "decision_conditioned"
    assert conditioned.artifact_ids == ("V2",)
    assert conditioned.artifact_ids != shared.artifact_ids


def test_view_projection_keeps_nested_dependencies():
    runtime = CropRuntime(make_image(), min_dimension=8)
    parent = crop_event(runtime, "T0", 0, "A0", (0, 0, 800, 800))
    child = crop_event(runtime, "T0", 1, "A1", (250, 250, 750, 750))
    experience = CollectiveExperienceCompiler().compile("task", 1, [parent, child])
    view = ExperienceViewProjector(artifact_budget=2).project(experience)

    assert view.artifact_ids == ("V1", "V2")
    assert any(
        relation.relation_type == "derived_from" and relation.source == "V2" and relation.target == "V1"
        for relation in experience.relations
    )


def test_adaptive_view_novelty_uses_iou_but_keeps_nested_crop():
    runtime = CropRuntime(make_image(), min_dimension=8)
    parent = crop_event(runtime, "T0", 0, "A0", (0, 0, 800, 800))
    child = crop_event(runtime, "T0", 1, "A1", (250, 250, 750, 750))
    similar = crop_event(runtime, "T0", 2, "A0", (0, 0, 750, 750))
    experience = CollectiveExperienceCompiler().compile("task", 1, [parent, child, similar])
    view = ExperienceViewProjector(artifact_budget=2, min_visual_novelty=0.5).project(experience)

    # The nested child has low IoU with its parent and remains eligible; this
    # avoids dropping useful zoom detail merely because it is contained.
    assert "V1" in view.artifact_ids and "V2" in view.artifact_ids


def test_compact_serializer_aliases_trajectory_and_decision_ids():
    runtime = CropRuntime(make_image(), min_dimension=8)
    event = crop_event(runtime, "trajectory-with-a-long-id", 0, "A0", (0, 0, 400, 400))
    experience = CollectiveExperienceCompiler().compile("task", 1, [event])
    view = ExperienceViewProjector(artifact_budget=1).project(experience)
    text, _ = ExperienceSerializer(include_outcomes=True, compact_ids=True).render_view(experience, view)

    assert 'trajectory="T1"' in text
    assert "trajectory-with-a-long-id" not in text
    assert 'decision="D:T1:0"' in text


def test_compact_serializer_aggregates_graph_metadata_without_dropping_nodes():
    runtime = CropRuntime(make_image(), min_dimension=8)
    events = [
        crop_event(runtime, "trajectory-with-a-long-id-0", 0, "A0", (0, 0, 400, 400)),
        crop_event(runtime, "trajectory-with-a-long-id-0", 1, "A1", (100, 100, 500, 500)),
    ]
    experience = CollectiveExperienceCompiler().compile("task", 1, events)
    view = ExperienceViewProjector(artifact_budget=2).project(experience)
    compact, compact_images = ExperienceSerializer(
        include_outcomes=True, compact_ids=True, compact_graph=True
    ).render_view(experience, view)
    verbose, verbose_images = ExperienceSerializer(
        include_outcomes=True, compact_ids=True, compact_graph=False
    ).render_view(experience, view)

    assert len(compact) < len(verbose)
    assert compact_images == verbose_images
    assert 'decision="D:T1:0"' in compact
    assert "artifact=V1" in compact
    assert "relations=produces:" in compact


def test_structural_relations_do_not_materialize_quadratic_disjoint_edges():
    runtime = CropRuntime(make_image(), min_dimension=8)
    events = [
        crop_event(runtime, "T0", 0, "A0", (0, 0, 800, 800)),
        crop_event(runtime, "T0", 1, "A0", (100, 100, 200, 200)),
        crop_event(runtime, "T0", 2, "A0", (850, 850, 1000, 1000)),
    ]
    experience = CollectiveExperienceCompiler().compile("task", 1, events)
    relation_types = {relation.relation_type for relation in experience.relations}

    assert {"contains", "next", "produces", "derived_from", "belongs_to"} <= relation_types
    assert "disjoint" not in relation_types
    json.loads(experience.dumps())


def test_outcomes_are_categorical_associations_not_causal_labels():
    success_runtime = CropRuntime(make_image(), min_dimension=8)
    success_events = annotate(
        [
            crop_event(success_runtime, "T-success", 0, "A0", (0, 0, 400, 400), label="small sign"),
            final_event("T-success", 1, True, success_runtime.artifact_order),
        ],
        episode_return=1.0,
    )
    failure_runtime = CropRuntime(make_image(), min_dimension=8)
    failure_crop = crop_event(failure_runtime, "T-failure", 0, "A0", (600, 600, 1000, 1000), label="corner")
    failure_crop["action"]["rationale"] = r"Maybe here. \boxed{B}"
    failure_events = annotate(
        [failure_crop, final_event("T-failure", 1, False, failure_runtime.artifact_order)],
        episode_return=0.2,
    )
    experience = CollectiveExperienceCompiler().compile("task", 7, success_events + failure_events)

    assert {trajectory.outcome.status for trajectory in experience.trajectories} == {
        "success",
        "failure",
    }
    assert {association.association_type for association in experience.associations} == {
        "observed_in_success",
        "observed_in_failure",
    }
    assert experience.diagnostics[0].diagnostic_type == "coverage_gap_candidate"
    payload = experience.dumps()
    assert "supports_success" not in payload
    assert "rescue_candidate" not in payload
    assert "\\boxed" not in payload


def test_shared_region_is_marked_mixed_and_serialization_is_answer_free():
    image = make_image(127)
    first = crop_event(CropRuntime(image, min_dimension=8), "T0", 0, "A0", (0, 0, 500, 500))
    second = crop_event(CropRuntime(image, min_dimension=8), "T1", 0, "A0", (0, 0, 500, 500))
    first["trajectory_outcome"] = {
        "status": "success",
        "answer_correct": True,
        "trajectory_valid": True,
        "crop_count": 1,
        "failure_type": None,
        "episode_return": 1.0,
    }
    second["trajectory_outcome"] = {
        "status": "failure",
        "answer_correct": False,
        "trajectory_valid": True,
        "crop_count": 1,
        "failure_type": "wrong_after_crop",
        "episode_return": 0.2,
    }
    experience = CollectiveExperienceCompiler().compile("task", 1, [first, second])
    view = ExperienceViewProjector(artifact_budget=1).project(experience)
    text, images = ExperienceSerializer(include_outcomes=True).render_view(experience, view)

    assert experience.associations[0].association_type == "mixed_outcome_observation"
    assert len(images) == text.count("<image>") == 1
    assert "outcome=success" in text and "outcome=failure" in text
    assert "ground_truth" not in text
    assert "episode_return" not in text


def test_decision_alignment_separates_visible_peer_and_future_self_artifacts():
    runtime_t0 = CropRuntime(make_image(), min_dimension=8)
    t0_first = crop_event(runtime_t0, "T0", 0, "A0", (0, 0, 300, 300))
    t0_second = crop_event(runtime_t0, "T0", 1, "A0", (350, 350, 650, 650))
    runtime_t1 = CropRuntime(make_image(), min_dimension=8)
    t1 = crop_event(runtime_t1, "T1", 0, "A0", (700, 700, 1000, 1000))
    experience = CollectiveExperienceCompiler().compile("task", 1, [t0_first, t0_second, t1])
    view = ExperienceViewProjector(artifact_budget=3).project(experience)
    aligner = DecisionAligner()

    first_alignment = aligner.align(experience, view, t0_first)
    assert first_alignment.student_visible_artifact_ids == ()
    assert first_alignment.peer_only_artifact_ids == ("V3",)
    assert first_alignment.self_future_artifact_ids == ("V2",)

    second_alignment = aligner.align(experience, view, t0_second)
    assert second_alignment.student_visible_artifact_ids == ("V1",)
    rendered = ExperienceSerializer.render_alignment(second_alignment)
    assert 'student_visible=["V1"]' in rendered


@pytest.mark.parametrize("compact_graph", [True, False])
@pytest.mark.parametrize("include_outcomes", [True, False])
@pytest.mark.parametrize("condition_on_target", [True, False])
def test_teacher_view_excludes_generated_final_answers(compact_graph, include_outcomes, condition_on_target):
    target = final_event("T-target", 0, True, ["A0"], answer="TARGET_ANSWER_SENTINEL")
    successful_peer = final_event("T-peer-ok", 0, True, ["A0"], answer="SUCCESS_ANSWER_SENTINEL")
    failed_peer = final_event("T-peer-fail", 0, False, ["A0"], answer="FAILURE_ANSWER_SENTINEL")
    events = []
    for event, reward in (
        (target, 1.0),
        (successful_peer, 1.0),
        (failed_peer, 0.0),
    ):
        events.extend(annotate([event], episode_return=reward))

    experience = CollectiveExperienceCompiler().compile("task", 1, events)
    view = ExperienceViewProjector(artifact_budget=1).project(experience, target if condition_on_target else None)
    serializer = ExperienceSerializer(include_outcomes=include_outcomes, compact_graph=compact_graph)
    text, images = serializer.render_view(experience, view)

    assert images == []
    if condition_on_target:
        assert view.target_trajectory_id == "T-target"
    assert "peer_final_answer" not in text
    assert all(decision.final_answer is None for decision in experience.decisions)
    for event in (target, successful_peer, failed_peer):
        # Verification inputs survive, but never enter compiled teacher data.
        assert event["final_answer"].endswith("_ANSWER_SENTINEL")
        assert event["final_answer"] not in text
        assert event["final_answer"] not in experience.dumps()
    assert ("outcome=success" in text) == include_outcomes
    assert ("outcome=failure" in text) == include_outcomes

    # Older records may still carry answers; the serializer must ignore them.
    legacy = replace(
        experience,
        decisions=tuple(replace(decision, final_answer="LEGACY_ANSWER_SENTINEL") for decision in experience.decisions),
    )
    legacy_text, _ = serializer.render_view(legacy, view)
    assert legacy_text == text

    student_messages = [{"role": "user", "content": "Inspect the image."}]
    teacher_messages = append_teacher_context(student_messages, legacy_text, images)
    teacher_text = json.dumps(teacher_messages)
    assert "ANSWER_SENTINEL" not in teacher_text
    assert "[ANSWER_PRIVILEGE]" not in teacher_text
    assert student_messages == [{"role": "user", "content": "Inspect the image."}]
    instruction = COLLECTIVE_EXPERIENCE_TEACHER_SYSTEM_INSTRUCTION
    assert "Generated final answers are excluded" in instruction
    assert "reliable peer-answer privilege" not in instruction


def test_excluding_peer_answers_preserves_visual_context_and_feedback():
    runtime = CropRuntime(make_image(), min_dimension=8)
    crop = crop_event(runtime, "T-peer", 0, "A0", (0, 0, 500, 500))
    final = final_event("T-peer", 1, True, ["A0", "A1"], answer="PEER_ANSWER_SENTINEL")
    events = annotate([crop, final], episode_return=1.0)
    experience = CollectiveExperienceCompiler().compile("task", 1, events)
    view = ExperienceViewProjector(artifact_budget=1).project(experience)
    text, images = ExperienceSerializer().render_view(experience, view)

    assert "PEER_ANSWER_SENTINEL" not in text
    assert "outcome=success" in text
    assert "inspect the stated region" in text
    assert "relations=produces:" in text
    assert text.count("<image>") == len(images) == 1
    messages = append_teacher_context([{"role": "user", "content": "Inspect the image."}], text, images)
    assert sum(item.get("type") == "image" for item in messages[-1]["content"]) == 1


def test_invalid_regions_are_rejected_and_small_regions_are_upscaled():
    runtime = CropRuntime(make_image(), min_dimension=32)
    with pytest.raises(ValueError, match="zero area"):
        runtime.crop(CropAction("A0", (100, 100, 100, 500)))
    small = runtime.crop(CropAction("A0", (0, 0, 20, 10)))
    assert small.root_bbox == (0, 0, 20, 10)
    assert small.image.size == (64, 32)
