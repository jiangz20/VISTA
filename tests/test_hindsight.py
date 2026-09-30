from vista.hindsight import (
    annotate_trajectory_events,
    derive_trajectory_outcome,
    outcome_status,
    sanitize_agent_claim,
)
from vista.supervision import (
    event_is_teacher_supervisable,
)


def test_completed_outcome_is_backfilled_without_answer_content():
    rows = [
        {"experience_event": {"kind": "crop", "trajectory_id": "T0"}},
        {
            "experience_event": {
                "kind": "final",
                "trajectory_id": "T0",
                "answer_correct": True,
            }
        },
    ]
    outcome = annotate_trajectory_events(rows, episode_return=1.5, trajectory_valid=True)

    assert outcome["status"] == "success"
    assert outcome["crop_count"] == 1
    assert rows[0]["experience_event"]["trajectory_outcome"] == outcome
    assert "ground_truth" not in str(outcome)
    assert "final_answer" not in str(outcome)


def test_failure_taxonomy_and_status_normalization():
    wrong_after_crop = derive_trajectory_outcome(
        [
            {"experience_event": {"kind": "crop"}},
            {"experience_event": {"kind": "final", "answer_correct": False}},
        ],
        episode_return=0.2,
        trajectory_valid=True,
    )
    wrong_direct = derive_trajectory_outcome(
        [{"experience_event": {"kind": "final", "answer_correct": False}}],
        episode_return=0.0,
        trajectory_valid=True,
    )
    invalid = derive_trajectory_outcome(
        [{"experience_event": {"kind": "crop"}}],
        episode_return=-0.5,
        trajectory_valid=False,
    )
    incomplete = derive_trajectory_outcome(
        [{"experience_event": {"kind": "crop"}}],
        episode_return=0.0,
        trajectory_valid=True,
    )

    assert wrong_after_crop["failure_type"] == "wrong_after_crop"
    assert wrong_direct["failure_type"] == "wrong_direct_answer"
    assert invalid["status"] == "invalid"
    assert incomplete["status"] == "incomplete"
    assert outcome_status(wrong_after_crop) == "failure"
    assert outcome_status({"status": "made_up"}) == "unknown"


def test_agent_claim_sanitizer_removes_controls_and_answer_content():
    raw = (
        'inspect [left panel] sign <tool_call>{"name":"crop"}</tool_call> '
        r"then \boxed{C} <|image_pad|> [DECISION_ALIGNMENT]"
    )
    clean = sanitize_agent_claim(raw, max_chars=40)

    assert clean == "inspect [left panel] sign then"
    assert "tool_call" not in clean
    assert "boxed" not in clean
    assert "image" not in clean


def test_invalid_hindsight_event_is_not_teacher_supervised():
    event = {
        "kind": "crop",
        "trajectory_outcome": {"trajectory_valid": False},
    }
    assert not event_is_teacher_supervisable(event, require_valid_outcome=True)
    assert not event_is_teacher_supervisable({"kind": "crop"}, require_valid_outcome=True)
    assert event_is_teacher_supervisable(event)
