from api.agents import (
    INTENTS,
    detect_intent,
    generate_follow_ups,
    generate_practice,
    generate_step_by_step,
)

_ATOM_CONTEXT = (
    "Pressure is the force acting per unit area. Pressure = Force / Area. "
    "Force is measured in newtons (N) and area in square metres. "
    "An atom contains protons, neutrons and electrons."
)


def test_intents_constant():
    assert INTENTS == {"explanation", "step_by_step", "quiz", "practice"}


def test_detect_intent_explanation():
    assert detect_intent("Explain how photosynthesis works") == "explanation"


def test_detect_intent_quiz():
    assert detect_intent("Quiz me on the water cycle") == "quiz"


def test_detect_intent_step_by_step():
    assert detect_intent(
        "Solve: a force of 10 N acts on an area of 2 square metres, find the pressure"
    ) == "step_by_step"


def test_detect_intent_practice():
    assert detect_intent("Give me some practice questions on atoms") == "practice"


def test_generate_step_by_step_returns_nonempty_text():
    out = generate_step_by_step(
        "A force of 10 N acts on an area of 2 square metres. Find the pressure.",
        _ATOM_CONTEXT,
    )
    assert isinstance(out, str) and out.strip()
    assert "5" in out  # 10 / 2 = 5


def test_generate_practice_returns_list_of_questions():
    out = generate_practice("Give me practice questions on atoms", _ATOM_CONTEXT)
    assert isinstance(out, list)
    assert 1 <= len(out) <= 10
    assert all(isinstance(q, str) and q.strip() for q in out)


def test_generate_follow_ups_returns_questions():
    out = generate_follow_ups("What is a star?", _ATOM_CONTEXT)
    assert isinstance(out, list)
    assert 1 <= len(out) <= 5
    assert all(isinstance(q, str) and q.strip() for q in out)
