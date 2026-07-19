from api.agents import INTENTS, detect_intent


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
