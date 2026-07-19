from api.quiz_agent import (
    build_quiz_prompt,
    generate_quiz,
    quiz_schema,
    score_answers,
)
from api.schemas import DIFFICULTIES, QUESTION_TYPES


def test_question_types_and_difficulties_constants():
    assert QUESTION_TYPES == {"mcq", "fill_in_the_blank", "assertion_reason"}
    assert DIFFICULTIES == {"easy", "medium", "hard", "mixed"}


def test_build_quiz_prompt_includes_count_difficulty_types_and_context():
    prompt = build_quiz_prompt(
        context="Photosynthesis happens in leaves.",
        subject="Science",
        class_="8",
        question_types=["mcq"],
        difficulty="easy",
        count=3,
    )
    assert "3" in prompt
    assert "easy" in prompt
    assert "mcq" in prompt
    assert "Photosynthesis happens in leaves." in prompt
    assert "Science" in prompt and "8" in prompt


def test_build_quiz_prompt_explains_fill_in_the_blank_empty_options():
    prompt = build_quiz_prompt(
        context="Water boils at 100 degrees.",
        subject="Science",
        class_="8",
        question_types=["fill_in_the_blank"],
        difficulty="mixed",
        count=2,
    )
    # fill-in-the-blank items must carry no options
    assert "fill_in_the_blank" in prompt
    assert "options" in prompt.lower()


def test_quiz_schema_is_strict_and_covers_all_types():
    schema = quiz_schema()
    item = schema["properties"]["quiz"]["items"]
    props = item["properties"]
    assert set(props) == {
        "type",
        "question",
        "options",
        "correct_answer",
        "difficulty",
        "explanation",
    }
    assert set(item["required"]) == set(props)
    assert item["additionalProperties"] is False
    assert set(props["type"]["enum"]) == QUESTION_TYPES
    assert set(props["difficulty"]["enum"]) == {"easy", "medium", "hard"}


def test_score_answers_exact_match():
    quiz = [
        {"question": "Q1", "correct_answer": "A"},
        {"question": "Q2", "correct_answer": "B"},
    ]
    correct, total = score_answers(quiz, {"Q1": "A", "Q2": "C"})
    assert (correct, total) == (1, 2)


def test_score_answers_normalizes_case_and_whitespace():
    # fill-in-the-blank answers should match despite case/spacing differences
    quiz = [{"question": "Water boils at ____ C.", "correct_answer": "100"}]
    correct, total = score_answers(quiz, {"Water boils at ____ C.": " 100 "})
    assert (correct, total) == (1, 1)

    quiz2 = [{"question": "Capital?", "correct_answer": "Mumbai"}]
    correct2, _ = score_answers(quiz2, {"Capital?": "mumbai"})
    assert correct2 == 1


def test_score_answers_missing_answer_counts_as_wrong():
    quiz = [
        {"question": "Q1", "correct_answer": "A"},
        {"question": "Q2", "correct_answer": "B"},
    ]
    correct, total = score_answers(quiz, {"Q1": "A"})
    assert (correct, total) == (1, 2)


def test_generate_quiz_returns_requested_count_and_type():
    result = generate_quiz(
        context=(
            "An atom is made of three sub-atomic particles: protons (positive "
            "charge), neutrons (no charge), and electrons (negative charge). "
            "Protons and neutrons are in the nucleus; electrons orbit it."
        ),
        subject="Science",
        class_="8",
        question_types=["mcq"],
        difficulty="easy",
        count=2,
    )
    quiz = result["quiz"]
    assert len(quiz) == 2
    for q in quiz:
        assert q["type"] == "mcq"
        assert len(q["options"]) == 4
        assert q["correct_answer"]
        assert q["difficulty"] in {"easy", "medium", "hard"}
