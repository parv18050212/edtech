from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from api.agents import (
    GENERATION_MODEL,
    detect_intent,
    generate_follow_ups,
    generate_practice,
    generate_step_by_step,
)
from api.groq_client import call_groq
from api.quiz_agent import generate_quiz
from api.retrieval import search_chunks
from api.routes.chat import EXPLANATION_PROMPT_TEMPLATE
from ingestion.embed import embed_query

RETRIEVAL_TOP_K = 5


class OrchestratorState(TypedDict, total=False):
    question: str
    subject: str
    class_: str
    chapter_number: int
    intent: str
    context: str
    source_chunk_ids: list
    explanation: Optional[str]
    step_by_step: Optional[str]
    quiz: Optional[list]
    practice_questions: Optional[list]
    follow_up_questions: list


def build_orchestrator(conn):
    def detect_intent_node(state: OrchestratorState) -> dict:
        return {"intent": detect_intent(state["question"])}

    def retrieve_node(state: OrchestratorState) -> dict:
        embedding = embed_query(state["question"])
        chunks = search_chunks(
            conn,
            embedding,
            state["subject"],
            state["class_"],
            state["chapter_number"],
            top_k=RETRIEVAL_TOP_K,
        )
        return {
            "context": "\n\n".join(c["chunk_text"] for c in chunks),
            "source_chunk_ids": [c["chunk_id"] for c in chunks],
        }

    def explanation_node(state: OrchestratorState) -> dict:
        prompt = EXPLANATION_PROMPT_TEMPLATE.format(
            context=state["context"], question=state["question"]
        )
        return {"explanation": call_groq(prompt, model=GENERATION_MODEL)}

    def step_by_step_node(state: OrchestratorState) -> dict:
        return {
            "step_by_step": generate_step_by_step(state["question"], state["context"])
        }

    def quiz_node(state: OrchestratorState) -> dict:
        result = generate_quiz(
            context=state["context"],
            subject=state["subject"],
            class_=state["class_"],
            question_types=["mcq"],
            difficulty="mixed",
            count=5,
        )
        return {"quiz": result["quiz"]}

    def practice_node(state: OrchestratorState) -> dict:
        return {
            "practice_questions": generate_practice(
                state["question"], state["context"]
            )
        }

    def follow_up_node(state: OrchestratorState) -> dict:
        return {
            "follow_up_questions": generate_follow_ups(
                state["question"], state["context"]
            )
        }

    # Node names are prefixed so they never collide with state keys of the
    # same name (LangGraph forbids that, e.g. a node "quiz" vs state["quiz"]).
    graph = StateGraph(OrchestratorState)
    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("gen_explanation", explanation_node)
    graph.add_node("gen_step_by_step", step_by_step_node)
    graph.add_node("gen_quiz", quiz_node)
    graph.add_node("gen_practice", practice_node)
    graph.add_node("gen_follow_up", follow_up_node)

    graph.add_edge(START, "detect_intent")
    graph.add_edge("detect_intent", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        lambda state: state["intent"],
        {
            "explanation": "gen_explanation",
            "step_by_step": "gen_step_by_step",
            "quiz": "gen_quiz",
            "practice": "gen_practice",
        },
    )
    for node in ("gen_explanation", "gen_step_by_step", "gen_quiz", "gen_practice"):
        graph.add_edge(node, "gen_follow_up")
    graph.add_edge("gen_follow_up", END)

    return graph.compile()


def run_ask(
    conn, question: str, subject: str, class_: str, chapter_number: int
) -> dict:
    graph = build_orchestrator(conn)
    initial: OrchestratorState = {
        "question": question,
        "subject": subject,
        "class_": class_,
        "chapter_number": chapter_number,
        "explanation": None,
        "step_by_step": None,
        "quiz": None,
        "practice_questions": None,
    }
    return graph.invoke(initial)
