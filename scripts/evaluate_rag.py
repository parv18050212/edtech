"""Evaluate the real RAG pipeline (retrieval + generation) with RAGAS.

Uses reference-free metrics so no hand-labelled answers are needed:
  - faithfulness: are the answer's claims grounded in the retrieved chunks
    (anti-hallucination)?
  - answer_relevancy: does the answer actually address the question?
  - llm_context_precision_without_reference: are the retrieved chunks
    relevant to the question (retrieval quality)?

Generation + judge LLM: local Ollama llama3.1:8b (via Ollama's
OpenAI-compatible endpoint) -- fully local, no API rate limits.
Embeddings: the same local Ollama embeddinggemma used for ingestion.

Note: this measures the RAG pipeline with the LOCAL 8B model as the
generator, not the Groq 70B production chat path. It's a local, free,
rate-limit-free way to validate the retrieval + generation mechanics.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg2
import requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from api.retrieval import search_chunks
from api.routes.chat import EXPLANATION_PROMPT_TEMPLATE
from ingestion.embed import embed_prompts, embed_query

load_dotenv()

# Local Ollama model used for BOTH generation and RAGAS judging.
LOCAL_MODEL = "llama3.1:8b"
OLLAMA_OPENAI_BASE = "http://localhost:11434/v1"


def call_local(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_OPENAI_BASE}/chat/completions",
        json={
            "model": LOCAL_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# (subject, class, chapter_number, question) spread across both books and
# a variety of chapters.
TEST_CASES = [
    ("Environmental Studies", "5", 1, "What is a star and how is it different from a planet?"),
    ("Environmental Studies", "5", 1, "What is the solar system made up of?"),
    ("Environmental Studies", "5", 16, "What are the different sources of water?"),
    ("Environmental Studies", "5", 19, "What are the main constituents of food?"),
    ("Science", "8", 1, "What are the five kingdoms in the classification of living organisms?"),
    ("Science", "8", 3, "What is pressure and how is it related to force and area?"),
    ("Science", "8", 5, "What are the sub-atomic particles present in an atom?"),
    ("Science", "8", 15, "How is sound produced and how does it travel?"),
]


class OllamaGemmaEmbeddings(Embeddings):
    """Adapter exposing our ingestion embedding function to LangChain/RAGAS."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_prompts(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return embed_prompts([text])[0]


def build_sample(conn, subject: str, class_: str, chapter_number: int, question: str) -> SingleTurnSample:
    query_embedding = embed_query(question)
    chunks = search_chunks(conn, query_embedding, subject, class_, chapter_number)
    contexts = [c["chunk_text"] for c in chunks]
    prompt = EXPLANATION_PROMPT_TEMPLATE.format(
        context="\n\n".join(contexts), question=question
    )
    answer = call_local(prompt)
    return SingleTurnSample(
        user_input=question,
        retrieved_contexts=contexts,
        response=answer,
    )


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        samples = []
        for i, (subject, class_, chapter, question) in enumerate(TEST_CASES, 1):
            print(f"[{i}/{len(TEST_CASES)}] {subject} ch{chapter}: {question}", flush=True)
            # Resilience against transient network/DNS blips over a long run.
            for attempt in range(4):
                try:
                    samples.append(build_sample(conn, subject, class_, chapter, question))
                    break
                except Exception as exc:  # noqa: BLE001 - retry any transient failure
                    if attempt == 3:
                        raise
                    print(f"    transient error ({type(exc).__name__}), retrying...", flush=True)
                    time.sleep(5 * (attempt + 1))
    finally:
        conn.close()

    dataset = EvaluationDataset(samples=samples)

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=LOCAL_MODEL,
            base_url=OLLAMA_OPENAI_BASE,
            api_key="ollama",  # ignored by Ollama, but ChatOpenAI requires a value
            temperature=0,
            timeout=300,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaGemmaEmbeddings())

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ]

    print(f"\nScoring with RAGAS (local {LOCAL_MODEL} judge + Ollama embeddings)...\n")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=2, timeout=300),
        show_progress=True,
    )

    print("\n=== Aggregate scores ===")
    print(result)

    df = result.to_pandas()
    print("\n=== Per-question scores ===")
    metric_cols = [c for c in df.columns if c in
                   ("faithfulness", "answer_relevancy", "llm_context_precision_without_reference")]
    for idx, row in df.iterrows():
        print(f"\nQ{idx + 1}: {row['user_input']}")
        for col in metric_cols:
            print(f"    {col}: {row[col]:.3f}")


if __name__ == "__main__":
    main()
