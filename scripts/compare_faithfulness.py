"""Before/after faithfulness comparison for the explanation prompt.

Generates an answer for each test question with the OLD and the NEW
explanation prompt (local llama3.1:8b), scores each with RAGAS faithfulness
(local judge), and prints the delta. Used to verify that tightening the
prompt actually raises faithfulness before committing the change.
"""

import os
import sys
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
from ragas.metrics import Faithfulness
from ragas.run_config import RunConfig

from api.retrieval import search_chunks
from ingestion.embed import embed_prompts, embed_query

load_dotenv()

LOCAL_MODEL = "llama3.1:8b"
OLLAMA_OPENAI_BASE = "http://localhost:11434/v1"

OLD_PROMPT = """You are a helpful tutor. Answer using ONLY the context below.
Context: {context}
Student question: {question}
Provide: a simple explanation, one worked example, and one common mistake students make."""

NEW_PROMPT = """You are a helpful tutor for a school student.

Answer the student's question using ONLY the information in the context below. Follow these rules strictly:
- Use only facts that are stated in or directly supported by the context.
- Do NOT add any facts, figures, examples, or claims that are not in the context.
- If the context does not contain enough information to answer, say so plainly instead of guessing.
- Write a clear, simple explanation in your own words, appropriate for the student's level.
- You may include an example or point out a common misconception ONLY if it is supported by the context; otherwise leave it out.

Context:
{context}

Student question: {question}"""

TEST_CASES = [
    ("Environmental Studies", "5", 16, "What are the different sources of water?"),
    ("Environmental Studies", "5", 19, "What are the main constituents of food?"),
    ("Science", "8", 1, "What are the five kingdoms in the classification of living organisms?"),
    ("Science", "8", 5, "What are the sub-atomic particles present in an atom?"),
]


class OllamaGemmaEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return embed_prompts(list(texts))

    def embed_query(self, text):
        return embed_prompts([text])[0]


def call_local(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_OPENAI_BASE}/chat/completions",
        json={"model": LOCAL_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def score_faithfulness(samples, judge):
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness()],
        llm=judge,
        run_config=RunConfig(max_workers=1, timeout=300),
        show_progress=False,
    )
    return result.to_pandas()["faithfulness"].tolist()


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    contexts, questions = [], []
    old_samples, new_samples = [], []
    try:
        for subject, class_, chapter, question in TEST_CASES:
            emb = embed_query(question)
            chunks = search_chunks(conn, emb, subject, class_, chapter)
            context = "\n\n".join(c["chunk_text"] for c in chunks)
            contexts.append([c["chunk_text"] for c in chunks])
            questions.append(question)
            print(f"generating: {subject} ch{chapter}: {question}", flush=True)
            old_ans = call_local(OLD_PROMPT.format(context=context, question=question))
            new_ans = call_local(NEW_PROMPT.format(context=context, question=question))
            old_samples.append(SingleTurnSample(user_input=question, retrieved_contexts=contexts[-1], response=old_ans))
            new_samples.append(SingleTurnSample(user_input=question, retrieved_contexts=contexts[-1], response=new_ans))
    finally:
        conn.close()

    judge = LangchainLLMWrapper(
        ChatOpenAI(model=LOCAL_MODEL, base_url=OLLAMA_OPENAI_BASE, api_key="ollama", temperature=0, timeout=300)
    )
    print("\nscoring OLD prompt...", flush=True)
    old_scores = score_faithfulness(old_samples, judge)
    print("scoring NEW prompt...", flush=True)
    new_scores = score_faithfulness(new_samples, judge)

    print("\n=== Faithfulness: OLD -> NEW ===")
    for q, o, n in zip(questions, old_scores, new_scores):
        print(f"  {o:.3f} -> {n:.3f}   {q}")
    import statistics
    print(f"\n  MEAN {statistics.mean(old_scores):.3f} -> {statistics.mean(new_scores):.3f}")


if __name__ == "__main__":
    main()
