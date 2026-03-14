from __future__ import annotations

import json
import math
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

from ingestion.loader import Document
from generation.chain import RAGChain
from config import FAITHFULNESS_MIN, RELEVANCY_MIN

_DEFAULT_DATASET = Path(__file__).parent / "golden_dataset.json"


class Evaluator:

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        faithfulness_min: float = FAITHFULNESS_MIN,
        relevancy_min: float = RELEVANCY_MIN,
    ) -> None:
        path = Path(dataset_path) if dataset_path else _DEFAULT_DATASET
        with open(path) as f:
            self._golden: list[dict] = json.load(f)

        self._faithfulness_min = faithfulness_min
        self._relevancy_min = relevancy_min

    def run(self, rag_chain: RAGChain | None = None) -> dict:
        if rag_chain is None:
            rag_chain = RAGChain()

        questions: list[str] = []
        ground_truths: list[str] = []
        answers: list[str] = []
        contexts_list: list[list[str]] = []

        for entry in self._golden:
            question = entry["question"]
            docs = [
                Document(
                    page_content=ctx,
                    metadata={
                        "source": "golden_dataset",
                        "file_name": "golden_dataset.json",
                        "page": i + 1,
                    },
                )
                for i, ctx in enumerate(entry["contexts"])
            ]

            answer = rag_chain.generate(question, docs)

            questions.append(question)
            ground_truths.append(entry["ground_truth"])
            answers.append(answer)
            contexts_list.append(entry["contexts"])

        ds = Dataset.from_dict(
            {
                "question": questions,
                "ground_truth": ground_truths,
                "answer": answers,
                "contexts": contexts_list,
            }
        )

        result = evaluate(
            dataset=ds,
            metrics=[Faithfulness(), AnswerRelevancy()],
        )

        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            per_question.append(
                {
                    "question": row["question"],
                    "faithfulness": float(row.get("faithfulness", 0.0)),
                    "answer_relevancy": float(
                        row.get("answer_relevancy", 0.0)
                    ),
                }
            )

        return {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
            "per_question": per_question,
        }

    def assert_thresholds(self, results: dict | None = None) -> None:
        if results is None:
            results = self.run()

        faith = results["faithfulness"]
        relevancy = results["answer_relevancy"]

        failures = []

        if math.isnan(faith) or faith < self._faithfulness_min:
            failures.append(
                f"  faithfulness: {faith:.4f} < {self._faithfulness_min} (FAIL)"
            )

        if math.isnan(relevancy) or relevancy < self._relevancy_min:
            failures.append(
                f"  answer_relevancy: {relevancy:.4f} < {self._relevancy_min} (FAIL)"
            )

        if failures:
            msg = "Evaluation threshold failed:\n" + "\n".join(failures)
            raise AssertionError(msg)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    evaluator = Evaluator()
    results = evaluator.run()
    print(json.dumps(results, indent=2))
    evaluator.assert_thresholds(results)
    print("All thresholds passed.")
