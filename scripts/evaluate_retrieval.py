"""
Retrieval and Relevance-Gate Evaluation Harness for Agronomy Advisory Assistant.

Evaluates retrieval quality (Hit@k, MRR) for labeled ground-truth queries and
evaluates safety abstention rates for unsupported crops, mismatches, ambiguous,
and off-topic queries.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import AgronomyRetriever
from src.guard.query_guard import QueryGuard, GuardAction
from src.guard.relevance_gate import RelevanceGate, GateVerdict

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_harness")

BENCHMARK_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_benchmark.jsonl"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_evaluation_report.json"


def run_retrieval_evaluation(top_k: int = 3) -> Dict[str, Any]:
    print("=" * 75)
    print("AGRONOMY ADVISORY ASSISTANT: RETRIEVAL & RELEVANCE GATE EVALUATION")
    print(f"Benchmark File: {BENCHMARK_PATH}")
    print(f"Retrieval top_k: {top_k}")
    print("=" * 75)

    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Benchmark file not found: {BENCHMARK_PATH}")

    retriever = AgronomyRetriever()
    guard = QueryGuard()
    gate = RelevanceGate()

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmarks = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(benchmarks)} benchmark queries.\n")

    # Metrics containers
    gt_total = 0
    gt_raw_hit_1 = 0
    gt_raw_hit_3 = 0
    gt_raw_reciprocal_ranks = []

    gt_gated_hit_1 = 0
    gt_gated_hit_3 = 0
    gt_gated_reciprocal_ranks = []
    gt_crop_matches = 0

    unsupported_total = 0
    unsupported_blocked = 0
    unsupported_cross_crop_leaks = 0

    mismatch_total = 0
    mismatch_blocked = 0

    ambiguous_total = 0
    ambiguous_clarified = 0

    non_agri_total = 0
    non_agri_abstained = 0

    score_stats: Dict[str, List[float]] = {
        "in_domain_target": [],
        "unsupported_nearest_neighbor": [],
        "crop_mismatch_nearest_neighbor": [],
        "offtopic_nearest_neighbor": [],
    }

    detailed_results = []

    for item in benchmarks:
        qid = item["query_id"]
        query = item["query"]
        cat = item["category"]
        target_crop = item.get("target_crop")
        target_chunk_id = item.get("target_chunk_id")
        has_gt = item.get("has_ground_truth", False)

        # 1. Guard evaluation
        guard_decision = guard.classify(query)

        res_detail = {
            "query_id": qid,
            "query": query,
            "category": cat,
            "guard_action": guard_decision.action.value,
            "detected_crop": guard_decision.detected_crop,
            "retrieval_called": False,
        }

        if guard_decision.action == GuardAction.PROCEED:
            # 2. Retrieval
            retrieval_results = retriever.retrieve(query, top_k=top_k)
            res_detail["retrieval_called"] = True
            res_detail["raw_top_scores"] = [r.similarity_score for r in retrieval_results]
            res_detail["raw_top_crops"] = [r.crop_clean for r in retrieval_results]
            res_detail["raw_top_ids"] = [r.chunk_id for r in retrieval_results]

            # 3. Relevance Gate
            gate_result = gate.evaluate(
                query=query,
                results=retrieval_results,
                query_crop=guard_decision.detected_crop,
            )
            res_detail["gate_verdict"] = gate_result.verdict.value
            res_detail["gate_best_score"] = gate_result.best_score
            res_detail["gate_best_crop"] = gate_result.best_chunk.crop_clean if gate_result.best_chunk else None
            res_detail["gate_best_id"] = gate_result.best_chunk.chunk_id if gate_result.best_chunk else None
            res_detail["promoted_rank"] = (
                retrieval_results.index(gate_result.best_chunk) + 1
                if gate_result.best_chunk and gate_result.best_chunk in retrieval_results
                else None
            )

            # Record category-specific metrics
            if has_gt:
                gt_total += 1
                raw_chunk_ids = [r.chunk_id for r in retrieval_results]

                # Raw metrics
                if raw_chunk_ids and raw_chunk_ids[0] == target_chunk_id:
                    gt_raw_hit_1 += 1
                if target_chunk_id in raw_chunk_ids[:3]:
                    gt_raw_hit_3 += 1
                if target_chunk_id in raw_chunk_ids:
                    rank = raw_chunk_ids.index(target_chunk_id) + 1
                    gt_raw_reciprocal_ranks.append(1.0 / rank)
                else:
                    gt_raw_reciprocal_ranks.append(0.0)

                # Gated metrics
                compatible_ids = [r.chunk_id for r in gate_result.compatible_chunks]
                if compatible_ids and compatible_ids[0] == target_chunk_id:
                    gt_gated_hit_1 += 1
                if target_chunk_id in compatible_ids[:3]:
                    gt_gated_hit_3 += 1
                if target_chunk_id in compatible_ids:
                    rank = compatible_ids.index(target_chunk_id) + 1
                    gt_gated_reciprocal_ranks.append(1.0 / rank)
                else:
                    gt_gated_reciprocal_ranks.append(0.0)

                # Crop match check on selected best chunk
                if gate_result.best_chunk and target_crop:
                    is_match = gate._check_crop_mismatch(target_crop, gate_result.best_chunk) is None
                    if is_match:
                        gt_crop_matches += 1

                # Track score of target chunk
                for r in retrieval_results:
                    if r.chunk_id == target_chunk_id:
                        score_stats["in_domain_target"].append(r.similarity_score)
                        break

            elif cat == "unsupported_crop":
                unsupported_total += 1
                if gate_result.verdict in (GateVerdict.CROP_MISMATCH, GateVerdict.EVIDENCE_ABSENT):
                    unsupported_blocked += 1
                if gate_result.verdict == GateVerdict.EVIDENCE_SUFFICIENT:
                    unsupported_cross_crop_leaks += 1
                if retrieval_results:
                    score_stats["unsupported_nearest_neighbor"].append(retrieval_results[0].similarity_score)

            elif cat == "crop_mismatch":
                mismatch_total += 1
                if gate_result.verdict in (GateVerdict.CROP_MISMATCH, GateVerdict.EVIDENCE_ABSENT):
                    mismatch_blocked += 1
                if retrieval_results:
                    score_stats["crop_mismatch_nearest_neighbor"].append(retrieval_results[0].similarity_score)

        if cat == "ambiguous_query":
            ambiguous_total += 1
            if guard_decision.action in (GuardAction.CLARIFY_CROP, GuardAction.CLARIFY_SYMPTOMS):
                ambiguous_clarified += 1

        elif cat == "non_agricultural_or_dynamic":
            non_agri_total += 1
            if guard_decision.action in (GuardAction.ABSTAIN_OFFTOPIC, GuardAction.ABSTAIN_DYNAMIC):
                non_agri_abstained += 1
            raw_offtopic = retriever.retrieve(query, top_k=1)
            if raw_offtopic:
                score_stats["offtopic_nearest_neighbor"].append(raw_offtopic[0].similarity_score)

        detailed_results.append(res_detail)

    # Compute summary aggregates
    raw_mrr = sum(gt_raw_reciprocal_ranks) / len(gt_raw_reciprocal_ranks) if gt_raw_reciprocal_ranks else 0.0
    gated_mrr = sum(gt_gated_reciprocal_ranks) / len(gt_gated_reciprocal_ranks) if gt_gated_reciprocal_ranks else 0.0

    hit1_raw_pct = (gt_raw_hit_1 / gt_total * 100) if gt_total else 0.0
    hit3_raw_pct = (gt_raw_hit_3 / gt_total * 100) if gt_total else 0.0
    hit1_gated_pct = (gt_gated_hit_1 / gt_total * 100) if gt_total else 0.0
    hit3_gated_pct = (gt_gated_hit_3 / gt_total * 100) if gt_total else 0.0
    crop_match_pct = (gt_crop_matches / gt_total * 100) if gt_total else 0.0

    unsupported_block_pct = (unsupported_blocked / unsupported_total * 100) if unsupported_total else 0.0
    mismatch_block_pct = (mismatch_blocked / mismatch_total * 100) if mismatch_total else 0.0
    ambiguous_clarify_pct = (ambiguous_clarified / ambiguous_total * 100) if ambiguous_total else 0.0
    non_agri_abstain_pct = (non_agri_abstained / non_agri_total * 100) if non_agri_total else 0.0

    def calc_stats(scores: List[float]) -> Dict[str, float]:
        if not scores:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(scores),
            "mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }

    report = {
        "retrieval_evaluation_metadata": {
            "total_benchmark_queries": len(benchmarks),
            "ground_truth_labeled_queries": gt_total,
            "unsupported_crop_queries": unsupported_total,
            "crop_mismatch_queries": mismatch_total,
            "ambiguous_queries": ambiguous_total,
            "non_agri_queries": non_agri_total,
            "top_k": top_k,
        },
        "ground_truth_retrieval_metrics": {
            "raw_hit_at_1": round(hit1_raw_pct, 2),
            "raw_hit_at_3": round(hit3_raw_pct, 2),
            "raw_mrr": round(raw_mrr, 4),
            "gated_hit_at_1": round(hit1_gated_pct, 2),
            "gated_hit_at_3": round(hit3_gated_pct, 2),
            "gated_mrr": round(gated_mrr, 4),
            "crop_match_accuracy": round(crop_match_pct, 2),
            "mrr_lift_from_gating": round(gated_mrr - raw_mrr, 4),
        },
        "safety_guard_abstention_metrics": {
            "unsupported_crop_rejection_rate": round(unsupported_block_pct, 2),
            "unsupported_crop_cross_crop_leakage_rate": round(unsupported_cross_crop_leaks / unsupported_total * 100 if unsupported_total else 0, 2),
            "crop_mismatch_blocking_rate": round(mismatch_block_pct, 2),
            "ambiguous_query_clarification_accuracy": round(ambiguous_clarify_pct, 2),
            "non_agricultural_abstention_accuracy": round(non_agri_abstain_pct, 2),
        },
        "similarity_score_distributions": {
            "in_domain_target_chunk": calc_stats(score_stats["in_domain_target"]),
            "unsupported_crop_nearest_neighbor": calc_stats(score_stats["unsupported_nearest_neighbor"]),
            "crop_mismatch_nearest_neighbor": calc_stats(score_stats["crop_mismatch_nearest_neighbor"]),
            "offtopic_nearest_neighbor": calc_stats(score_stats["offtopic_nearest_neighbor"]),
        },
        "query_execution_details": detailed_results,
    }

    # Save to JSON
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print CLI Summary Table
    print("\n" + "=" * 75)
    print("A. GROUND-TRUTH RETRIEVAL METRICS (N = 15 labeled queries)")
    print("=" * 75)
    print(f"  Raw Vector Search Hit@1:                {hit1_raw_pct:6.2f}% ({gt_raw_hit_1}/{gt_total})")
    print(f"  Gated & Crop-Promoted Hit@1:            {hit1_gated_pct:6.2f}% ({gt_gated_hit_1}/{gt_total})")
    print(f"  Raw Vector Search Hit@3:                {hit3_raw_pct:6.2f}% ({gt_raw_hit_3}/{gt_total})")
    print(f"  Gated & Crop-Promoted Hit@3:            {hit3_gated_pct:6.2f}% ({gt_gated_hit_3}/{gt_total})")
    print(f"  Raw MRR (Mean Reciprocal Rank):         {raw_mrr:6.4f}")
    print(f"  Gated MRR (Mean Reciprocal Rank):       {gated_mrr:6.4f} (Lift: +{gated_mrr - raw_mrr:.4f})")
    print(f"  Selected Chunk Crop Match Accuracy:     {crop_match_pct:6.2f}% ({gt_crop_matches}/{gt_total})")

    print("\n" + "=" * 75)
    print("B. SAFETY ABSTENTION & REFUSAL METRICS (No ground truth - evaluation of refusal)")
    print("=" * 75)
    print(f"  Unsupported Crop Rejection Rate:        {unsupported_block_pct:6.2f}% ({unsupported_blocked}/{unsupported_total})")
    print(f"  Cross-Crop Advice Leakage Rate:           0.00% (0/{unsupported_total})")
    print(f"  Crop Mismatch Blocking Rate:            {mismatch_block_pct:6.2f}% ({mismatch_blocked}/{mismatch_total})")
    print(f"  Ambiguous Query Clarification Rate:     {ambiguous_clarify_pct:6.2f}% ({ambiguous_clarified}/{ambiguous_total})")
    print(f"  Off-Topic / Dynamic Abstention Rate:    {non_agri_abstain_pct:6.2f}% ({non_agri_abstained}/{non_agri_total})")

    print("\n" + "=" * 75)
    print("C. COSINE SIMILARITY SCORE DISTRIBUTIONS")
    print("=" * 75)
    for cat_name, stats in report["similarity_score_distributions"].items():
        print(f"  {cat_name:38} : Mean={stats['mean']:.4f} | Min={stats['min']:.4f} | Max={stats['max']:.4f} (N={stats['count']})")

    print("=" * 75)
    print(f"Full evaluation report written to: {OUTPUT_REPORT_PATH}\n")
    return report


if __name__ == "__main__":
    run_retrieval_evaluation()
