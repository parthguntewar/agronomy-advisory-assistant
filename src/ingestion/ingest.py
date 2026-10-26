"""
Ingestion pipeline for KCC Agronomy knowledge base.
Reads source records, applies strict quality filters, and generates processed knowledge chunks.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

from src.config import (
    PROCESSED_DIR,
    PROCESSED_KB,
    INGESTION_REPORT,
    SAMPLE_RAG_KB,
)
from src.ingestion.schema import KnowledgeBaseChunk, validate_and_filter_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion(
    input_file: Path = SAMPLE_RAG_KB,
    output_file: Path = PROCESSED_KB,
    report_file: Path = INGESTION_REPORT,
) -> Dict[str, Any]:
    """
    Executes the ingestion pipeline on the input dataset.
    
    Args:
        input_file: Path to source JSONL file.
        output_file: Path to output processed knowledge base JSONL.
        report_file: Path to output JSON summary report.
        
    Returns:
        Summary dictionary with ingestion statistics.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    report_path = Path(report_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Source file not found at: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting ingestion from: {input_path}")

    valid_chunks: List[KnowledgeBaseChunk] = []
    rejection_log: List[Dict[str, Any]] = []
    rejection_counts: Counter = Counter()
    domain_counts: Counter = Counter()
    crop_counts: Counter = Counter()
    state_counts: Counter = Counter()
    rag_mode_counts: Counter = Counter()

    total_rows = 0

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            line_str = line.strip()
            if not line_str:
                continue
            total_rows += 1

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as e:
                rejection_counts["malformed_json"] += 1
                rejection_log.append({
                    "row_index": idx,
                    "example_id": f"row_{idx}",
                    "reason": f"malformed_json: {str(e)}"
                })
                continue

            is_valid, reason, chunk = validate_and_filter_record(record)

            if is_valid and chunk is not None:
                valid_chunks.append(chunk)
                domain_counts[chunk.advisory_domain] += 1
                crop_counts[chunk.crop] += 1
                state_counts[chunk.state] += 1
                rag_mode_counts[chunk.rag_use_mode] += 1
            else:
                rejection_counts[reason] += 1
                rejection_log.append({
                    "row_index": idx,
                    "example_id": record.get("metadata", {}).get("example_id") or record.get("example_id", f"row_{idx}"),
                    "reason": reason
                })

    # Write processed knowledge base chunks
    with open(output_path, "w", encoding="utf-8") as f_out:
        for chunk in valid_chunks:
            f_out.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(valid_chunks)} validated chunks to: {output_path}")

    # Build summary report
    report = {
        "source_input_file": str(input_path),
        "processed_output_file": str(output_path),
        "total_records_processed": total_rows,
        "valid_chunks_retained": len(valid_chunks),
        "rejected_records_count": len(rejection_log),
        "retention_rate_pct": round((len(valid_chunks) / total_rows * 100), 2) if total_rows > 0 else 0.0,
        "rejection_breakdown": dict(rejection_counts.most_common()),
        "domain_distribution": dict(domain_counts.most_common()),
        "crop_distribution": dict(crop_counts.most_common()),
        "state_distribution": dict(state_counts.most_common()),
        "rag_mode_distribution": dict(rag_mode_counts.most_common()),
        "sample_rejections": rejection_log[:15],
    }

    with open(report_path, "w", encoding="utf-8") as f_rep:
        json.dump(report, f_rep, indent=2, ensure_ascii=False)

    logger.info(f"Ingestion report written to: {report_path}")
    logger.info(f"Summary: {len(valid_chunks)} accepted, {len(rejection_log)} rejected out of {total_rows}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest and filter KCC agronomy knowledge base")
    parser.add_argument("--input", type=str, default=str(SAMPLE_RAG_KB), help="Path to input JSONL file")
    parser.add_argument("--output", type=str, default=str(PROCESSED_KB), help="Path to output JSONL file")
    parser.add_argument("--report", type=str, default=str(INGESTION_REPORT), help="Path to output report file")
    args = parser.parse_args()

    run_ingestion(Path(args.input), Path(args.output), Path(args.report))
