"""CPU-only evaluation and explicit, pinned public-source ingestion."""

import argparse
import json
from pathlib import Path
import tempfile

from .evaluation import evaluate, json_text, markdown_report
from .data import sha256_bytes
from .schema import ValidationError
from .ingestion import ingest_source
from .sources import download_source
from .provenance import download_provenance, verify_forecastbench_subset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foretellmesh")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="audit data and evaluate deterministic baselines")
    evaluate_parser.add_argument("--data", required=True, type=Path)
    evaluate_parser.add_argument("--config", required=True, type=Path)
    evaluate_parser.add_argument("--output", required=True, type=Path, help="new output directory")
    fetch_parser = subparsers.add_parser("fetch", help="download registered immutable source files")
    ingest_parser = subparsers.add_parser("ingest", help="audit native source files and apply provenance annotations")
    for command_parser in (fetch_parser, ingest_parser):
        command_parser.add_argument("--source", required=True,
                                    choices=("prophet_arena", "forecastbench", "prediction_market_analysis"))
        command_parser.add_argument("--registry", type=Path, default=Path("configs/data_sources_v1.json"))
        command_parser.add_argument("--output", type=Path, required=True)
    ingest_parser.add_argument("--data", type=Path, required=True)
    ingest_parser.add_argument("--resolutions", type=Path)
    ingest_parser.add_argument("--annotations", type=Path)
    proof_fetch = subparsers.add_parser("fetch-provenance", help="download and verify pinned provenance artifacts")
    proof_verify = subparsers.add_parser("verify-forecastbench", help="verify the predeclared held-out Manifold cohort")
    for command_parser in (proof_fetch, proof_verify):
        command_parser.add_argument("--plan", type=Path, required=True)
        command_parser.add_argument("--output", type=Path, required=True)
    proof_verify.add_argument("--registry", type=Path, default=Path("configs/data_sources_v1.json"))
    proof_verify.add_argument("--data", type=Path, required=True)
    proof_verify.add_argument("--resolutions", type=Path, required=True)
    proof_verify.add_argument("--proofs", type=Path, required=True)
    synthetic = subparsers.add_parser("generate-sft-warmup", help="generate deterministic synthetic SFT examples")
    synthetic.add_argument("--config", type=Path, required=True)
    index = subparsers.add_parser("index-heldout", help="index all registered ForecastBench questions for SFT exclusion")
    index.add_argument("--registry", type=Path, default=Path("configs/data_sources_v1.json"))
    index.add_argument("--data", type=Path, required=True)
    index.add_argument("--resolutions", type=Path, required=True)
    sft = subparsers.add_parser("build-sft", help="validate supervision and export chronological SFT partitions")
    for name in ("data", "targets", "config", "heldout-index"):
        sft.add_argument("--" + name, type=Path, required=True)
    sft.add_argument("--benchmark-review", type=Path)
    sft_audit = subparsers.add_parser("audit-prophet-sft", help="audit native Prophet records for historical SFT readiness")
    sft_audit.add_argument("--data", type=Path, required=True)
    sft_audit.add_argument("--registry", type=Path, default=Path("configs/data_sources_v1.json"))
    tokenize = subparsers.add_parser("tokenize-sft", help="offline Qwen tokenization, completion-only loss, no truncation")
    for name in ("bundle", "model-manifest", "config"):
        tokenize.add_argument("--" + name, type=Path, required=True)
    for command_parser in (synthetic, index, sft, sft_audit, tokenize):
        command_parser.add_argument("--output", type=Path, required=True)
    collect = subparsers.add_parser("collect-markets", help="archive public Kalshi observations prospectively")
    collect.add_argument("--config", type=Path, default=Path("configs/kalshi_collection_v1.json"))
    labels = subparsers.add_parser("poll-market-labels", help="archive later Kalshi states for an existing capture")
    market_build = subparsers.add_parser("build-market-dataset", help="offline replay of observations and settlement polls")
    for command_parser in (labels, market_build):
        command_parser.add_argument("--capture", type=Path, required=True)
    market_build.add_argument("--labels", type=Path, nargs="*", default=[])
    for command_parser in (collect, labels, market_build):
        command_parser.add_argument("--output", type=Path, required=True)
    poly = subparsers.add_parser("collect-polymarket", help="archive public Polymarket contracts and order books")
    poly.add_argument("--config", type=Path, default=Path("configs/polymarket_collection_v1.json"))
    poly_poll = subparsers.add_parser("poll-polymarket-labels", help="archive Polymarket resolution proofs")
    poly_build = subparsers.add_parser("build-polymarket-dataset", help="offline Polymarket replay")
    for command_parser in (poly_poll, poly_build):
        command_parser.add_argument("--capture", type=Path, required=True)
    poly_build.add_argument("--labels", type=Path, nargs="*", default=[])
    quality = subparsers.add_parser("audit-market-quality", help="verify bundles and group cross-platform events")
    quality.add_argument("--datasets", type=Path, nargs="+", required=True)
    quality.add_argument("--groups", type=Path, required=True)
    quality.add_argument("--heldout-index", type=Path, required=True)
    for command_parser in (poly, poly_poll, poly_build, quality):
        command_parser.add_argument("--output", type=Path, required=True)
    historical = subparsers.add_parser("capture-historical-markets", help="archive predeclared settled FOMC events and historical evidence")
    historical.add_argument("--config", type=Path, default=Path("configs/historical_fomc_collection_v1.json"))
    historical_build = subparsers.add_parser("build-historical-markets", help="offline historical replay with outcome isolation and review gates")
    historical_build.add_argument("--capture", type=Path, required=True)
    historical_build.add_argument("--heldout-index", type=Path, required=True)
    for command_parser in (historical, historical_build):
        command_parser.add_argument("--output", type=Path, required=True)
    agent_plan = subparsers.add_parser("plan-agents", help="inspect task-to-capability routes without model calls")
    agent_check = subparsers.add_parser("check-agent-workflow", help="scripted offline integration check, not a model benchmark")
    from .capabilities import WORKFLOWS
    for command_parser in (agent_plan, agent_check):
        command_parser.add_argument("--config", type=Path, default=Path("configs/capability_agents_v1.json"))
        command_parser.add_argument("--workflow", choices=tuple(WORKFLOWS), default="reviewed_forecast")
    agent_plan.add_argument("--mode", choices=("base", "capability"), default="base")
    agent_check.add_argument("--fixture", type=Path, default=Path("examples/agent_workflow_fixture_v1.json"))
    agent_check.add_argument("--output", type=Path, required=True)
    agent_check.add_argument("--engine", choices=("serial", "langgraph"), default="serial")
    agent_cohort = subparsers.add_parser("prepare-agent-baseline", help="freeze synthetic validation inputs with separate judge artifacts")
    for name in ("raw-dataset", "split-config", "agent-config", "evaluation-config", "output"):
        agent_cohort.add_argument("--" + name, type=Path, required=True)
    capability_data = subparsers.add_parser("build-research-tool-data", help="replayable synthetic capability supervision with event-group isolation")
    for name in ("raw-dataset", "split-config", "config", "output"):
        capability_data.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build-research-tool-data":
            from .research_tool_data import build_research_tool_data
            result = build_research_tool_data(args.raw_dataset, args.split_config, args.config, args.output)
            print(json_text(result["counts"]))
            return 0
        if args.command == "prepare-agent-baseline":
            from .agent_baseline_data import prepare_agent_baseline
            result = prepare_agent_baseline(args.raw_dataset, args.split_config, args.agent_config, args.evaluation_config, args.output)
            print(json_text(result["counts"]))
            return 0
        if args.command == "plan-agents":
            from .capabilities import load_capabilities, route_plan
            config, config_hash = load_capabilities(args.config)
            print(json_text({**route_plan(config, args.workflow, args.mode), "config_sha256": config_hash}))
            return 0
        if args.command == "check-agent-workflow":
            from .agent_check import check_agent_workflow
            report = check_agent_workflow(args.config, args.fixture, args.output, args.workflow, engine=args.engine)
            print(json.dumps({"kind": report["kind"], "status": report["result"]["status"]}))
            return 0 if report["result"]["status"] == "completed" else 1
        if args.command in ("capture-historical-markets", "build-historical-markets"):
            from .historical_market import capture_historical_markets, build_historical_markets
            if args.command == "capture-historical-markets":
                result = capture_historical_markets(args.config, args.output)
            else:
                result = build_historical_markets(args.capture, args.heldout_index, args.output)["counts"]
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "audit-market-quality":
            from .market_quality import audit_market_quality
            result = audit_market_quality(args.datasets, args.groups, args.heldout_index, args.output)
            print(json.dumps(result["counts"], ensure_ascii=False, sort_keys=True))
            return 0
        if args.command in ("collect-polymarket", "poll-polymarket-labels", "build-polymarket-dataset"):
            from .polymarket_dataset import collect_polymarket, poll_polymarket_labels, build_polymarket_dataset
            if args.command == "collect-polymarket":
                result = collect_polymarket(args.config, args.output)
            elif args.command == "poll-polymarket-labels":
                result = poll_polymarket_labels(args.capture, args.output)
            else:
                report = build_polymarket_dataset(args.capture, args.labels, args.output)
                result = {"record_count": report["record_count"], "counts": report["counts"]}
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command in ("collect-markets", "poll-market-labels", "build-market-dataset"):
            from .market_dataset import build_market_dataset, collect_markets, poll_market_labels
            if args.command == "collect-markets":
                result = collect_markets(args.config, args.output)
            elif args.command == "poll-market-labels":
                result = poll_market_labels(args.capture, args.output)
                result = {"polled_markets": len(result["requests"])}
            else:
                result = build_market_dataset(args.capture, args.labels, args.output)
                result = {"record_count": result["record_count"], "counts": result["counts"]}
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command in ("generate-sft-warmup", "index-heldout", "build-sft", "audit-prophet-sft", "tokenize-sft"):
            from .sft_data import audit_prophet_sft, build_sft, make_forecastbench_index
            if args.command == "generate-sft-warmup":
                from .synthetic_sft import generate_synthetic_sft
                result = generate_synthetic_sft(args.config, args.output)
                print(f"Synthetic records: {result['record_count']}; groups: {result['event_group_count']}")
            elif args.command == "index-heldout":
                result = make_forecastbench_index(args.registry, args.data, args.resolutions, args.output)
                print(f"Held-out entries: {len(result['entries'])}")
            elif args.command == "build-sft":
                result = build_sft(args.data, args.targets, args.config, args.heldout_index, args.output, args.benchmark_review)
                print(f"SFT counts: {result['counts']}; excluded/quarantined: {result['quarantined_or_excluded']}")
            elif args.command == "audit-prophet-sft":
                result = audit_prophet_sft(args.data, args.registry, args.output)
                print(f"Ready: {result['ready_count']} / {result['candidate_count']}; review queue: {args.output}")
            else:
                from .sft_tokenize import tokenize_sft
                result = tokenize_sft(args.bundle, args.model_manifest, args.config, args.output)
                print(f"Tokenized: {result['statistics']}")
            return 0
        if args.command == "fetch-provenance":
            result = download_provenance(args.plan, args.output)
            print(f"Verified {len(result['artifacts'])} provenance artifacts in {args.output}")
            return 0
        if args.command == "verify-forecastbench":
            result = verify_forecastbench_subset(plan_path=args.plan, registry=args.registry, data=args.data,
                                                resolutions=args.resolutions, proofs=args.proofs, output=args.output)
            print(f"Verified {result['verified_count']} / {result['cohort_count']} cohort questions")
            print(f"Records: {args.output / 'import/records.jsonl'}")
            return 0
        if args.command == "fetch":
            result = download_source(args.registry, args.source, args.output)
            print(f"Downloaded {len(result['artifacts'])} pinned files to {args.output}")
            return 0
        if args.command == "ingest":
            result = ingest_source(source_name=args.source, registry=args.registry, data=args.data,
                                   output=args.output, resolutions=args.resolutions,
                                   annotations_path=args.annotations)
            print(f"Audit: {args.output / 'audit.md'}")
            print(f"Candidates: {result['candidate_count']}; accepted: {result['accepted_count']}; "
                  f"quarantined: {result['quarantined_count']}; excluded: {result['excluded_count']}")
            return 0
        if args.output.exists():
            raise ValidationError(f"output already exists: {args.output}; use a new run directory")
        report, predictions, splits = evaluate(args.data, args.config)
        config_bytes = args.config.read_bytes()
        if sha256_bytes(config_bytes) != report["experiment"]["config_sha256"]:
            raise ValidationError("config changed during evaluation; rerun with a stable config")
        # Finish validation and scoring before publishing any run artifacts.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".foretellmesh-", dir=args.output.parent) as staging:
            stage = Path(staging) / "run"
            stage.mkdir()
            (stage / "report.json").write_text(json_text(report), encoding="utf-8")
            (stage / "report.md").write_text(markdown_report(report), encoding="utf-8")
            (stage / "split_manifest.json").write_text(json_text(splits.manifest()), encoding="utf-8")
            (stage / "predictions.jsonl").write_text("".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
                for row in predictions
            ), encoding="utf-8")
            (stage / "config.json").write_bytes(config_bytes)
            stage.rename(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"foretellmesh: {exc}\n")
    print(f"Report: {args.output / 'report.md'}")
    print(f"Retained: {report['audit']['retained_counts']}; excluded: {report['audit']['excluded_count']}")
    return 0
