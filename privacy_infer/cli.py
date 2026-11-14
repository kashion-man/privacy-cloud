import argparse
from dataclasses import asdict, replace
import getpass
import json
import os
from pathlib import Path
import sys

from .config import Config
from .credentials import get_api_key
from .errors import PrivacyError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Qwen3 / privacy-gated DeepSeek inference")
    parser.add_argument("command", choices=["local", "local-batch", "sanitize", "infer"])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--input", type=Path, help="UTF-8 JSON with task and context; otherwise hidden interactive input")
    parser.add_argument("--send", action="store_true", help="Send sanitized fields to DeepSeek (infer only)")
    parser.add_argument("--preview", action="store_true", help="Print sanitized text locally; may still contain missed private information")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--privacy-mode", choices=["redact", "stamp"], default=None,
                        help="Select default redaction or STAMP token-level Polar noise")
    parser.add_argument("--sensitive-importance", choices=["prosan", "cosine"], default=None,
                        help="Importance method for NER-sensitive spans only")
    parser.add_argument("--prosan-threshold", type=float, default=None,
                        help="High/low cutoff for normalized ProSan sensitive importance")
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--stamp-base-epsilon", type=float, default=None,
                         help="Set STAMP budgets to this value times 2:1:4:3")
    budgets.add_argument("--stamp-epsilons", nargs=4, type=float, default=None,
                         metavar=("G1", "G2", "G3", "G4"), help="Explicit per-group angular budgets")
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config)
        overrides = {}
        if args.privacy_mode is not None:
            overrides["privacy_mode"] = args.privacy_mode
        if args.sensitive_importance is not None:
            overrides["sensitive_importance"] = args.sensitive_importance
        if args.prosan_threshold is not None:
            overrides["prosan_threshold"] = args.prosan_threshold
        if args.stamp_base_epsilon is not None:
            overrides.update(stamp_base_epsilon=args.stamp_base_epsilon, stamp_group_epsilons=None)
        if args.stamp_epsilons is not None:
            overrides["stamp_group_epsilons"] = tuple(args.stamp_epsilons)
        config = replace(config, **overrides)
        if args.command in {"local", "local-batch"} and overrides:
            raise PrivacyError("Privacy mode options require sanitize or infer")
        if args.prosan_threshold is not None and config.sensitive_importance != "prosan":
            raise PrivacyError("ProSan threshold requires ProSan sensitive importance")
        if config.privacy_mode != "stamp" and (args.stamp_base_epsilon is not None or args.stamp_epsilons is not None):
            raise PrivacyError("STAMP budget options require --privacy-mode stamp")
        if args.send and args.command != "infer":
            raise PrivacyError("--send requires infer")
        if args.send and not get_api_key():
            raise PrivacyError("Set DEEPSEEK_API_KEY before sending")
        if args.command == "local-batch":
            from .local import LocalModel

            if args.input is None or args.input.stat().st_size > config.max_input_chars * 12 * config.max_batch_size + 1024:
                raise PrivacyError("local-batch requires a bounded JSON input file")
            rows = json.loads(args.input.read_text(encoding="utf-8"))
            if type(rows) is not list or not 1 <= len(rows) <= config.max_batch_size:
                raise PrivacyError("Invalid local batch size")
            messages = []
            for row in rows:
                if type(row) is not dict or set(row) != {"task", "context"}:
                    raise PrivacyError("Batch entries must contain task and context")
                task, context = row["task"], row["context"]
                if type(task) is not str or type(context) is not str or len(task) + len(context) > config.max_input_chars:
                    raise PrivacyError("Invalid input or size limit exceeded")
                messages.append(("You are a helpful assistant.", task + "\n\n" + context))
            model = LocalModel(config)
            try:
                answers = model.answer_many(messages, args.max_new_tokens)
            finally:
                model.release_answer_backend()
            print(json.dumps(answers, ensure_ascii=False))
            return 0
        if args.input:
            if args.input.stat().st_size > config.max_input_chars * 12 + 1024:
                raise PrivacyError("Input file too large")
            data = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or set(data) != {"task", "context"}:
                raise PrivacyError("Input JSON must contain only task and context")
            task, context = data["task"], data["context"]
        else:
            task = getpass.getpass("Task (hidden): ")
            context = getpass.getpass("Context (hidden): ")
        if not isinstance(task, str) or not isinstance(context, str) or len(task) + len(context) > config.max_input_chars:
            raise PrivacyError("Invalid input or size limit exceeded")
        from .local import LocalModel
        model = LocalModel(config)
        if args.command == "local":
            try:
                print(model.answer("You are a helpful assistant.", task + "\n\n" + context, args.max_new_tokens))
            finally:
                model.release_answer_backend()
            return 0
        from .cloud import DeepSeekClient
        from .detection import LocalDetector
        from .pipeline import Pipeline
        from .policy import PrivacyGate
        if config.privacy_mode == "stamp":
            from .stamp import StampGate
            gate = StampGate(LocalDetector(model), model, config)
        else:
            importance = None
            if config.sensitive_importance == "prosan":
                from .prosan import ProSanImportance
                importance = ProSanImportance(model, config)
            gate = PrivacyGate(LocalDetector(model), model, config.relevance_threshold,
                               config.max_input_chars, importance=importance)
        pipeline = Pipeline(gate,
                            DeepSeekClient(os.environ.get("DEEPSEEK_MODEL", config.cloud_model),
                                           config.cloud_timeout, config.cloud_max_tokens))
        request, report, answer = pipeline.run(task, context, dry_run=not args.send)
        print(json.dumps({"sent": args.send, "privacy_mode": config.privacy_mode,
                          "sensitive_importance": config.sensitive_importance,
                          "report": asdict(report)}, ensure_ascii=False))
        if args.preview:
            print(json.dumps({"task": request.task, "context": request.context}, ensure_ascii=False))
        if answer is not None:
            print(answer)
        return 0
    except Exception:
        # Do not print arbitrary exception bodies (may contain prompts, secrets or API output).
        exc = sys.exc_info()[1]
        message = str(exc) if isinstance(exc, PrivacyError) else "Configuration, input or local dependency failure"
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
