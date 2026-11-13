"""Separate CLI for the local agent; the legacy inference CLI is unaffected."""
import argparse
from functools import cache
import getpass
import json
from pathlib import Path
import sys

from .agent_policy import Permissions, TOOL_NAMES, strict_json, validate_inputs
from .config import Config
from .errors import PrivacyError


def terminal_json(value):
    # JSON escapes C0 controls. Escape C1/DEL as well on terminals.
    return "".join(f"\\u{ord(c):04x}" if 0x7f <= ord(c) <= 0x9f else c
                   for c in json.dumps(value, ensure_ascii=False))


def approve_on_tty(review):
    # Do not accept consent from a redirected input stream or from model output.
    with open("/dev/tty", "r", encoding="utf-8") as source, open("/dev/tty", "w", encoding="utf-8") as tty:
        tty.write("以下是本次请求的目的地与正文（仅显示在本地终端）：\n")
        tty.write(terminal_json({"endpoint": review.endpoint, "body": review.payload()})+"\n")
        tty.write("核对后如同意发送，请输入 SEND：")
        tty.flush()
        return source.readline(32).rstrip("\r\n") == "SEND"


def read_input(path, limit):
    if path is None:
        return getpass.getpass("任务（隐藏输入）："), getpass.getpass("上下文（隐藏输入）：")
    with path.open("rb") as source:
        raw = source.read(limit * 12 + 1025)
    if len(raw)>limit*12+1024:
        raise PrivacyError("Agent 输入文件过大")
    data = strict_json(raw.decode("utf-8"))
    if type(data) is not dict or set(data) != {"task", "context"}:
        raise PrivacyError("输入 JSON 必须仅包含 task 与 context")
    return data["task"], data["context"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="本地 Qwen3 隐私 Agent")
    parser.add_argument("--config", default="config.json", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--route", choices=["auto", "local", "cloud"], default="auto")
    parser.add_argument("--allow-cloud", action="store_true")
    parser.add_argument("--allow-tool", action="append", choices=sorted(TOOL_NAMES), default=[])
    parser.add_argument("--shadow-adapter", type=Path, help="本地 LoRA 旁路，只返回统计")
    parser.add_argument("--max-new-tokens", type=int, default=256, choices=range(1,1025),
                        metavar="1..1024")
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config)
        task, context = validate_inputs(*read_input(args.input, config.max_input_chars),
                                        config.max_input_chars)
        from .agent import PrivacyAgent
        from .cloud import DeepSeekClient
        from .local import LocalModel

        shadow = None
        if args.shadow_adapter:
            from .privacy_judge import adapter_shadow

            @cache
            def load_shadow():
                return adapter_shadow(config.model_path, str(args.shadow_adapter.resolve(strict=True)))

            def shadow(task, context):
                return load_shadow().inspect(task, context)
        agent = PrivacyAgent(LocalModel(config),
                             DeepSeekClient(config.cloud_model, config.cloud_timeout, config.cloud_max_tokens),
                             permissions=Permissions(args.allow_cloud, frozenset(args.allow_tool)),
                             max_chars=config.max_input_chars, shadow=shadow)
        result = agent.run(task, context, preferred_route=args.route, approve=approve_on_tty,
                           max_new_tokens=args.max_new_tokens)
        print(terminal_json(result))
        return 1 if result["outcome"] == "blocked" else 0
    except PrivacyError as exc:
        print(str(exc), file=sys.stderr)
    except Exception:
        print("Agent 配置、输入或本地运行失败", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
