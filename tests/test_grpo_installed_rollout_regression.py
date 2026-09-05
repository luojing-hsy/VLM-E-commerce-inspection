"""Regression checks against the server's installed veRL; no GPU/model load."""
import ast
import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from packaging.version import Version
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
VERL = ROOT / ".venv/lib/python3.12/site-packages/verl"

pytestmark = pytest.mark.skipif(not VERL.is_dir(), reason="requires the patched server veRL environment")

def method(relative, name, scope):
    tree = ast.parse((VERL / relative).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        node,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<installed-verl>", "exec"), scope)
    return scope[name]

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        ROOT / "outputs/sft_qwen35_4b/latest/huggingface", local_files_only=True
    )

@pytest.mark.parametrize("tokens", [[], [1], [1, 2, 3]])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("mask", [False, True])
def test_padding(tokenizer, tokens, side, mask):
    pad = method("experimental/agent_loop/agent_loop.py", "_pad_token_ids", {})
    result = pad(SimpleNamespace(tokenizer=tokenizer), tokens,
                 max_length=8, padding_side=side, return_attention_mask=mask)
    assert tuple(result["input_ids"].shape) == (1, 8)
    expected = [tokenizer.pad_token_id] * (8 - len(tokens))
    expected = expected + tokens if side == "left" else tokens + expected
    assert result["input_ids"].tolist() == [expected]
    if mask:
        expected_mask = [0] * (8 - len(tokens))
        expected_mask = expected_mask + [1] * len(tokens) if side == "left" else [1] * len(tokens) + expected_mask
        assert result["attention_mask"].tolist() == [expected_mask]

@pytest.mark.parametrize("lora", [False, True])
def test_sleep_synchronizes_caches(lora):
    sender, receiver = {"image": True}, {"image": True}
    events = []
    async def sleep(*, level):
        receiver.clear()
        events.append(("sleep", level))
    async def reset_mm_cache():
        sender.clear()
        receiver.clear()
        events.append(("reset_mm_cache",))
    async def reset_encoder_cache():
        events.append(("reset_encoder_cache",))
    engine = SimpleNamespace(sleep=sleep, reset_mm_cache=reset_mm_cache,
                             reset_encoder_cache=reset_encoder_cache)
    fn = method("workers/rollout/vllm_rollout/vllm_async_server.py", "_sleep_hybrid", {
        "is_torch_npu_available": lambda **kw: False,
        "_VLLM_VERSION": Version("0.20.2"),
        "version": SimpleNamespace(parse=Version),
    })
    asyncio.run(fn(SimpleNamespace(engine=engine, lora_as_adapter=lora)))
    assert sender == receiver == {}
    assert events == [("sleep", 1 if lora else 2), ("reset_mm_cache",), ("reset_encoder_cache",)]

@pytest.mark.parametrize("reason,tokens", [
    ("error", []), ("aborted", []), ("error", [1]), ("aborted", [1]), ("completed", [1, 2]),
])
def test_failed_rollout_is_not_used_as_training_data(reason, tokens):
    output = SimpleNamespace(token_ids=tokens, stop_reason=reason, num_preempted=0,
                             log_probs=None, routed_experts=None, extra_fields={})
    server = SimpleNamespace(generate=AsyncMock(return_value=output))
    agent = SimpleNamespace(
        process_multi_modal_info=AsyncMock(return_value={}),
        apply_chat_template=AsyncMock(return_value=[7, 8]),
        _get_mm_processor_kwargs=lambda audios: {},
        server_manager=server, response_length=8,
    )
    fn = method("experimental/agent_loop/single_turn_agent_loop.py", "run", {
        "simple_timer": lambda *args: nullcontext(),
        "uuid4": lambda: SimpleNamespace(hex="test-request"),
        "AgentLoopOutput": lambda **kw: SimpleNamespace(**kw),
    })
    if reason in ("error", "aborted"):
        with pytest.raises(RuntimeError, match=f"request_id=test-request.*stop_reason={reason}"):
            asyncio.run(fn(agent, {}, raw_prompt=[]))
    else:
        result = asyncio.run(fn(agent, {}, raw_prompt=[]))
        assert result.response_ids == tokens
        assert result.response_mask == [1, 1]
