from src.data.render_page import _split_for, _template_for_split
from src.training.runtime import assert_lora_targets, assert_standard_lora_config, assert_verl_grpo_config, build_verl_command


def test_split_is_stable_for_same_family_and_seed() -> None:
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    first = _split_for("family-a", 42, ratios)
    assert first == _split_for("family-a", 42, ratios)


def test_test_template_is_held_out_from_training() -> None:
    train_templates = {_template_for_split("train", index) for index in range(20)}
    test_templates = {_template_for_split("test", index) for index in range(20)}
    assert train_templates == {0, 1, 2}
    assert test_templates == {3}
    assert train_templates.isdisjoint(test_templates)


def test_verl_generation_group_contract() -> None:
    assert_verl_grpo_config(
        {
            "framework": "verl",
            "rollout_n": 4,
            "train_batch_size": 2,
            "ppo_mini_batch_size": 4,
            "rollout_tensor_model_parallel_size": 1,
            "n_gpus_per_node": 1,
        }
    )


def test_verl_command_uses_grpo_and_custom_reward() -> None:
    from src.common import load_yaml

    command = build_verl_command(load_yaml("configs/grpo.yaml"))
    assert "-m" in command and "verl.trainer.main_ppo" in command
    assert "algorithm.adv_estimator=grpo" in command
    assert "reward.custom_reward_function.name=compute_score" in command
    assert not any("trl" in item.lower() for item in command)


def test_training_stage_checkpoint_order_is_sft_grpo_opd() -> None:
    from src.common import load_yaml

    grpo = load_yaml("configs/grpo.yaml")
    opd = load_yaml("configs/opd.yaml")
    assert grpo["lora_adapter_path"] == "outputs/sft/best"
    assert opd["student_checkpoint"] == "outputs/grpo/best"
    assert opd["teacher_checkpoint"] == "outputs/grpo/best"


def test_lora_scope_rejects_visual_modules() -> None:
    module_names = ["model.layers.0.self_attn.q_proj", "visual.blocks.0.attn.q_proj"]
    try:
        assert_lora_targets(module_names, ["q_proj"])
    except ValueError as exc:
        assert "visual" in str(exc)
    else:
        raise AssertionError("visual LoRA target was not rejected")


def test_standard_lora_rejects_quantized_loading() -> None:
    assert_standard_lora_config({"precision": "bf16", "quantization": "none"})

    for config in (
        {"precision": "bf16", "quantization": "4bit"},
        {"precision": "bf16", "quantization": "none", "load_in_4bit": True},
    ):
        try:
            assert_standard_lora_config(config)
        except ValueError:
            pass
        else:
            raise AssertionError("quantized loading was not rejected for standard LoRA")
