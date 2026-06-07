from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("CohereForCausalLM")
class CommandR2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.COMMAND_R

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # max_position_embeddings = 8192 in config.json but model was actually
        # trained on 128k context length
        # aya-23 models don't have model_max_length specified
        self.hparams["max_position_embeddings"] = self.find_hparam(["model_max_length", "max_position_embeddings"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)


@ModelBase.register("Cohere2ForCausalLM")
class Cohere2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.COHERE2

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        rotary_pct = self.hparams["rotary_pct"]
        hidden_size = self.hparams["hidden_size"]
        num_attention_heads = self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rotary_pct * (hidden_size // num_attention_heads)))
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Cohere2 runtime in llama.cpp expects no bias tensors;
        # the actual weight only contains 0-value tensors as bias, we can skip them
        if name.endswith(".bias"):
            if torch.any(data_torch != 0):
                raise ValueError(f"Bias tensor {name!r} is not zero.")
            logger.debug(f"Skipping bias tensor {name!r} for Cohere2 conversion.")
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Cohere2MoeForCausalLM")
class Cohere2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.COHERE2_MOE

    def set_vocab(self):
        super().set_vocab()

        # tokenizer_config.json carries stale Command-A templates; chat_template.jinja is canonical
        template_path = self.dir_model / "chat_template.jinja"
        if template_path.is_file():
            with open(template_path, encoding="utf-8") as f:
                self.gguf_writer.add_chat_template(self._normalize_chat_template(f.read()))

    @staticmethod
    def _normalize_chat_template(template: str) -> str:
        # additively map enable_thinking/reasoning_content onto the template's native
        # reasoning/reasoning_effort/thinking variables for chat parser support
        if "enable_thinking" in template or "reasoning_content" in template:
            return template

        replacements = [
            (
                '{%- set reasoning = reasoning if reasoning is not undefined else (false '
                'if reasoning_effort is defined and reasoning_effort | lower == "none" else true) -%}',
                # reasoning_effort must precede enable_thinking, which llama.cpp always defines
                '{%- set reasoning = reasoning if reasoning is not undefined else (false '
                'if reasoning_effort is defined and reasoning_effort | lower == "none" else '
                '(enable_thinking if enable_thinking is defined else true)) -%}',
            ),
            (
                "{%- if msg.thinking -%}\n{{ msg.thinking }}\n    {%- elif msg.content",
                "{%- if msg.thinking -%}\n{{ msg.thinking }}\n    {%- elif msg.reasoning_content -%}\n{{ msg.reasoning_content }}\n    {%- elif msg.content",
            ),
            (
                '{%- elif message.thinking or (message.content and message.content[0].type == "thinking") -%}',
                '{%- elif message.thinking or message.reasoning_content or (message.content and message.content[0].type == "thinking") -%}',
            ),
            (
                '{% if (message.thinking or (message.content and message.content[0].type == "thinking")) and not skip_thinking -%}',
                '{% if (message.thinking or message.reasoning_content or (message.content and message.content[0].type == "thinking")) and not skip_thinking -%}',
            ),
        ]

        normalized = template
        for old, new in replacements:
            if old not in normalized:
                logger.warning("chat_template.jinja changed upstream, keeping it verbatim")
                return template
            normalized = normalized.replace(old, new)
        return normalized

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # intermediate_size is the routed expert FFN size; the leading dense
        # layers use prefix_dense_intermediate_size
        self.n_ff_exp = self.hparams["intermediate_size"]
        self.hparams["intermediate_size"] = self.hparams["prefix_dense_intermediate_size"]

        # AutoConfig replaces first_k_dense_replace with mlp_layer_types
        if (mlp_layer_types := self.hparams.get("mlp_layer_types")) is not None:
            self.n_layer_dense_lead = sum(1 for t in mlp_layer_types if t == "dense")
        else:
            self.n_layer_dense_lead = self.hparams["first_k_dense_replace"]

        if self.hparams.get("num_shared_experts", 0) > 0:
            strategy = self.hparams.get("shared_expert_combination_strategy", "average")
            if strategy == "average":
                self._shexp_scale = 0.5
            elif strategy == "sum":
                self._shexp_scale = 1.0
            else:
                raise ValueError(f"Unknown shared_expert_combination_strategy {strategy!r}")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in self.hparams["layer_types"]])
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["head_dim"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

        self.gguf_writer.add_expert_feed_forward_length(self.n_ff_exp)
        self.gguf_writer.add_leading_dense_block_count(self.n_layer_dense_lead)
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        self.gguf_writer.add_expert_weights_norm(self.hparams["norm_topk_prob"])

        # the shared expert branch is a single MLP of width intermediate_size *
        # num_shared_experts; "average" combines as (routed + shared) / 2, which
        # maps onto existing mechanics as expert_weights_scale = 0.5 on the
        # routed weights and 0.5 folded into the shared down_proj
        if (n_shexp := self.hparams.get("num_shared_experts", 0)) > 0:
            if self._shexp_scale != 1.0:
                self.gguf_writer.add_expert_weights_scale(self._shexp_scale)
            self.gguf_writer.add_expert_shared_count(n_shexp)
            self.gguf_writer.add_expert_shared_feed_forward_length(self.n_ff_exp * n_shexp)

    _experts: list[dict[str, Tensor]] | None = None
    _shexp_scale: float = 1.0

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith("mlp.shared_experts.down_proj.weight") and self._shexp_scale != 1.0:
            data_torch = data_torch * self._shexp_scale

        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
