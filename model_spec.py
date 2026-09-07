"""Inference parameters + speed presets + Modelfile helpers."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import ollama


@dataclass
class ModelSpec:
    """Runtime generation + performance options (sent on every chat call)."""

    # Quality / sampling
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    seed: int = 0  # 0 = random
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1

    # Length / context (big speed levers)
    num_ctx: int = 4096  # smaller = faster prompt eval + less VRAM
    num_predict: int = 512  # cap answer length (-1 = unlimited)

    # Throughput / hardware
    num_batch: int = 512  # higher often = faster prompt processing (more VRAM)
    num_gpu: int = -1  # -1 = all layers on GPU; lower if VRAM-bound
    num_thread: int = 0  # 0 = Ollama default; CPU path: try physical cores
    use_mmap: bool = True
    use_mlock: bool = False
    numa: bool = False
    main_gpu: int = 0
    low_vram: bool = False  # extra conservative GPU memory

    def to_options(self) -> Dict[str, Any]:
        opts: Dict[str, Any] = {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
            "num_ctx": int(self.num_ctx),
            "repeat_penalty": float(self.repeat_penalty),
            "mirostat": int(self.mirostat),
            "mirostat_tau": float(self.mirostat_tau),
            "mirostat_eta": float(self.mirostat_eta),
            "num_batch": int(self.num_batch),
            "use_mmap": bool(self.use_mmap),
            "use_mlock": bool(self.use_mlock),
            "numa": bool(self.numa),
            "main_gpu": int(self.main_gpu),
            "low_vram": bool(self.low_vram),
        }
        if int(self.num_predict) != -1:
            opts["num_predict"] = int(self.num_predict)
        if int(self.seed) != 0:
            opts["seed"] = int(self.seed)
        if int(self.num_gpu) >= 0:
            opts["num_gpu"] = int(self.num_gpu)
        if int(self.num_thread) > 0:
            opts["num_thread"] = int(self.num_thread)
        return opts

    def set_param(self, key: str, value: str) -> str:
        key = key.strip().lower().replace("-", "_")
        valid = {f.name for f in fields(self)}
        if key not in valid:
            return f"Unknown `{key}`. Valid: {', '.join(sorted(valid))}"
        try:
            current = getattr(self, key)
            if isinstance(current, bool):
                setattr(self, key, value.lower() in ("1", "true", "yes", "on"))
            elif isinstance(current, int):
                setattr(self, key, int(float(value)))
            else:
                setattr(self, key, type(current)(value))
        except Exception as e:
            return f"Invalid value for {key}: {e}"
        return f"{key} = {getattr(self, key)}"

    def apply_preset(self, name: str) -> str:
        """Named speed/quality presets."""
        name = name.strip().lower()
        presets = {
            "fast": dict(
                temperature=0.5,
                top_p=0.85,
                top_k=30,
                num_ctx=2048,
                num_predict=256,
                num_batch=512,
                low_vram=False,
            ),
            "turbo": dict(
                temperature=0.4,
                top_p=0.8,
                top_k=20,
                num_ctx=1024,
                num_predict=128,
                num_batch=256,
                low_vram=True,
            ),
            "balanced": dict(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                num_ctx=4096,
                num_predict=512,
                num_batch=512,
                low_vram=False,
            ),
            "quality": dict(
                temperature=0.8,
                top_p=0.95,
                top_k=40,
                num_ctx=8192,
                num_predict=-1,
                num_batch=512,
                low_vram=False,
            ),
            "longctx": dict(
                temperature=0.6,
                top_p=0.9,
                top_k=40,
                num_ctx=16384,
                num_predict=1024,
                num_batch=256,
                low_vram=True,
            ),
            "gpu": dict(
                num_gpu=-1,  # all layers (omit in options = default all)
                low_vram=False,
                num_batch=512,
                use_mmap=True,
            ),
            "cpu": dict(
                temperature=0.5,
                top_p=0.85,
                top_k=30,
                num_ctx=2048,
                num_predict=256,
                num_batch=128,
                num_gpu=0,
                num_thread=0,  # set physical cores: /params set num_thread N
                use_mmap=True,
                use_mlock=False,
                low_vram=False,
                numa=False,
            ),
            # AMD iGPU / laptop — CPU-first (iGPU rarely helps Ollama)
            "amd": dict(
                temperature=0.4,
                top_p=0.8,
                top_k=20,
                num_ctx=1536,
                num_predict=192,
                num_batch=64,
                num_gpu=0,
                num_thread=0,
                use_mmap=True,
                use_mlock=False,
                low_vram=True,
                numa=False,
            ),
        }
        if name not in presets:
            return f"Unknown preset. Use: {', '.join(presets)}"
        for k, v in presets[name].items():
            setattr(self, k, v)
        # special: num_gpu -1 means "don't force" → leave field as -1 so to_options omits it
        if name == "gpu":
            self.num_gpu = -1
        return f"Applied **{name}** preset.\n\n{self.summary()}"

    def summary(self) -> str:
        lines = ["**Generation / speed parameters**", ""]
        for f in fields(self):
            lines.append(f"- `{f.name}` = `{getattr(self, f.name)}`")
        lines.append("")
        lines.append(
            "Presets: `/speed amd` · `turbo` · `fast` · `cpu` · `balanced` · `quality`"
        )
        return "\n".join(lines)

    def to_modelfile(
        self,
        base_model: str,
        system: str = "",
        template: str = "",
        extra_parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines = [f"FROM {base_model}", ""]
        if system.strip():
            sys_escaped = system.replace('"""', '\\"\\"\\"')
            lines.append(f'SYSTEM """{sys_escaped}"""')
            lines.append("")
        if template.strip():
            lines.append(f'TEMPLATE """{template}"""')
            lines.append("")
        opts = self.to_options()
        if extra_parameters:
            opts.update(extra_parameters)
        for k, v in opts.items():
            if isinstance(v, bool):
                v = "true" if v else "false"
            lines.append(f"PARAMETER {k} {v}")
        return "\n".join(lines) + "\n"


def create_model_from_spec(
    name: str,
    base_model: str,
    spec: ModelSpec,
    system: str = "",
) -> str:
    name = name.strip().replace(" ", "-")
    if not name:
        return "Error: model name required"
    modelfile = spec.to_modelfile(base_model=base_model, system=system)
    try:
        stream = ollama.create(model=name, modelfile=modelfile, stream=True)
        last = ""
        for part in stream:
            if isinstance(part, dict):
                last = part.get("status") or part.get("error") or str(part)
            else:
                last = str(getattr(part, "status", part))
        return f"Created model `{name}` from `{base_model}`.\nLast status: {last}"
    except Exception as e:
        try:
            ollama.create(
                model=name,
                from_=base_model,
                system=system,
                parameters=spec.to_options(),
            )
            return f"Created model `{name}` from `{base_model}` (parameters API)."
        except Exception as e2:
            return f"Create failed: {e} / {e2}\n\nModelfile:\n```\n{modelfile}\n```"


def show_model_info(model: str) -> str:
    try:
        info = ollama.show(model)
        if hasattr(info, "model_dump"):
            data = info.model_dump()
        elif isinstance(info, dict):
            data = info
        else:
            data = {"raw": str(info)}
        interesting = {}
        for k in ("modelfile", "parameters", "template", "details", "model_info", "system"):
            if k in data and data[k]:
                interesting[k] = data[k]
        if not interesting:
            interesting = data
        text = json.dumps(interesting, indent=2, default=str)
        if len(text) > 6000:
            text = text[:6000] + "\n…"
        return f"**Model info: `{model}`**\n\n```json\n{text}\n```"
    except Exception as e:
        return f"Could not show model `{model}`: {e}"
