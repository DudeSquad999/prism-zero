"""Local-only Hugging Face generation with lazy optional ML imports."""
from __future__ import annotations
from pathlib import Path
from typing import Any

SUPPORTED_DEVICES = ('cpu', 'cuda', 'mps', 'xpu')


def local_model_path(path: str) -> Path:
    result = Path(path).expanduser().resolve(strict=True)
    if not result.is_dir():
        raise ValueError('Expected a local checkpoint directory')
    return result


def build_messages(prompt: str, feedback: str | None = None) -> list[dict[str, str]]:
    content = prompt + ('\n\nVerifier feedback: ' + feedback if feedback else '')
    system = ('Solve the task. Return a concise explanation in <think> tags '
              'then the final answer in <answer> tags. Return no other text.')
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': content}]


class LocalHFGenerator:
    """Load trusted local safetensors; require a compatible chat template.

    An optional local adapter directory is loaded on top of the base model for
    inference and evaluation. Nothing is downloaded and adapters are never
    promoted automatically.
    """
    def __init__(self, model_path: str, max_new_tokens: int = 256,
                 max_input_tokens: int = 2048, device: str = 'cpu',
                 adapter_path: str | None = None) -> None:
        for name, value in (('max_new_tokens', max_new_tokens), ('max_input_tokens', max_input_tokens)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be a plain integer')
        if not 1 <= max_new_tokens <= 8192 or max_input_tokens < 32:
            raise ValueError('max_new_tokens must be in [1, 8192] and max_input_tokens >= 32')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not isinstance(device, str) or not device.strip():
            raise ValueError('Device must be a nonempty string such as cpu, cuda, or cuda:0')
        name, _, index = device.partition(':')
        if name not in SUPPORTED_DEVICES or (index and not index.isdigit()):
            raise ValueError(f'Unsupported device {device!r}; expected one of {SUPPORTED_DEVICES} '
                             'with an optional index such as cuda:0')
        if name == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but torch.cuda.is_available() is false')
        path = str(local_model_path(model_path))
        self.torch: Any = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                            trust_remote_code=False)
        try:
            self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': 'template check'}],
                tokenize=False, add_generation_prompt=True)
        except Exception as error:
            raise ValueError('Checkpoint tokenizer has no usable chat template; '
                             'provide one in tokenizer_config.json') from error
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True).to(device)
        if adapter_path is not None:
            from peft import PeftModel
            adapter = str(local_model_path(adapter_path))
            self.model = PeftModel.from_pretrained(self.model, adapter, local_files_only=True)
        self.model.eval()
        self.max_new_tokens, self.max_input_tokens = max_new_tokens, max_input_tokens

    def generate(self, prompt: str, feedback: str | None) -> str:
        rendered = self.tokenizer.apply_chat_template(build_messages(prompt, feedback),
                    tokenize=False, add_generation_prompt=True)
        batch = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False)
        length = batch['input_ids'].shape[1]
        context = getattr(self.model.config, 'max_position_embeddings', None)
        if length > self.max_input_tokens or (isinstance(context, int) and length + self.max_new_tokens > context):
            raise ValueError('Token budget exceeded; refusing silent truncation')
        batch = {key: value.to(self.model.device) for key, value in batch.items()}
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        with self.torch.inference_mode():
            generated = self.model.generate(**batch, do_sample=False,
                        max_new_tokens=self.max_new_tokens, pad_token_id=pad)
        return self.tokenizer.decode(generated[0, length:], skip_special_tokens=True)
