"""Tokenizer for the Inkling2 processor stand-in: SGLang's own loader (it attaches the attributes the scheduler reads,
e.g. additional_stop_token_ids), falling back to transformers."""
def load_tokenizer(path):
    try:
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
        return get_tokenizer(path)
    except ImportError:
        from sglang.srt.utils.hf_transformers import get_tokenizer
        return get_tokenizer(path)
