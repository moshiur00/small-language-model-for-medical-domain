## Tokenizer comparison

The project trains a custom GPT-2-style ByteLevel BPE tokenizer and
compares it against the original GPT-2 tokenizer.

The comparison evaluates both tokenizers on the same validation and
test documents using:

- vocabulary size
- total token count
- tokens per word
- characters per token
- bytes per token
- unknown-token rate
- vocabulary utilization
- document sequence lengths
- medical-term fragmentation

Run the comparison:

```bash
python scripts/tokenizer/compare_tokenizers.py \
    --config configs/tokenizer.yaml
```
