# tokens first

I'm keeping the input path simple: one user message and an optional system
message. That lets me follow the token sequence without batching or conversation
storage getting in the way.

The tokenizer turns text into integer IDs. A token can be a word fragment,
punctuation, whitespace, or part of a character. Comparing decoded strings would
throw away information I need later, so the loop compares token IDs.

Code: [prepare_prompt](../src/specdecode_lab/models.py).

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
```

The template adds role markers and the start of the assistant reply.
`add_special_tokens=False` on the second step avoids adding another set.
Both models get this exact token sequence.

Thinking mode stays off. I'm trying to follow the generation machinery first.
Qwen's [model card](https://huggingface.co/Qwen/Qwen3-8B#switching-between-thinking-and-non-thinking-mode)
documents that template switch.

## ordinary generation

Code: [generate_baseline](../src/specdecode_lab/decoding.py).

Run the prompt through the target once. That's prefill.
The logits at the last prompt position score the first generated token.

Greedy decoding means taking the largest score:

```python
next_id = output.logits.argmax(dim=-1)
```

Then feed that selected token back into the model. Its output scores the next
token. Repeat until EOS or the output cap.

The distinction I keep coming back to: **predicting a token hasn't cached that token.**
Only feeding it through a forward pass does that.

The cache holds attention keys and values for tokens already processed.
It saves rebuilding that history on every step.

After prefill, the ordinary loop only passes one new input token per call.
Passing the whole prompt again alongside its cache would duplicate the prefix.

## small boundaries

The input has no padding. Batch size is one. That's why the forward calls can
use the model's causal mask without a separate padding mask.

`--max-new-tokens` counts generated IDs, including EOS if one is selected.
Display removes special tokens, so the visible text can look shorter.

The decoder calls the model directly. It doesn't use `generate()` or its sampling
defaults, repetition penalties, or other logit processors. This ordinary loop
is the reference behavior for the speculative loop.

To follow this path on your own machine, start with the baseline command in the
[README](../README.md#start-with-ordinary-generation), then read `CachedModel.predict`
and `generate_baseline` in [decoding.py](../src/specdecode_lab/decoding.py). The
next note adds [draft verification](02-checking-a-draft.md) to this same setup.

