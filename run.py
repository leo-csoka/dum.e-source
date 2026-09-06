import argparse

import mlx.core as mx
from model import load_checkpoint


def sample_next_token(logits, temperature, top_k):
    # top token chosen deterministically
    if temperature == 0:
        return int(mx.argmax(logits, axis=-1).item())
    # choose some token from the top k choices
    if top_k > 0 and top_k < logits.shape[-1]:
        top_indices = mx.topk(logits, top_k)
        top_values = mx.take_along_axis(logits, top_indices, axis=-1)
        cutoff = mx.min(top_values, axis=-1, keepdims=True)
        logits = mx.where(logits >= cutoff, logits, -float("inf"))

    # pick a token from the top logits
    token = mx.random.categorical(logits / temperature, axis=-1)
    return int(token.item())


def main():
    # get all user args
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/small")
    parser.add_argument("--outputlen", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    # safety checks
    if args.outputlen < 0:
        parser.error("--outputlen must be non-negative")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")

    model = load_checkpoint(args.checkpoint)
    text = input("Text: ")
    token_ids = model.tokenizer.encoding.encode(text, allowed_special=set())
    context_length = model.cfg.context_length

    print("\n\n")
    print(text, end="")

    # evaluate all tokens
    for _ in range(args.outputlen):
        inputs = mx.array([token_ids[-context_length:]])
        logits = model(inputs)
        next_token_id = sample_next_token(
            logits[:, -1, :], args.temperature, args.top_k
        )
        token_ids.append(next_token_id)
        print(model.tokenizer.decode([next_token_id]), end="", flush=True)

    print()


if __name__ == "__main__":
    main()