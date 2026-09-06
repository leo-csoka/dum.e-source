import argparse
import json
import os

from datasets import load_dataset
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dotenv import load_dotenv

from config import Config
from model import Transformer


def fineweb_batches(model, dataset_name, batch_size, tok):
    # load requested dataset using my access token
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name=dataset_name,
        split="train",
        streaming=True,
        token=tok
    )

    # load all tokens model can take at once
    tokens_per_batch = batch_size * (model.cfg.context_length + 1)
    token_buffer = []

    # encode all text as tokens for model
    for example in dataset:
        token_buffer.extend(model.tokenizer.encoding.encode(example["text"], allowed_special=set()))
        while len(token_buffer) >= tokens_per_batch:
            batch_tokens = token_buffer[:tokens_per_batch]
            del token_buffer[:tokens_per_batch]
            batch = mx.array(batch_tokens).reshape(
                batch_size, model.cfg.context_length + 1
            )
            yield batch[:, :-1], batch[:, 1:]


def loss_fn(model, inputs, targets):
    logits = model(inputs)
    return mx.mean(nn.losses.cross_entropy(logits, targets, reduction="none"))


def train_step(model, optimizer, inputs, targets):
    # get model gradients and update model parameters
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, gradients = loss_and_grad(model, inputs, targets)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    return loss


def save_checkpoint(model, output_dir):
    # save model weights in usr requested directory
    os.makedirs(output_dir, exist_ok=True)
    model.save_weights(os.path.join(output_dir, "weights.npz"))

    # save model configuration alongside model
    config = {
        "token_vector_len": model.cfg.token_vector_len,
        "vocab_size": model.cfg.vocab_size,
        "context_length": model.cfg.context_length,
        "head_dim": model.cfg.head_dim,
        "num_blocks": model.cfg.num_blocks,
        "ffn_dim": model.cfg.ffn_dim,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def main():
    load_dotenv()
    access_token = os.getenv("ACCESS_TOKEN")
    # get all user arguments for training time, etc
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset", default="sample-100BT")
    parser.add_argument("--output", default="checkpoints/small")
    args = parser.parse_args()

    # create model, optimizer, training data
    config = Config()
    model = Transformer(config)
    optimizer = optim.AdamW(
        learning_rate=3e-4,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=0.1,
        bias_correction=True,
    )
    batches = fineweb_batches(model, args.dataset, args.batch_size, access_token)

    # evaluate all user requested training steps
    for step in range(args.steps):
        inputs, targets = next(batches)
        loss = train_step(model, optimizer, inputs, targets)
        print(f"step {step + 1:02d} loss {loss.item():.4f}")

    # save model once all steps are completed
    save_checkpoint(model, args.output)
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()