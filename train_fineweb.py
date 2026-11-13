import argparse
import csv
import json
import re
import time
from pathlib import Path

import matplotlib
import tiktoken
import torch
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PROMPT = "Don't care about"
SEED = 42
CONTEXT = 64
WIDTH = 64
BATCH = 8


class TinyTransformer(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.tokens = nn.Embedding(vocabulary_size, WIDTH)
        self.positions = nn.Embedding(CONTEXT, WIDTH)
        layer = nn.TransformerEncoderLayer(
            WIDTH, 4, dim_feedforward=128, dropout=0, batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, vocabulary_size))
        self.register_buffer(
            "mask", torch.ones(CONTEXT, CONTEXT, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, ids):
        length = ids.size(1)
        hidden = self.tokens(ids) + self.positions(torch.arange(length))
        return self.head(self.blocks(hidden, mask=self.mask[:length, :length]))


def load_tokens(corpus_path, token_path, encoding):
    # Read bytes to preserve the corpus exactly, including any newline characters.
    expected = encoding.encode_ordinary(corpus_path.read_bytes().decode("utf-8-sig"))
    try:
        raw = token_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raw = ""
    valid_format = re.fullmatch(r"[0-9,\s\[\]]*", raw)
    actual = [int(item) for item in re.findall(r"[0-9]+", raw)] if valid_format else []
    if not valid_format or actual != expected or not token_path.exists():
        token_path.write_text(" ".join(map(str, expected)) + "\n", encoding="utf-8")
        print(f"Regenerated token IDs in {token_path}", flush=True)
    else:
        print(f"Validated {len(expected)} token IDs.", flush=True)
    return expected


@torch.inference_mode()
def sample(model, encoding, vocabulary, to_internal, prompt, prompt_ids, step, destination):
    import codecs

    model.eval()
    generator = torch.Generator().manual_seed(SEED)
    ids = [to_internal[token] for token in prompt_ids]
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    generated_ids = []
    chunks = []
    print(f"\nSample at step {step}\n{prompt}", end="", flush=True)
    for _ in range(40):
        context = torch.tensor([ids[-CONTEXT:]], dtype=torch.long)
        probabilities = model(context)[0, -1].softmax(dim=-1)
        token = torch.multinomial(probabilities, 1, generator=generator).item()
        ids.append(token)
        original = vocabulary[token]
        generated_ids.append(original)
        chunk = decoder.decode(encoding.decode_single_token_bytes(original))
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    tail = decoder.decode(b"", final=True)
    print(tail, flush=True)
    record = {"step": step, "seed": SEED, "prompt": prompt,
              "completion": "".join(chunks) + tail, "token_ids": generated_ids}
    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    destination.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("fineweb_corpus.txt"))
    parser.add_argument("--tokens", type=Path, default=Path("fineweb_tokens.txt"))
    parser.add_argument("--output", type=Path, default=Path("fineweb_run"))
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.seconds <= 0 or args.threads < 1:
        parser.error("--seconds and --threads must be positive")
    if args.corpus.resolve() == args.tokens.resolve():
        parser.error("--corpus and --tokens must refer to different files")

    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    encoding = tiktoken.get_encoding("o200k_base")
    original_ids = load_tokens(args.corpus, args.tokens, encoding)
    vocabulary = sorted(set(original_ids))
    to_internal = {token: index for index, token in enumerate(vocabulary)}
    if len(original_ids) <= CONTEXT:
        parser.error(f"Corpus must contain more than {CONTEXT} tokens")
    prompt = PROMPT
    prompt_ids = encoding.encode_ordinary(prompt)
    if any(token not in to_internal for token in prompt_ids):
        prompt = " " + PROMPT
        prompt_ids = encoding.encode_ordinary(prompt)
    if any(token not in to_internal for token in prompt_ids):
        prompt_ids = original_ids[:8]
        prompt = encoding.decode(prompt_ids)
    print(f"Sampling prompt: {prompt!r}", flush=True)
    corpus = torch.tensor([to_internal[token] for token in original_ids])
    offsets = torch.arange(CONTEXT + 1)
    model = TinyTransformer(len(vocabulary))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    args.output.mkdir(parents=True, exist_ok=True)
    losses = []
    elapsed = 0.0
    step = 0
    with (
        (args.output / "training_log.csv").open("w", newline="", encoding="utf-8") as log,
        (args.output / "samples.jsonl").open("w", encoding="utf-8") as samples,
    ):
        writer = csv.writer(log)
        writer.writerow(["step", "loss", "elapsed_seconds", "tokens_per_second"])
        sample(model, encoding, vocabulary, to_internal, prompt, prompt_ids, step, samples)
        print("\nstep | loss | elapsed time | tokens/second", flush=True)
        while elapsed < args.seconds:
            model.train()
            start = time.perf_counter()
            starts = torch.randint(len(corpus) - CONTEXT, (BATCH, 1))
            batch = corpus[starts + offsets]
            logits = model(batch[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, len(vocabulary)), batch[:, 1:].reshape(-1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            value = loss.item()
            elapsed += time.perf_counter() - start
            step += 1
            losses.append(value)
            rate = step * BATCH * CONTEXT / elapsed
            writer.writerow([step, value, elapsed, rate])
            log.flush()
            print(f"{step} | {value:.4f} | {elapsed:.2f}s | {rate:.1f}", flush=True)
            if step % 20 == 0:
                sample(model, encoding, vocabulary, to_internal, prompt, prompt_ids, step, samples)
        if step % 20 != 0:
            sample(model, encoding, vocabulary, to_internal, prompt, prompt_ids, step, samples)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "internal_to_o200k_base": vocabulary,
        "encoding": encoding.name,
        "step": step,
        "context": CONTEXT,
        "prompt": prompt,
        "prompt_o200k_ids": prompt_ids,
        "seed": SEED,
    }, args.output / "checkpoint.pt")
    fig, axis = plt.subplots()
    axis.plot(range(1, step + 1), losses)
    axis.set(xlabel="Update", ylabel="Next-token cross-entropy")
    fig.tight_layout()
    fig.savefig(args.output / "loss.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved results to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
