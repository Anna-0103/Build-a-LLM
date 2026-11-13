import argparse
import re
from pathlib import Path

import tiktoken
import torch

from train_fineweb import TinyTransformer


def corpus_prompt(corpus, encoding, lookup):
    for match in re.finditer(r"(?:^|(?<=[.!?])\s+)([A-Z][^.!?\r\n]{8,140}[.!?])", corpus):
        sentence = match.group(1)
        words = list(re.finditer(r"\S+", sentence))
        if not 4 <= len(words) <= 18:
            continue
        beginning = sentence[:words[2].end()]
        for prompt in (beginning, " " + beginning):
            ids = encoding.encode_ordinary(prompt)
            if all(token in lookup for token in ids):
                return sentence, prompt, ids
    raise SystemExit("No short corpus sentence with a usable opening was found.")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["sky", "corpus"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    folder = Path("fineweb_run")
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    vocabulary = checkpoint["internal_to_o200k_base"]
    lookup = {token: index for index, token in enumerate(vocabulary)}
    encoding = tiktoken.get_encoding(checkpoint["encoding"])
    context = checkpoint["context"]
    sentence = None
    if args.mode == "sky":
        prompt = "The sky is"
        ids = encoding.encode_ordinary(prompt)
        missing = [token for token in ids if token not in lookup]
        if missing:
            details = ", ".join(f"{encoding.decode([token])!r} (ID {token})" for token in missing)
            raise SystemExit(f"Missing prompt tokens: {details}. Run corpus mode next.")
    else:
        corpus = Path("fineweb_corpus.txt").read_text(encoding="utf-8-sig")
        sentence, prompt, ids = corpus_prompt(corpus, encoding, lookup)

    model = TinyTransformer(len(vocabulary))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    generator = torch.Generator().manual_seed(42)
    sequence = torch.tensor([[lookup[token] for token in ids]]).repeat(100, 1)
    for _ in range(40):
        probabilities = model(sequence[:, -context:])[:, -1].softmax(dim=-1)
        next_ids = torch.multinomial(probabilities, 1, generator=generator)
        sequence = torch.cat((sequence, next_ids), dim=1)

    output = folder / f"{args.mode}_completions.txt"
    with output.open("w", encoding="utf-8") as file:
        if sentence is not None:
            file.write(f"Corpus sentence: {sentence}\n")
        file.write(f"Prompt: {prompt!r}\n\n")
        for number, row in enumerate(sequence.tolist(), start=1):
            completion = encoding.decode([vocabulary[token] for token in row])
            file.write(f"{number}. {completion}\n")
    print(f"Saved 100 completions to {output}")


if __name__ == "__main__":
    main()
