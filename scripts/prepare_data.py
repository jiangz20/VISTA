"""Convert image-question-answer JSONL to verl active-visual Parquet."""

import argparse
import json
from pathlib import Path
from datasets import Dataset
from vista.prompts import THINK_WITH_IMAGES_SYSTEM_PROMPT


def prepare(source, destination):
    source = Path(source).resolve()
    records = []
    for index, line in enumerate(source.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        image = Path(row["image"])
        image = image if image.is_absolute() else source.parent / image
        if not image.is_file():
            raise FileNotFoundError(image)
        question = str(row["question"]).replace("<image>", "").strip()
        records.append(
            {
                "data_source": "vista_vqa",
                "prompt": [
                    {"role": "system", "content": THINK_WITH_IMAGES_SYSTEM_PROMPT},
                    {"role": "user", "content": "<image>\nQuestion:\n" + question},
                ],
                "images": [{"image": str(image.resolve())}],
                "env_kwargs": {
                    "task_id": str(row.get("id", index)),
                    "image_path": str(image.resolve()),
                    "question": question,
                    "ground_truth": str(row["answer"]),
                },
                "extra_info": {"index": index},
            }
        )
    if not records:
        raise ValueError("empty dataset")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.input, args.output)
