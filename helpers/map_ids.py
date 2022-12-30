import os
import json
import argparse
import pickle

import torch
from transformers import AutoTokenizer, squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV1Processor

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True, cache_dir="..", use_fast=False,
)


def load_and_cache_examples(
    tokenizer, evaluate=False, output_examples=False, name="", train_file=""
):
    # Load data features from cache or dataset file
    input_dir = ".."
    cached_features_file = os.path.join(
        input_dir, "cached_{}_{}".format("dev" if evaluate else "train", name,),
    )

    print("Creating features from dataset file at %s", input_dir)

    processor = SquadV1Processor()
    if evaluate:
        examples = processor.get_dev_examples(".", filename=args.predict_file)
    else:
        examples = processor.get_train_examples(".", filename=train_file)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=not evaluate,
        return_dataset="pt",
        threads=10,
    )

    print("Saving features into cached file %s", cached_features_file)
    torch.save({"features": features, "dataset": dataset}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset", default=None, type=str, required=True,
    )
    parser.add_argument(
        "--noise_type", default=None, type=str, required=True,
    )
    parser.add_argument(
        "--noise_rate", default=None, type=str, required=True,
    )
    parser.add_argument(
        "--working_dir", default=".", type=str, required=True,
    )
    parser.add_argument(
        "--seed", default=None, type=str, required=True,
    )

    args = parser.parse_args()

    SEED = args.seed
    DATASET = args.dataset
    NOISE_TYPE = args.noise_type
    NOISE_RATE = float(args.noise_rate)
    WORK_DIR = args.working_dir

    if NOISE_TYPE == "struct":
        p1 = f"{DATASET}_noisy_TQpaired_Astructured_start"

    elif NOISE_TYPE == "unrel":
        p1 = f"{DATASET}_noisy_corrAT_incorrQ"

    elif NOISE_TYPE == "rand":

        p1 = f"{DATASET}_noisy_TQpair_Arand"
    else:
        print("noise type error", NOISE_TYPE)

    p = f"{WORK_DIR}/datasets_noise_{DATASET}/"

    clean = f"{WORK_DIR}/clean_datasets/train_{DATASET}30k.json"
    p_to_ds = os.path.join(p, p1, f"train_{DATASET}_seed{SEED}_{NOISE_RATE}.json")

    train_dataset = load_and_cache_examples(
        tokenizer,
        evaluate=False,
        output_examples=False,
        name=f"{DATASET}_{NOISE_TYPE}{int(NOISE_RATE * 100)}_{SEED}.pt",
        train_file=p_to_ds,
    )

    dexf = torch.load(
        f"cached_train_{DATASET}_{NOISE_TYPE}{int(NOISE_RATE * 100)}_{SEED}.pt"
    )
    os.remove(f"cached_train_{DATASET}_{NOISE_TYPE}{int(NOISE_RATE * 100)}_{SEED}.pt")

    if DATASET != "newsqa":
        dexf_features = dexf["features"]

    else:
        print("FILTERING FOR NEWS QA")
        dataset = dexf["dataset"].tensors
        features = dexf["features"]
        to_filter = {
            "input_ids": dataset[0],
            "attention_mask": dataset[1],
            "token_type_ids": dataset[2],
            "start_positions": dataset[3],
            "end_positions": dataset[4],
        }
        has_answer = list(
            set(range(len(dexf["dataset"])))
            - set(
                torch.nonzero(to_filter["start_positions"] == 0).view(-1).tolist()
            ).intersection(
                set(torch.nonzero(to_filter["end_positions"] == 0).view(-1).tolist())
            )
        )

        tensors_ds, tensors_features = [], []
        for tens_ds in dexf["dataset"].tensors:
            tensors_ds.append(tens_ds[has_answer])

        dexf_features = [features[i] for i in has_answer]  # features[has_answer]

    with open(p_to_ds) as f:
        train_row = json.load(f)

    with open(clean) as f2:
        clean_row = json.load(f2)

    d = {x: [] for x in range(30_000)}
    for x in dexf_features:
        d[x.example_index].append(x.input_ids)

    for x in range(30_000):
        if (
            train_row["data"][0]["paragraphs"][x]["qas"][0]["answers"][0]["text"]
            == clean_row["data"][0]["paragraphs"][x]["qas"][0]["answers"][0]["text"]
        ):
            d[x].append(1)
        else:
            d[x].append(0)

    p = f"{WORK_DIR}/grad_w_b1_{SEED}_{DATASET}_{NOISE_TYPE}{int(NOISE_RATE * 100)}/"
    input_ids = torch.load(p + "inputs_ids.pt")
    d_rev = {}
    for k, v in d.items():
        for vi in v[:-1]:
            d_rev[tuple(vi)] = v[-1]

    is_clean = []
    for i in input_ids:
        is_clean.append(d_rev[tuple(i.cpu().numpy().tolist()[0])])

    with open(
        f"is_clean_{DATASET}_{NOISE_TYPE}{int(NOISE_RATE * 100)}_b1_seed{SEED}_5ep.pkl",
        "wb",
    ) as f:
        pickle.dump(is_clean, f)


if __name__ == "__main__":
    main()
