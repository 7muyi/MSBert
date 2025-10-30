import sys
sys.path.append("/mnt/HDD0/home/yk24/project/msbert")

import argparse
from datetime import datetime
import time
import traceback
import os

from msbert.train import run
from msbert.config import TrainConfig


def main(args) -> None:
    time_str = datetime.now().strftime("%m-%d-%H:%M")

    output_dir = os.path.join(args.output_dir, time_str)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    train_config = TrainConfig(
        # offline process
        pretrained_model= "/mnt/HDD0/home/yk24/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
        neg_sampling=args.neg_sample,
        device=args.device,
    )

    start = time.time()
    try:
        run(train_config, args.dataset_dir, log_dir, checkpoint_dir)
    except Exception as e:
        traceback.print_exc()
        if time.time() - start <= 120:
            import shutil
            shutil.rmtree(output_dir)
    else:
        print(f"{args.model}'s {time_str} training time: {time.time() - start}")


if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/HDD0/home/yk24/project/msbert/output"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/mnt/HDD0/home/yk24/project/msbert/data/Marco/processed/train"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True
    )
    parser.add_argument(
        "--neg_sample",
        action="store_true"
    )

    args = parser.parse_args()
    main(args)

    # os.system("shutdown -h now")