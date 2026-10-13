"""Command-line entry points for the RSAM training and inference workflow."""

import argparse
import json
import os

import torch


def main():
    parser = argparse.ArgumentParser(prog="rsam")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--manifest", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--resume")
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--device")
    inference = commands.add_parser("infer")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--manifest", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--split", default="test", choices=["train", "val", "test"])
    inference.add_argument("--batch-size", type=int, default=1)
    inference.add_argument("--device")
    visual = commands.add_parser("visualize", help="Render MRI, synthetic CT, and optional paired CT")
    visual.add_argument("--mr", required=True)
    visual.add_argument("--prediction", required=True)
    visual.add_argument("--ct", help="Optional registered paired CT on the same grid")
    visual.add_argument("--slice", type=int, required=True)
    visual.add_argument("--output", required=True)
    visual.add_argument("--ct-min", type=float, default=-1000)
    visual.add_argument("--ct-max", type=float, default=2000)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "train":
        from rsam.train import train

        train(
            args.config,
            args.manifest,
            args.output,
            args.resume,
            args.max_steps,
            args.device,
        )
    elif args.command == "infer":
        from rsam.infer import infer_manifest

        rank, world = (
            int(os.environ.get("RANK", 0)),
            int(os.environ.get("WORLD_SIZE", 1)),
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = args.device or (
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
        infer_manifest(
            args.checkpoint,
            args.manifest,
            args.output,
            args.split,
            device,
            args.batch_size,
            rank,
            world,
        )
    elif args.command == "visualize":
        from rsam.visualize import visualize
        visualize(args.mr, args.prediction, args.output, args.slice, args.ct,
                  (args.ct_min, args.ct_max))
    elif args.command == "inspect":
        from rsam.checkpoint import load_generator, sha256

        model, bundle = load_generator(args.checkpoint)
        print(
            json.dumps(
                {key: value for key, value in bundle.items() if key != "state_dict"},
                indent=2,
            )
        )
        print(
            json.dumps(
                {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "sha256": sha256(args.checkpoint),
                }
            )
        )


if __name__ == "__main__":
    main()
