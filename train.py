import argparse


def main():
    parser = argparse.ArgumentParser(description="fuminimind Trainer")
    parser.add_argument("--task", type=str, choices=["pretrain", "sft"], required=True)
    args, rest = parser.parse_known_args()

    if args.task == "pretrain":
        from trainer import pretrain as module
    else:
        from trainer import sft as module

    sub_parser = module.get_parser()
    sub_args = sub_parser.parse_args(rest)
    module.run(sub_args)


if __name__ == "__main__":
    main()
