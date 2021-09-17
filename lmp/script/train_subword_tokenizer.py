import argparse
import os
from tokenizers import BertWordPieceTokenizer, Tokenizer

from lmp.tknzr import SUBWORD_TKNZR_OPTS
from lmp.dset.news import NewsDataset
import lmp.util.cfg


def parse_arg() -> argparse.Namespace:
    # Create parser.
    parser = argparse.ArgumentParser(
        'python -m lmp.script.train_subword_tknzr',
        description='Train subword_tknzr.',
    )

    # Create subparser for each subword_tknzr.
    subparsers = parser.add_subparsers(
        dest='subword_tknzr_name', required=True)

    for subword_tknzr_name, subword_tknzr_clss in SUBWORD_TKNZR_OPTS.items():
        # Use subword_tknzr name as CLI argument.
        subword_tknzr_parser = subparsers.add_parser(
            subword_tknzr_name,
            description=f'Training {subword_tknzr_name} subword_tknzr.',
        )

        # Add customized arguments.
        subword_tknzr_clss.train_parser(subword_tknzr_parser)

    return parser.parse_args()


def main():
    args = parse_arg()
    lmp.util.cfg.save(args, exp_name=args.exp_name)
    files = os.listdir("data/tokenized_data")
    files = [os.path.join("data/tokenized_data", x) for x in files]
    # files = ['data/news_v2.3_raw.txt']
    tknzr = SUBWORD_TKNZR_OPTS[args.__dict__[
        "subword_tknzr_name"]](**args.__dict__)
    # tknzr.train(
    #     exp_name=args.exp_name,
    #     files=files,
    #     min_count=args.min_count,
    #     vocab_size=args.max_vocab,
    # )
    print(args.max_vocab)
    print(tknzr.processor.model)
    tknzr = tknzr.load(args.exp_name)
    print(tknzr.tknz("多次發表多多多的多"))
    print(tknzr.enc("多次發表多多多的多"))


if __name__ == "__main__":
    main()
"""
python -m lmp.script.train_subword_tokenizer BertWordPiece \
    --exp_name wordpiece_test \
    --max_vocab 10000 \
    --min_count 4
"""
