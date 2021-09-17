r""":term:`Tokenizer` base class."""

import abc
import argparse
import json
import os
import re
from typing import ClassVar, Dict, List, Optional, Sequence
from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer, AddedToken, decoders, trainers, normalizers
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import NFKC, BertNormalizer

import lmp.dset
import lmp.dset.util
import lmp.path
import lmp.util.cfg


class BertWordPiece():
    bos_tk: ClassVar[str] = '[bos]'
    bos_tkid: ClassVar[int] = 0
    eos_tk: ClassVar[str] = '[eos]'
    eos_tkid: ClassVar[int] = 1
    file_name: ClassVar[str] = 'tknzr.json'
    pad_tk: ClassVar[str] = '[pad]'
    pad_tkid: ClassVar[int] = 2
    tknzr_name: ClassVar[str] = 'BertWordPiece'
    unk_tk: ClassVar[str] = '[unk]'
    unk_tkid: ClassVar[int] = 3

    def __init__(
        self,
        clean_text: bool = True,
        handle_chinese_chars: bool = False,
        strip_accents: Optional[bool] = None,
        lowercase: bool = True,
        file_path: Optional[str] = None,
        **kwargs: Optional[Dict],
    ):
        if file_path:
            self.processor = Tokenizer.from_file(file_path)
        else:
            # self.processor = BertWordPieceTokenizer(
            #     unk_token=self.__class__.unk_tk,
            #     sep_token=self.__class__.eos_tk,
            #     cls_token=self.__class__.bos_tk,
            #     pad_token=self.__class__.pad_tk,
            #     clean_text=clean_text,
            #     handle_chinese_chars=handle_chinese_chars,
            #     strip_accents=strip_accents,
            #     lowercase=lowercase,
            # )
            self.processor = Tokenizer(
                WordPiece(unk_token=self.__class__.unk_tk)
            )
            self.processor.normalizer = normalizers.Sequence(
                [
                    NFKC(),
                    BertNormalizer(
                        clean_text=clean_text,
                        handle_chinese_chars=handle_chinese_chars,
                        strip_accents=strip_accents,
                        lowercase=lowercase,
                    )
                ]
            )
            self.processor.pre_tokenizer = BertPreTokenizer()

            self.processor.post_processor = BertProcessing(
                (str(self.__class__.eos_tk), self.__class__.eos_tkid),
                (str(self.__class__.bos_tk), self.__class__.bos_tkid)
            )
            self.processor.decoder = decoders.WordPiece(prefix="")

    def save(self, exp_name: str) -> None:
        file_dir = os.path.join(lmp.path.EXP_PATH, exp_name)
        file_path = os.path.join(file_dir, self.__class__.file_name)

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        elif not os.path.isdir(file_dir):
            raise FileExistsError(f'{file_dir} is not a directory.')

        elif os.path.isdir(file_path):
            raise FileExistsError(f'{file_path} is a directory.')

        self.processor.save(file_path, pretty=True)

    @classmethod
    def load(cls, exp_name: str):
        if not isinstance(exp_name, str):
            raise TypeError('`exp_name` must be an instance of `str`.')

        if not exp_name:
            raise ValueError('`exp_name` must be non-empty.')

        file_path = os.path.join(lmp.path.EXP_PATH, exp_name, cls.file_name)
        cfg = lmp.util.cfg.load(exp_name=exp_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(' '.join([
                f'Tokenizer file path {file_path} does not exist.',
                'You must run `python -m lmp.script.train_tokenizer` first.',
            ]))

        if os.path.isdir(file_path):
            raise FileExistsError(' '.join([
                f'Tokenizer file path {file_path} is a directory.',
                f'Remove {file_path} first then do',
                '`python -m lmp.script.train_tokenizer`.',
            ]))

        return cls(file_path=file_path, **cfg.__dict__)

    def tknz(self, txt: str) -> List[str]:
        out = self.processor.encode(txt)
        return out.tokens

    def enc(self, txt: str, *, max_seq_len: Optional[int] = -1) -> List[int]:
        if max_seq_len != -1:
            self.processor.enable_truncation(max_length=max_seq_len)
            self.processor.enable_padding(
                length=max_seq_len,
                pad_id=self.__class__.pad_tkid,
                pad_token=self.__class__.pad_tk,
            )
        out = self.processor.encode(txt)
        return out.ids

    def dec(
            self,
            tkids: Sequence[int],
            *,
            rm_sp_tks: Optional[bool] = False,
    ) -> str:

        out = self.processor.decode(tkids, skip_special_tokens=False)
        if rm_sp_tks:
            out.replace(self.__class__.bos_tk, "")
            out.replace(self.__class__.pad_tk, "")
            out.replace(self.__class__.eos_tk, " ")
        return out

    def batch_enc(
            self,
            batch_txt: Sequence[str],
            *,
            max_seq_len: int = -1,
    ) -> List[List[int]]:
        if max_seq_len != -1:
            self.processor.enable_truncation(max_length=max_seq_len)
            self.processor.enable_padding(
                length=max_seq_len,
                pad_id=self.__class__.pad_tkid,
                pad_token=self.__class__.pad_tk,
            )
        out = self.processor.encode_batch(batch_txt)
        return out.ids

    def batch_dec(
            self,
            batch_tkids: Sequence[Sequence[int]],
            *,
            rm_sp_tks: bool = False,
    ) -> List[str]:

        # Decode each sequence of token ids in the batch.
        return [self.dec(tkids, rm_sp_tks=rm_sp_tks) for tkids in batch_tkids]

    def train(
        self,
        exp_name: str,
        files: List[str],
        min_count: int,
        vocab_size: int,
    ) -> None:
        special_tokens = [
            self.__class__.bos_tk,
            self.__class__.eos_tk,
            self.__class__.pad_tk,
            self.__class__.unk_tk,
        ]
        add_tokens = ["<en>", "<num>", ]
        initial_alphabet = ['，', ',', '。', '：',
                            ':', '；', ';', '！', '!', '？', '?']
        for i in range(20):
            add_tokens.append(f'<per{i}>')
            add_tokens.append(f'<org{i}>')
            add_tokens.append(f'<loc{i}>')

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_count,
            limit_alphabet=7800,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens,
            continuing_subword_prefix="",
        )
        self.processor.train(files=files, trainer=trainer)
        self.processor.add_tokens(add_tokens)
        self.save(exp_name)

    @property
    def vocab_size(self) -> int:
        return self.processor.get_vocab_size()

    @staticmethod
    def train_parser(parser: argparse.ArgumentParser) -> None:
        # Required arguments.
        group = parser.add_argument_group('common arguments')
        group.add_argument(
            '--exp_name',
            help='Name of the tokenizer training experiment.',
            required=True,
            type=str,
        )
        group.add_argument(
            '--max_vocab',
            required=True,
            type=int,
        )
        group.add_argument(
            '--min_count',
            required=True,
            type=int,
        )

        # Optional arguments.
        group.add_argument(
            '--clean_text',
            default=True,
            type=bool,
        )
        group.add_argument(
            '--handle_chinese_chars',
            default=False,
            type=bool,
        )
        group.add_argument(
            '--strip_accents',
            default=None,
            type=Optional[bool],
        )
        group.add_argument(
            '--lowercase',
            default=True,
            type=bool,
        )
        group.add_argument(
            '--wordpieces_prefix',
            default='',
            type=str,
        )
