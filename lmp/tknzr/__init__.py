r""":term:`Tokenizer` module.

All tokenizer classes must be re-imported in this file.

Attributes
==========
ALL_TKNZRS: List[:py:class:`lmp.tknzr.BaseTknzr`]
    All available tokenizers.
    Every time a new tokenizer is added, it must also be added to
    ``ALL_TKNZRS`` list.
TKNZR_OPTS: Final[Dict[str, :py:class:`lmp.tknzr.BaseTknzr`]]
    Mapping from tokenizer's name to tokenizer's class.
    All tokenizers must have class attribute ``tknzr_name``.

Examples
========
Check ``'character'`` is an available tokenizer class.

>>> from lmp.tknzr import TKNZR_OPTS
>>> 'character' in TKNZR_OPTS
True

Get ``'character'`` tokenizer class.

>>> from lmp.tknzr import CharTknzr
>>> TKNZR_OPTS['character'] == CharTknzr
True
"""


from typing import Dict, Final, List, Type

from lmp.tknzr._base import BaseTknzr
from lmp.tknzr._char import CharTknzr
from lmp.tknzr._ws import WsTknzr
from lmp.tknzr._BERT_wordpiece import BertWordPiece

ALL_TKNZRS: Final[List[Type[BaseTknzr]]] = [
    CharTknzr,
    WsTknzr,
]
TKNZR_OPTS: Final[Dict[str, Type[BaseTknzr]]] = {
    t.tknzr_name: t
    for t in ALL_TKNZRS
}
SUBWORD_TKNZR_OPTS = {
    BertWordPiece.tknzr_name: BertWordPiece,
}
