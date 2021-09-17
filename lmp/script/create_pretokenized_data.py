import re
import os
from tqdm import tqdm
from ckip_transformers.nlp import CkipWordSegmenter
from unicodedata import normalize

import lmp.dset
import lmp.path


def main():
    dset = lmp.dset.NewsDataset("news_v2.3_train.db", 250000)
    # print(len(dset))

    # # Initialize drivers
    # ws_driver = CkipWordSegmenter(level=3, device=0)
    # split_pattern = re.compile(r'[，,。：:；;！!？?]')
    # remove_pattern = re.compile(r'(<\w*>)|\s*')
    # path = os.path.join(lmp.path.DATA_PATH, "tokenized_data")
    # if not os.path.isdir(path):
    #     os.mkdir(path)
    # count = 0
    # text = []
    # for title, article in tqdm(dset):
    #     count += 1
    #     temp = re.sub(remove_pattern, "", normalize("NFKC", article))
    #     temp = re.split(split_pattern, temp)
    #     text += list(filter(lambda x: len(x.strip()) != 0, temp))
    #     text += [re.sub(remove_pattern, "", normalize("NFKC", title))]
    #     # print(text)
    #     if count % 50000 == 0:
    #         result = ws_driver(text, max_length=15, batch_size=256)
    #         text = []
    #         idx = count // 50000
    #         with open(os.path.join(path, f"_{idx}.txt"), "w") as f:
    #             result = ["\n".join(x) for x in result]
    #             result = "\n".join(result)
    #             f.write(result)

    split_pattern = re.compile(r'[，,。：:；;！!？?]')
    remove_pattern = re.compile(r'(<\w*>)|\s*')

    count = 0
    text = []
    for title, article in tqdm(dset):
        count += 1
        temp = re.sub(remove_pattern, "", normalize("NFKC", article))
        temp = re.split(split_pattern, temp)
        text += list(filter(lambda x: len(x.strip()) != 0, temp))
        text += [re.sub(remove_pattern, "", normalize("NFKC", title))]

    with open(os.path.join(lmp.path.DATA_PATH, "news_v2.3_raw.txt"), "w") as f:
        result = "\n".join(text)
        f.write(result)


if __name__ == "__main__":
    main()
