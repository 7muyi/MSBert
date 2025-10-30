import json

from msbert.tokenizer import Tokenizer
from msbert.config import TrainConfig


class Collection:
    def __init__(self, path: str) -> None:
        self.data: dict = self._load_file(path)

    def _load_file(self, path: str) -> bool:
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                data[record["id"]] = record["content"]

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data.items())

    def __getitem__(self, key) -> str:
        return self.data[key]


class Records:
    def __init__(self, path: str, n_negative:int) -> None:
        self.n_negative = n_negative
        self.data = self._load_file(path)

    def _load_file(self, path: str) -> list:
        records = []

        with open(path) as f:
            for line in f:
                record = json.loads(line)
                positive = None
                negative = []
                for sample in record[1:]:
                    if sample[1] == 1:  # positive sample
                        if positive == None: positive = sample[0]
                    if sample[1] == 0:  # negativte sample
                        if len(negative) < self.n_negative: negative.append(sample[0])

                    if positive is not None and len(negative) >= self.n_negative:
                        break
                else:
                    continue
                records.append([record[0], positive, *negative])

        return records

    def tolist(self) -> list:
        return list(self.data)


class DataLoader:
    def __init__(self, config: TrainConfig, queries: str, documents: str, records: str) -> None:
        self.bsize = config.bsize
        self.n_negative = config.n_negative
        self.position = 0

        self.tok = Tokenizer(config.tok_config)

        self.queries = Collection(queries)
        self.documents = Collection(documents)
        self.records = Records(records, self.n_negative).tolist()

    def __iter__(self) -> "DataLoader":
        return self

    def __len__(self) -> int:
        return len(self.records)

    def __next__(self):# -> tuple[list | tuple[Tensor, Tensor], tuple[list, Tensor] |...:
        offset, endpos = self.position, min(self.position + self.bsize, len(self.records))
        self.position = endpos

        if self.position >= len(self.records):  # ensure the batch size is fixed at `bsize`.
            raise StopIteration

        queries, documents = [], []
        for position in range(offset, endpos):
            qid, *dids = self.records[position]

            queries.append(self.queries[qid])
            documents.extend(self.documents[did] for did in dids)

        return self.tok.tensorize_qry(queries), self.tok.tensorize_doc(documents)