import numpy as np
import torch
from transformers import BertTokenizer

from .config import TokenizerConfig


class Tokenizer:
    def __init__(self, config: TokenizerConfig) -> None:
        self.tok = BertTokenizer.from_pretrained(config.pretrained_model)
        self.span_size = config.span_size

        self.qry_maxlen = config.qry_maxlen
        self.doc_maxlen = config.doc_maxlen

        # 3 for the special tokens: [CLS], [Q]/[D], and [SEP].
        # add a [S] token after every span_size token, making each group size span_size + 1.
        self.max_qry_tok = (self.qry_maxlen - 3) // (self.span_size + 1) * (self.span_size + 1) + 3
        self.max_doc_tok = (self.doc_maxlen - 3) // (self.span_size + 1) * (self.span_size + 1) + 3

        # special tokens: [Q], [D], [S], [SEP], [CLS]
        self.Q_token = config.qry_token
        self.D_token = config.doc_token
        self.span_token = config.span_token
        self.sep_token = self.tok.sep_token
        self.cls_token = self.tok.cls_token
        self.pad_token = self.tok.pad_token

        # the id corresponding to the special token
        self.Q_token_id = self.tok.convert_tokens_to_ids(config.qry_token_id)  # unused0
        self.D_token_id = self.tok.convert_tokens_to_ids(config.doc_token_id)  # unused1
        self.span_token_id = self.tok.convert_tokens_to_ids(config.span_token_id)  # unused2
        self.sep_token_id = self.tok.sep_token_id
        self.cls_token_id = self.tok.cls_token_id
        self.pad_token_id = self.tok.pad_token_id

    @staticmethod
    def _split_into_batches(ids: torch.Tensor, mask: torch.Tensor, bsize: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        ids_batches = torch.split(ids, bsize, dim=0)
        mask_batches = torch.split(mask, bsize, dim=0)
        return list(zip(ids_batches, mask_batches))

    def tokenize_qry(self, text: str, add_special_tokens: bool = True) -> list[str]:
        """Tokenize the input text into tokens with span tokens inserted.

        Args:
            text (str): The input text to be tokenized.
            add_special_tokens (bool, optional): A flag indicating whether to include the special tokens 
                [CLS], [Q], and [SEP]. Defaults to True.

        Returns:
            list[str]: A list of tokens corresponding to the tokenized input text.

        Notes:
            - The function splits the input text into spans of `self.span_size` tokens each. 
                If the span is smaller than `self.span_size`, padding tokens are added to fill the span.
            - Special tokens [CLS], [Q], and [SEP] are only added if `add_special_tokens` is True.
        """
        # Tokenize the input text into tokens with span tokens inserted.
        tokens_ = self.tok.tokenize(text)

        tokens = []
        if add_special_tokens:
            tokens.extend([self.cls_token, self.Q_token])  # add [CLS] and [Q] tokens

        for i in range(0, len(tokens_), self.span_size):
            if len(tokens) + (self.span_size + 1) + 1 > self.max_qry_tok - (0 if add_special_tokens else 2):
                break

            span_tokens = tokens_[i:i+self.span_size]  # add `span_size` tokens
            tokens.extend(span_tokens)

            if len(span_tokens) < self.span_size:  # padding to span_size if necessary
                tokens.extend([self.pad_token for _ in range(self.span_size - len(span_tokens))])

            tokens.append(self.span_token)  # add [S] token at the end of each span

        if add_special_tokens:
            tokens.append(self.sep_token)

        return tokens

    def encode_qry(self, text: str, add_special_tokens: bool = True) -> list[str]:
        # Tokenize the input text into tokens with span tokens inserted.
        tokens = self.tok.tokenize(text)

        ids = []
        if add_special_tokens:
            ids.extend([self.cls_token_id, self.Q_token_id])  # add [CLS] and [Q] tokens

        for i in range(0, len(tokens), self.span_size):
            if len(ids) + (self.span_size + 1) + 1 > self.max_qry_tok - (0 if add_special_tokens else 2):
                break

            span_ids = self.tok.convert_tokens_to_ids(tokens[i:i+self.span_size])  # add `span_size` tokens
            ids.extend(span_ids)

            if len(span_ids) < self.span_size:  # padding to span_size if necessary
                ids.extend([self.pad_token_id for _ in range(self.span_size - len(span_ids))])

            ids.append(self.span_token_id)  # add [S] token at the end of each span

        if add_special_tokens:
            ids.append(self.sep_token_id)

        return ids

    def tensorize_qry(self, batch_text: str | list[str], bsize: int | None = None):
        if isinstance(batch_text, str):
            batch_text = [batch_text]

        all_tok_ids = []
        max_length = -1
        for text in batch_text:
            tok_ids = self.encode_qry(text, False)
            all_tok_ids.append(tok_ids)
            max_length = max(max_length, len(tok_ids))

        # initialize
        # 3 for [CLS], [Q], and [SEP]
        ids = np.full((len(all_tok_ids), max_length + 3), self.pad_token_id, dtype=np.int64)
        masks = np.zeros_like(ids, dtype=np.int64)
        masks[:, 2::self.span_size + 1] = 1  # set mask for non-padding tokens [S]

        # set sepcial tokens [CLS] and [Q]
        ids[:, 0] = self.cls_token_id
        ids[:, 1] = self.Q_token_id

        # populate data
        for i, tok_ids in enumerate(all_tok_ids):
            seq_len = len(tok_ids)
            ids[i, 2:2+seq_len] = tok_ids
            ids[i, 2 + seq_len] = self.sep_token_id  # add [SEP] token
            masks[i, :2 + seq_len + 1] = 1

        # convert to Tensor at Once
        ids = torch.from_numpy(ids)
        masks = torch.from_numpy(masks)

        if bsize:
            return self._split_into_batches(ids, masks, bsize)

        return ids, masks

    def tokenize_doc(self, text: str, add_special_tokens: bool = True) -> list[str]:
        """similar to the tokenize_qry method but for documents"""
        # Tokenize the input text into tokens with span tokens inserted.
        tokens_ = self.tok.tokenize(text)

        tokens = []
        if add_special_tokens:
            tokens.extend([self.cls_token, self.D_token])  # add [CLS] and [D] tokens

        for i in range(0, len(tokens_), self.span_size):
            if len(tokens) + (self.span_size + 1) + 1 > self.max_doc_tok - (0 if add_special_tokens else 2):
                break

            span_tokens = tokens_[i:i+self.span_size]  # add `span_size` tokens
            tokens.extend(span_tokens)

            if len(span_tokens) < self.span_size:  # padding to span_size if necessary
                tokens.extend([self.pad_token for _ in range(self.span_size - len(span_tokens))])

            tokens.append(self.span_token)  # add [S] token at the end of each span

        if add_special_tokens:
            tokens.append(self.sep_token)

        return tokens

    def encode_doc(self, text: str, add_special_tokens: bool = True) -> list[str]:
        """similar to the encode_qry method but for documents"""
        # Tokenize the input text into tokens with span tokens inserted.
        tokens = self.tok.tokenize(text)

        # pre allocate list size to reduce dymamic resizing
        estimated_size = (len(tokens) + self.span_size - 1) // self.span_size  * (self.span_size + 1)
        ids = []
        ids.reserve(estimated_size) if hasattr(list, "reserve") else None

        if add_special_tokens:
            ids.extend([self.cls_token_id, self.D_token_id])  # add [CLS] and [D] tokens

        for i in range(0, len(tokens), self.span_size):
            if len(ids) + (self.span_size + 1) + 1 > self.max_doc_tok - (0 if add_special_tokens else 2):
                break

            span_ids = self.tok.convert_tokens_to_ids(tokens[i:i+self.span_size])  # add `span_size` tokens
            ids.extend(span_ids)

            if len(span_ids) < self.span_size:  # padding to span_size if necessary
                ids.extend([self.pad_token_id for _ in range(self.span_size - len(span_ids))])

            ids.append(self.span_token_id)  # add [S] token at the end of each span

        if add_special_tokens:
            ids.append(self.sep_token_id)

        return ids

    def tensorize_doc(self, batch_text: str | list[str], bsize: int | None = None):
        """similar to the tensorize_qry method but for documents"""
        if isinstance(batch_text, str):
            batch_text = [batch_text]

        all_tok_ids = []
        max_length = -1
        for text in batch_text:
            tok_ids = self.encode_doc(text, False)
            all_tok_ids.append(tok_ids)
            max_length = max(max_length, len(tok_ids))

        # initialize
        # 3 for [CLS], [D], and [SEP]
        ids = np.full((len(all_tok_ids), max_length + 3), self.pad_token_id, dtype=np.int64)
        masks = np.zeros_like(ids, dtype=np.int64)

        # set sepcial tokens [CLS] and [D]
        ids[:, 0] = self.cls_token_id
        ids[:, 1] = self.D_token_id

        # populate data
        for i, tok_ids in enumerate(all_tok_ids):
            seq_len = len(tok_ids)
            ids[i, 2:2+seq_len] = tok_ids
            ids[i, 2 + seq_len] = self.sep_token_id  # add [SEP] token
            masks[i, :2 + seq_len + 1] = 1

        # convert to Tensor at Once
        ids = torch.from_numpy(ids)
        masks = torch.from_numpy(masks)

        if bsize:
            return self._split_into_batches(ids, masks, bsize)

        return ids, masks