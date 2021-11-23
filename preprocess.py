import bisect
import math
import textwrap
import warnings
from collections import defaultdict
from itertools import chain
from math import ceil

import networkx as nx
import numpy as np
import pandas as pd
import regex
import termcolor
import torch

from data import label_mapping, enum_outputs, abbreviations, value_labels, ner_label_mapping
from nlstruct import BRATDataset, copy
from nlstruct.data_utils import mappable, huggingface_tokenize, regex_tokenize, split_spans, regex_sentencize, OverlappingEntityException, loop, mix
from nlstruct.datasets.base import BaseDataset
from nlstruct.models.common import register
from nlstruct.models.ner import NERPreprocessor
from nlstruct.torch_utils import list_factorize, batch_to_tensors, seed_all
from tools import *


def compute_token_slice_indices(tokens, begin, end):
    index_after_first_token_begin = bisect.bisect_left(tokens["begin"], begin)
    index_before_first_token_end = bisect.bisect_right(tokens["end"], begin)
    index_after_last_token_begin = bisect.bisect_left(tokens["begin"], end)
    index_before_last_token_end = bisect.bisect_right(tokens["end"], end)
    begin_indice = min(index_after_first_token_begin, index_before_first_token_end)
    end_indice = max(index_after_last_token_begin, index_before_last_token_end)

    return begin_indice, end_indice


def slice_tokenization_output(tokens, begin, end, insert_before=None, insert_after=None):
    begin_indice, end_indice = compute_token_slice_indices(tokens, begin, end)
    begins = np.asarray(([begin] if insert_before is not None else []) + list(tokens["begin"][begin_indice:end_indice]) + ([end] if insert_after is not None else []))
    ends = np.asarray(([begin] if insert_before is not None else []) + list(tokens["end"][begin_indice:end_indice]) + ([end] if insert_after is not None else []))

    return {
        "begin": begins,
        "end": ends,
        "text": ([insert_before] if insert_before is not None else []) + list(tokens["text"][begin_indice:end_indice]) + ([insert_after] if insert_after is not None else []),
        **{key: m[begin_indice:end_indice] for key, m in tokens.items() if key not in ("begin", "end", "text")},
    }


def slice_document(doc, begin, end, only_text=False, overlap_policy='raise', main_fragment_label=None, offset_spans=True):
    assert overlap_policy in ("raise", "split")
    absolute_begin = doc.get("begin", 0)
    sentence_size = end - begin
    new_entities = []
    offset = begin if offset_spans else 0

    doc_new_fragments = []
    for fragment in doc["fragments"]:
        if fragment["begin"] < end and begin < fragment["end"]:
            if not (begin <= fragment["begin"] and fragment["end"] <= end):
                if overlap_policy == "raise":
                    raise OverlappingEntityException(
                        "Fragment {} spans more than one sentence in document {}. "
                        "Use sentence_overlap_policy='split' in preprocessor to handle such cases.".format(
                            repr(doc["text"][fragment["begin"]:fragment["end"]]), doc["doc_id"]))
            doc_new_fragments.append({
                **fragment,
                "begin": min(max(fragment["begin"] - offset, 0), sentence_size),
                "end": max(min(fragment["end"] - offset, sentence_size), 0)
            })

    if "entities" in doc and not only_text:
        for entity in doc["entities"]:
            new_chunks = []
            for chunk in entity["chunks"]:
                min_begin = min(fragment["begin"] for fragment in chunk["fragments"])
                max_end = max(fragment["end"] for fragment in chunk["fragments"])
                if min_begin < end and begin < max_end:
                    if begin <= min_begin and max_end <= end:
                        new_chunks.append({**chunk, "fragments": [
                            {**fragment,
                             "begin": fragment["begin"] - offset,
                             "end": fragment["end"] - offset}
                            for fragment in chunk["fragments"]]})
                    elif any(fragment["label"].startswith("entity_type") for fragment in chunk["fragments"]):
                        if overlap_policy == "raise":
                            raise OverlappingEntityException(
                                "Chunk {} ({} -> {}) spans more than one sentence in document {}. "
                                "Use sentence_overlap_policy='split' in preprocessor to handle such cases.".format(
                                    repr(doc["text"][min_begin:max_end]), min_begin, max_end, doc["doc_id"]))
                        else:
                            new_fragments = [{**fragment,
                                              "begin": min(max(fragment["begin"] - offset, 0), sentence_size),
                                              "end": max(min(fragment["end"] - offset, sentence_size), 0)}
                                             for fragment in chunk["fragments"]
                                             if fragment["begin"] < end and begin < fragment["end"]]
                            if len(new_fragments) and main_fragment_label is None or any(f.get("label", "main") == main_fragment_label for f in new_fragments):
                                new_chunks.append({**chunk, "fragments": new_fragments})
            if len(new_chunks):
                new_entities.append({**entity, "chunks": new_chunks})
    new_doc = {
        **doc,
        "doc_id": doc["doc_id"] + "/{}-{}".format(absolute_begin + begin, absolute_begin + end),
        "text": doc["text"][begin:end],
        "begin": absolute_begin + begin,
        "end": absolute_begin + end,
        "entities": new_entities,
        "fragments": doc_new_fragments,
    }
    new_doc = relink_doc_items(new_doc)

    return new_doc


class LargeSentenceException(Exception):
    pass


@register("frame_preprocessor")
class FramePreprocessor(NERPreprocessor):
    memory_model_terms = (
        "n_samples",
        "n_sentences",
        "n_words",
        "n_samples * max_words * max_fragments",
        "n_samples * (max_fragments ** 2)",
        "n_sentences * max_wordpieces",
        "n_sentences * (max_wordpieces ** 2)",
        "n_samples * max_words",
    )

    def get_memory_model_features(self, doc):
        n_sentences = len(doc['tokens'])
        n_words = len(doc["words_mask"])
        n_wordpieces = max(map(len, doc["tokens"]))
        shapes = {
            # "n_full_samples": 1 if doc["has_chunks"] else 0,
            "n_samples": 1,
            "n_sentences": n_sentences,
            "n_words": n_words,
            "max_wordpieces": n_wordpieces,
            "max_words": n_words,
        }
        if "fragments_mask" in doc:
            shapes["max_fragments"] = len(doc["fragments_mask"])
        return shapes

    @mappable
    def forward(self, doc, only_text=False):
        self.last_doc = doc
        results = []
        for sample, tokenized_sample, tokenized_sentences in self.sentencize_and_tokenize(doc, only_text=only_text):
            # Here, we know that the sentence is not too long
            if "char" in self.vocabularies:
                words_chars = [[self.vocabularies["char"].get(char) for char in word]
                               for word, word_bert_begin in zip(tokenized_sample["words_text"],
                                                                tokenized_sample["words_bert_begin"]) if word_bert_begin != -1]
            else:
                words_chars = None

            if "word" in self.vocabularies:
                word_voc = self.vocabularies["word"]
                words = [word_voc.get(word) for word, word_bert_begin in zip(
                    tokenized_sample["words_text"],
                    tokenized_sample["words_bert_begin"]) if word_bert_begin != -1]
            else:
                words = None
            fragments_begin = []
            fragments_end = []
            fragments_label = []
            fragments_ner_label = []
            fragments_id = []
            fragments_chunks = []
            chunks_fragments = []
            chunks_labels = []
            chunks_allowed_labels = []
            chunks_entities = []
            entities_chunks = []

            tags = None
            if not only_text and "entities" in sample:
                fragments_dict = dict()

                # entities = [
                #    entity for entity in sample["entities"]
                #    if not self.filter_entities or len(set(entity["label"]) & set(self.filter_entities)) > 0
                # ]
                entities = []

                fragments = {(f["begin"], f["end"], f["label"]): f for f in sample["fragments"]}
                fragments = {key: (i, fragment) for i, (key, fragment) in enumerate(sorted(fragments.items(), key=lambda t: t[0]))}

                changed = True
                fragments = [
                    {
                        "begin": min(sample["fragments"][i]["begin"] for i in group),
                        "end": max(sample["fragments"][i]["end"] for i in group),
                        "label": tuple(sorted({sample["fragments"][i]["label"] for i in group})),
                        "fragment_ids": sorted({sample["fragments"][i]["fragment_id"] for i in group}),
                        "ner_label": ner_label_mapping[sample["fragments"][next(iter(group))]["label"]],
                    }
                    for group in (nx.connected_components(nx.from_numpy_array(np.asarray([
                        [not (a["begin"] >= b["end"] or b["begin"] >= a["end"]) and ner_label_mapping[a["label"]] == ner_label_mapping[b["label"]]
                         for b in sample["fragments"]]
                        for a in sample["fragments"]
                    ]).reshape(len(sample["fragments"]), len(sample["fragments"])))))
                ]
                # fragments = {(f["begin"], f["end"], f["label"]): f for f in fragments}
                # fragments = {key: (i, fragment) for i, (key, fragment) in enumerate(sorted(fragments.items(), key=lambda t: t[0]))}

                for fragment in fragments:
                    fragments_begin.append(fragment["begin"])
                    fragments_end.append(fragment["end"])
                    fragments_label.append(fragment["label"])
                    fragments_ner_label.append(fragment["ner_label"])
                    fragments_chunks.append([])

                for entity in sample["entities"]:
                    entity_idx = len(entities_chunks)
                    entities_chunks.append([])
                    for chunk in entity["chunks"]:
                        chunk_idx = len(chunks_fragments)
                        chunks_entities.append([entity_idx])  # TODO what if a chunk has multiple entities
                        entities_chunks[-1].append(chunk_idx)
                        chunks_labels.append([label in chunk["labels"] for label in self.vocabularies["entity_label"].values])
                        chunks_allowed_labels.append([label in chunk["complete_labels"] for label in self.vocabularies["entity_label"].values])
                        chunks_fragments.append([])
                        for original_fragment in chunk["fragments"]:
                            fragment_idx = next(i for i, f in enumerate(fragments) if original_fragment["fragment_id"] in f["fragment_ids"])
                            if chunk_idx not in fragments_chunks[fragment_idx]:
                                fragments_chunks[fragment_idx].append(chunk_idx)
                            if fragment_idx not in chunks_fragments[-1]:
                                chunks_fragments[-1].append(fragment_idx)

                sorter = sorted(range(len(fragments_begin)), key=lambda i: (fragments_begin[i], fragments_end[i], fragments_label[i]))
                fragments_begin = [fragments_begin[i] for i in sorter]
                fragments_end = [fragments_end[i] for i in sorter]
                fragments_label = [fragments_label[i] for i in sorter]
                fragments_ner_label = [fragments_ner_label[i] for i in sorter]
                fragments_chunks = [fragments_chunks[i] for i in sorter]

                fragments_begin, fragments_end = split_spans(fragments_begin, fragments_end, tokenized_sample["words_begin"], tokenized_sample["words_end"])
                empty_fragment_idx = next((i for i, begin in enumerate(fragments_begin) if begin == -1), None)
                if empty_fragment_idx is not None:
                    if self.empty_entities == "raise":
                        raise Exception(
                            f"Entity in {sample['doc_id']} could not be matched with any word"
                            f" (is it empty or outside the text ?). Use empty_entities='drop' to ignore these cases")
                    else:
                        warnings.warn("Empty fragments (start = end or outside the text) have been skipped")
                        fragments_label = [label for label, begin in zip(fragments_label, fragments_begin) if begin != -1]
                        fragments_ner_label = [label for label, begin in zip(fragments_ner_label, fragments_begin) if begin != -1]
                        fragments_end = np.asarray([end for end, begin in zip(fragments_end, fragments_begin) if begin != -1])
                        fragments_entities = np.asarray([e for e, begin in zip(fragments_entities, fragments_begin) if begin != -1])
                        fragments_begin = np.asarray([begin for begin in fragments_begin if begin != -1])

                chunks_fragments = list_factorize(chunks_fragments, sorter)[0]
                fragments_end -= 1  # end now means the index of the last word

                # One hot encoding
                fragments_label = [[label in fragment_label for label in self.vocabularies["entity_label"].values]
                                   for fragment_label in fragments_label]

                # Multiclass encoding
                fragments_ner_label = [self.vocabularies["fragment_label"].get(label) for label in fragments_ner_label]
                fragments_begin, fragments_end = fragments_begin.tolist(), fragments_end.tolist()

            if len(chunks_labels) == 0:
                entities_chunks = [[]]
                chunks_entities = [[]]
                chunks_labels = [[False] * len(self.vocabularies["entity_label"].values)]
                chunks_allowed_labels = [[False] * len(self.vocabularies["entity_label"].values)]
                chunks_fragments = [[]]
                chunks_mask = [False]
                entities_mask = [False]
            else:
                chunks_mask = [True] * len(chunks_fragments)
                entities_mask = [False] * len(entities_chunks)
            if len(fragments_label) == 0:
                fragments_begin = [0]
                fragments_end = [0]
                fragments_label = [[False] * len(self.vocabularies["entity_label"])]
                fragments_ner_label = [0]
                fragments_chunks = [[]]
                fragments_mask = [False]
            else:
                fragments_mask = [True] * len(fragments_label)
            # if len(tokens_indice) > self.max_tokens:
            results.append({
                "tokens": tokenized_sentences["bert_tokens_indice"],
                **({
                       "slice_begin": tokenized_sentences["slice_begin"],
                       "slice_end": tokenized_sentences["slice_end"],
                   } if "slice_begin" in tokenized_sentences else {}),

                "has_chunks": not sample.get("is_synthetic", False),
                "tokens": tokenized_sentences["bert_tokens_indice"],
                "tokens_mask": tokenized_sentences["bert_tokens_mask"],
                "sentence_mask": [True] * len(tokenized_sentences["bert_tokens_indice"]),
                "words": words,
                "words_mask": [True] * len(tokenized_sample["words_text"]),
                "words_text": tokenized_sample["words_text"],
                "words_chars_mask": [[True] * len(word_chars) for word_chars in words_chars] if words_chars is not None else None,
                "words_bert_begin": tokenized_sample["words_bert_begin"].tolist(),
                "words_bert_end": tokenized_sample["words_bert_end"].tolist(),
                "words_begin": tokenized_sample["words_begin"].tolist(),
                "words_end": tokenized_sample["words_end"].tolist(),
                "words_sentence_idx": tokenized_sample["words_sentence_idx"].astype(int).tolist(),
                "words_chars": words_chars,

                "fragments_begin": fragments_begin,
                "fragments_end": fragments_end,
                "fragments_label": fragments_label,
                "fragments_ner_label": fragments_ner_label,
                "fragments_mask": fragments_mask,
                "fragments_chunks": fragments_chunks,

                "chunks_fragments": chunks_fragments,
                "chunks_labels": chunks_labels,
                "chunks_mask": chunks_mask,
                "chunks_allowed_labels": chunks_allowed_labels,
                "chunks_entities": chunks_entities,

                "entities_chunks": entities_chunks,
                "entities_mask": entities_mask,

                "doc_id": sample["doc_id"],
                "original_sample": sample,
                "original_doc": doc,
            })
        return results

    def train(self, mode=True):
        self.training = mode

    def empty_cache(self):
        self.bert_tokenizer_cache = {}
        self.regex_tokenizer_cache = {}

    def tensorize(self, batch, device=None):
        tensors = batch_to_tensors(
            batch,
            dtypes={
                "tokens": torch.long,
                "tokens_mask": torch.bool,
                "sentence_mask": torch.bool,
                "words_mask": torch.bool,
                "words_chars": torch.long,
                "words_chars_mask": torch.bool,
                "words_bert_begin": torch.long,
                "words_bert_end": torch.long,
                "words_begin": torch.long,
                "words_end": torch.long,
                "words_sentence_idx": torch.long,
                "words_chars": torch.long,
                "fragments_begin": torch.long,
                "fragments_end": torch.long,
                "fragments_label": torch.bool,
                "fragments_ner_label": torch.long,
                "fragments_mask": torch.bool,
                "fragments_chunks": torch.long,
                "chunks_fragments": torch.long,
                "chunks_labels": torch.bool,
                "chunks_mask": torch.bool,
                "chunks_allowed_labels": torch.bool,
                "chunks_entities": torch.long,
                "entities_chunks": torch.long,
                "entities_mask": torch.bool,
            },
            pad={
                "tokens": -1,
                "sentence_mask": False,
                "words_mask": False,
                "words_bert_begin": 0,
                "words_bert_end": 0,
                "words_begin": 0,
                "words_end": 0,
                "words_sentence_idx": 0,
                "words_chars": -1,
                "fragments_begin": 0,
                "fragments_end": 0,
                "fragments_ner_label": 0,
                "fragments_label": False,
                "fragments_mask": False,
                "fragments_chunks": -1,
                "chunks_fragments": -1,
                "chunks_labels": False,
                "chunks_allowed_labels": False,
                "chunks_entities": -1,
                "entities_chunks": -1,
                "entities_mask": False,
            }, device=device)
        tensors["tokens_mask"] = tensors["tokens"] != -1
        tensors["tokens"].clamp_min_(0)
        if tensors.get("words_chars", None) is not None:
            tensors["words_chars_mask"] = tensors["words_chars"] != -1
            tensors["words_chars"].clamp_min_(0)
        return tensors

    def sentencize_and_tokenize(self, doc, only_text=False):
        text = doc["text"]

        if self.sentence_split_regex is not None:
            sentences_bounds = list(regex_sentencize(text, reg_split=self.sentence_split_regex, balance_chars=self.sentence_balance_chars))
        else:
            sentences_bounds = [(0, len(text))]

        if self.tokenizer is not None:
            if not self.training or text not in self.bert_tokenizer_cache:
                full_doc_bert_tokens = huggingface_tokenize(text.lower() if self.bert_lower else text,
                                                            tokenizer=self.tokenizer,
                                                            subs=self.substitutions,
                                                            do_unidecode=self.do_unidecode,
                                                            return_offsets_mapping=True,
                                                            add_special_tokens=False)
                if self.word_regex is None:
                    full_doc_bert_tokens["sentence_idx"] = [0] * len(full_doc_bert_tokens["begin"])
                    for sentence_idx, (begin, end) in enumerate(sentences_bounds):
                        sentence_begin_idx = compute_token_slice_indices(full_doc_bert_tokens, begin, end)[0]
                        full_doc_bert_tokens["sentence_idx"][sentence_begin_idx:] = [sentence_idx] * (len(full_doc_bert_tokens["sentence_idx"]) - sentence_begin_idx)
                if self.training:
                    self.bert_tokenizer_cache[text] = full_doc_bert_tokens
            else:
                full_doc_bert_tokens = self.bert_tokenizer_cache[text]
        else:
            full_doc_bert_tokens = None
        if self.word_regex is not None:
            if not self.training or text not in self.regex_tokenizer_cache:
                full_doc_words = regex_tokenize(text,
                                                reg=self.word_regex,
                                                subs=self.substitutions,
                                                do_unidecode=self.do_unidecode,
                                                return_offsets_mapping=True, )
                full_doc_words["sentence_idx"] = [0] * len(full_doc_words["begin"])
                for sentence_idx, (begin, end) in enumerate(sentences_bounds):
                    sentence_begin_idx = compute_token_slice_indices(full_doc_words, begin, end)[0]
                    full_doc_words["sentence_idx"][sentence_begin_idx:] = [sentence_idx] * (len(full_doc_words["sentence_idx"]) - sentence_begin_idx)
                if self.training:
                    self.regex_tokenizer_cache[text] = full_doc_words
            else:
                full_doc_words = self.regex_tokenizer_cache[text]
        else:
            full_doc_words = full_doc_bert_tokens
        if full_doc_bert_tokens is None:
            full_doc_bert_tokens = full_doc_words

        if self.split_into_multiple_samples:
            results = []
        else:
            results = [(
                doc, {
                    "words_begin": np.asarray([], dtype=int),
                    "words_end": np.asarray([], dtype=int),
                    "words_bert_begin": np.asarray([], dtype=int),
                    "words_bert_end": np.asarray([], dtype=int),
                    "words_text": [],
                    "words_sentence_idx": [],
                }, {
                    "bert_tokens_text": [],
                    "bert_tokens_begin": [],
                    "bert_tokens_end": [],
                    "bert_tokens_indice": [],
                    "bert_tokens_mask": [],
                    **({
                           "slice_begin": [],
                           "slice_end": [],
                       } if self.doc_context else {})
                })]
        bert_offset = 0
        begin = None
        while len(sentences_bounds):
            new_begin, end = sentences_bounds.pop(0)
            if begin is None:
                begin = new_begin

            sentence_text = text[begin:end]
            if not sentence_text.strip():
                continue

            bert_tokens = slice_tokenization_output(full_doc_bert_tokens, begin, end,
                                                    (getattr(self.tokenizer, '_bos_token', None) or self.tokenizer.special_tokens_map.get('cls_token', None)) if self.tokenizer is not None else None,
                                                    (getattr(self.tokenizer, '_eos_token', None) or self.tokenizer.special_tokens_map.get('sep_token', None)) if self.tokenizer is not None else None)
            if (
                  (self.min_tokens is not None and len(bert_tokens["text"]) < self.min_tokens) or
                  (self.max_tokens is not None and len(bert_tokens["text"]) < self.max_tokens and (
                        self.join_small_sentence_rate > 0.5 and (not self.training or random.random() < self.join_small_sentence_rate)))
            ) and len(sentences_bounds):
                if len(bert_tokens["text"]) + len(slice_tokenization_output(full_doc_bert_tokens, *sentences_bounds[0])["text"]) + 2 < self.max_tokens:
                    continue

            words = slice_tokenization_output(full_doc_words, begin, end, '' if self.keep_bert_special_tokens else None, '' if self.keep_bert_special_tokens else None)

            tokens_indice = self.tokenizer.convert_tokens_to_ids(bert_tokens["text"]) if self.tokenizer is not None else None
            words_bert_begin, words_bert_end = split_spans(words["begin"], words["end"], bert_tokens["begin"], bert_tokens["end"])
            words = {
                key: [i for i, j in zip(value, words_bert_begin) if j != -1] if isinstance(value, list) else value[words_bert_begin != -1]
                for key, value in words.items()
            }
            words_bert_end = words_bert_end[words_bert_begin != -1]
            words_bert_begin = words_bert_begin[words_bert_begin != -1]
            # words_bert_begin, words_bert_end = words_bert_begin.tolist(), words_bert_end.tolist()

            # if the sentence has too many tokens, split it
            if len(bert_tokens['text']) > self.max_tokens:
                warnings.warn(f'A sentence of more than {self.max_tokens} wordpieces will be split. Consider using a more restrictive regex for sentence splitting if you want to avoid it.')
                if self.large_sentences == "equal-split":
                    stop_bert_token = max(len(bert_tokens['text']) // ceil(len(bert_tokens['text']) / self.max_tokens), self.min_tokens)
                elif self.large_sentences == "max-split":
                    stop_bert_token = self.max_tokens
                else:
                    raise LargeSentenceException(repr(sentence_text))
                last_word = next(i for i in range(len(words_bert_end) - 1) if words_bert_end[i + 1] >= stop_bert_token)
                sentences_bounds[:0] = [(begin, words["end"][last_word]), (words["begin"][last_word + 1], end)]
                begin = None
                continue
                # else:
                #    print(len(bert_tokens["text"]) + len(slice_tokenization_output(full_doc_bert_tokens, *sentences_bounds[0])["text"]))
            if not self.split_into_multiple_samples:
                # words["begin"] += begin
                # words["end"] += begin
                # if bert_tokens is not words:
                #     bert_tokens["begin"] += begin
                #     bert_tokens["end"] += begin
                words_bert_begin += bert_offset
                words_bert_end += bert_offset
            bert_offset += len(bert_tokens["text"])
            if self.split_into_multiple_samples:
                results.append((
                    slice_document(
                        doc,
                        begin,
                        end,
                        entity_overlap=self.sentence_entity_overlap,
                        only_text=only_text,
                        main_fragment_label="main",
                        offset_spans=True,
                    ),
                    {
                        "words_bert_begin": words_bert_begin,
                        "words_bert_end": words_bert_end,
                        "words_text": words["text"],
                        "words_begin": words["begin"] - begin,
                        "words_end": words["end"] - begin,
                        "words_sentence_idx": words["sentence_idx"] - min(words["sentence_idx"]),
                    },
                    {
                        "bert_tokens_text": [bert_tokens["text"]],
                        "bert_tokens_begin": [bert_tokens["begin"]],
                        "bert_tokens_end": [bert_tokens["end"]],
                        "bert_tokens_indice": [tokens_indice],
                        **({
                               "slice_begin": [0],
                               "slice_end": [len(tokens_indice)],
                           } if self.doc_context else {})
                    }
                ))
            else:
                results[0][1]["words_text"] += words["text"]
                # numpy arrays
                results[0][1]["words_sentence_idx"] = np.concatenate([results[0][1]["words_sentence_idx"], words["sentence_idx"]])
                results[0][1]["words_begin"] = np.concatenate([results[0][1]["words_begin"], words["begin"]])
                results[0][1]["words_end"] = np.concatenate([results[0][1]["words_end"], words["end"]])
                results[0][1]["words_bert_begin"] = np.concatenate([results[0][1]["words_bert_begin"], words_bert_begin])
                results[0][1]["words_bert_end"] = np.concatenate([results[0][1]["words_bert_end"], words_bert_end])

                results[0][2]["bert_tokens_text"].append(bert_tokens["text"])
                results[0][2]["bert_tokens_begin"].append(bert_tokens["begin"])
                results[0][2]["bert_tokens_end"].append(bert_tokens["end"])
                if self.doc_context:
                    results[0][2]["slice_begin"].append(0)
                    results[0][2]["slice_end"].append(len(tokens_indice))
                results[0][2]["bert_tokens_indice"].append(tokens_indice)

            begin = None
        sentences = [(sent, list(sent[1:-1]), rid, sid) for rid, result in enumerate(results) for sid, sent in enumerate(result[2]['bert_tokens_indice'])]
        if self.doc_context:
            for i, (sentence_i, _, rid, sid) in enumerate(sentences):
                final_complete_mode = False
                sentences_before = sentences[:i]
                sentences_after = sentences[i + 1:]
                added_before = 0
                added_after = 0

                while True:
                    if len(sentences_before) and len(sentences_before[-1][1]) + len(sentence_i) > self.max_tokens:
                        sentences_before = []
                    if len(sentences_after) and len(sentences_after[0][1]) + len(sentence_i) > self.max_tokens:
                        sentences_after = [sentences_after[0]]

                    if len(sentences_before) + len(sentences_after) == 0:
                        break

                    if len(sentences_after) == 0:
                        sent = sentences_before.pop(-1)[1]
                        way = "before"
                    elif len(sentences_before) == 0:
                        sent = sentences_after.pop(0)[1]
                        way = "after"
                    else:
                        if added_before <= added_after:
                            way = "before"
                            sent = sentences_before.pop(-1)[1]
                        else:
                            way = "after"
                            sent = sentences_after.pop(0)[1]

                    if way == "before":
                        sentence_i[1:1] = sent
                        added_before += len(sent)
                    else:
                        if len(sent) + len(sentence_i) > self.max_tokens:
                            sent = sent[:self.max_tokens - len(sentence_i)]
                        sentence_i[-1:-1] = sent
                        added_after += len(sent)

                results[rid][2]['slice_begin'][sid] += added_before
                results[rid][2]['slice_end'][sid] += added_before

        return results

    def decode(self, predictions, prep, group_by_document=True):
        pad_tensor = None
        docs = []
        last_doc_id = None
        for doc_prep, doc_predictions in zip(prep, predictions):
            doc_id = doc_prep["doc_id"].rsplit("/", 1)[0]
            if group_by_document:
                text = doc_prep["original_doc"]["text"]
                char_offset = doc_prep["original_sample"].get("begin", 0)
                if doc_id != last_doc_id:
                    last_doc_id = doc_id
                    docs.append({
                        "doc_id": doc_id,
                        "text": doc_prep["original_doc"]["text"],
                        "entities": [],
                        "fragments": [],
                    })
            else:
                char_offset = 0
                text = doc_prep["original_sample"]["text"]
                docs.append({
                    "doc_id": doc_prep["doc_id"],
                    "text": text,
                    "entities": [],
                    "fragments": [],
                })
            for entity in doc_predictions["entities"]:
                res_entity = {"entity_id": entity["entity_id"], "chunks": [], "labels": []}
                for chunk in entity["chunks"]:
                    res_entity["labels"].append([self.vocabularies['entity_label'].values[l] for l in chunk["label"]])
                    res_entity["chunks"].append({
                        "labels": [self.vocabularies['entity_label'].values[l] for l in chunk["label"]],
                        "chunk_id": chunk["chunk_id"],
                        "fragments": [{
                            "fragment_id": f["fragment_id"],
                            "begin": char_offset + doc_prep["words_begin"][f["begin"]],
                            "end": char_offset + doc_prep["words_end"][f["end"]],
                            "label": self.vocabularies['entity_label'].values[f["label"]],
                            "text": text[char_offset + doc_prep["words_begin"][f["begin"]]:char_offset + doc_prep["words_end"][f["end"]]],
                        } for f in chunk["fragments"]],
                        "confidence": chunk["confidence"],
                    })
                docs[-1]["entities"].append(res_entity)
            for fragment in doc_predictions["fragments"]:
                docs[-1]["fragments"].append({
                    "begin": char_offset + doc_prep["words_begin"][fragment["begin"]],
                    "end": char_offset + doc_prep["words_end"][fragment["end"]],
                    "label": self.vocabularies['entity_label'].values[fragment["label"]],
                    "text": text[char_offset + doc_prep["words_begin"][fragment["begin"]]:char_offset + doc_prep["words_end"][fragment["end"]]],
                    "confidence": fragment["confidence"],
                })
        return docs


def path_extent(path):
    begin = min(fragment["begin"] for fragment in path)
    end = max(fragment["end"] for fragment in path)
    avg = sum((fragment["end"] + fragment["begin"]) / 2. for fragment in path) / len(path)
    return (begin, avg, end)


def extract_entities(doc):
    eid_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12}
    fragments = {}
    for e in doc["entities"]:
        for fidx, fragment in enumerate(e["fragments"]):
            fragment_id = e["entity_id"] if fidx == 0 else (e["entity_id"] + "/" + str(fidx))
            f = {
                "fragment_id": fragment_id,
                "begin": fragment["begin"],
                "end": fragment["end"],
                "label": e["label"],
                "attributes": [attribute["label"] for attribute in e["attributes"] if not attribute["label"].startswith("id") and not attribute["label"].startswith("scope")],
                "to": [],
                "from": [],
                "contains": [],
                "chunk_ids": [int(attribute["label"][2:]) for attribute in e["attributes"] if attribute["label"].startswith("id")],
                "entity_ids": [eid_map[attribute["label"][5:]] for attribute in e["attributes"] if attribute["label"].startswith("scope")]
            }
            fragments[fragment_id] = f
            if fidx > 0:
                fragments[e["entity_id"]]["contains"].append(f)

    for i, fragment_i in fragments.items():
        for j, fragment_j in fragments.items():
            if fragment_i["begin"] <= fragment_j["begin"] and fragment_j["end"] <= fragment_i["end"]:
                if i != j:
                    fragment_i["contains"].append(fragment_j)

    for relation in doc["relations"]:
        fragments[relation["from_entity_id"]]["to"].append({"label": relation["relation_label"], "fragment": fragments[relation["to_entity_id"]]})
        fragments[relation["to_entity_id"]]["from"].append({"label": relation["relation_label"], "fragment": fragments[relation["from_entity_id"]]})

    queue = []
    paths = []
    chunks = defaultdict(lambda: [])

    for head in fragments.values():
        if len([r for r in head["from"] if r["label"] == "entity"]) == 0:
            if head["label"].startswith("entity_type") or len(head["chunk_ids"]) > 0 or not any(fragment["label"].startswith("entity_type") for fragment in head["contains"]) or len(head["to"]) > 0:
                queue.append([head])

    max_id = max((cid for fragment in fragments.values() for cid in fragment["chunk_ids"]), default=0) + 1
    while len(queue):
        path = queue.pop(0)
        fragment = path[-1]
        to = list(fragment["to"])
        path_chunk_ids = [cid for fragment in path if fragment["label"].startswith("entity_type") for cid in fragment["chunk_ids"]]
        for overlap in sorted(fragment["contains"], key=lambda x: len(x["to"])):
            if (
                  not overlap["label"].startswith("entity_type")
                  and (not path_chunk_ids or not overlap["chunk_ids"] or set(overlap["chunk_ids"]) & set(path_chunk_ids))
            ):
                path = [*path, overlap]
                # to.extend(overlap["to"])
        for node in to:
            queue.append([*path, node["fragment"]])
        if len(fragment["to"]) == 0 or len(fragment["chunk_ids"]) > 0:
            paths.append(path)
            has_id = False
            for eid in fragment["chunk_ids"]:
                has_id = True
                chunks[eid].extend(path)
            if not has_id:
                chunks[max_id].extend(path)
                max_id += 1
    chunks = {cid: chunk for cid, chunk in chunks.items() if any(e["label"].startswith("entity_type") for e in chunk)}

    entities = defaultdict(lambda: {"chunks": []})
    max_eid = max((eid for chunk in chunks.values() for fragment in chunk for eid in fragment["entity_ids"]), default=0)
    for cid, chunk in chunks.items():
        found_eid = False
        for fragment in chunk:
            # if fragment["label"].startswith("entity_type"):
            for eid in fragment["entity_ids"]:
                entities[eid]["chunks"].append({"fragments": sorted(chunk, key=lambda f: f["begin"]), "labels": chunk_labels(chunk)})
                found_eid = True
        if not found_eid:
            entities[max_eid + 1]["chunks"].append({"fragments": sorted(chunk, key=lambda f: f["begin"]), "labels": chunk_labels(chunk)})
            max_eid += 1
    for fragment in fragments.values():
        del fragment["to"]
        del fragment["from"]
        del fragment["contains"]
        del fragment["chunk_ids"]
        del fragment["entity_ids"]
        del fragment["attributes"]

    return {
        "doc_id": doc["doc_id"],
        "text": doc["text"],
        "fragments": dedup(fragments.values(), key=lambda f: (f["begin"], f["end"], f["label"])),
        "entities": [{
            "chunks": sorted(entity["chunks"], key=lambda chunk: path_extent(chunk["fragments"])),
            "fragments": dedup([f for chunk in entity["chunks"] for f in chunk["fragments"]], key=lambda f: (f["begin"], f["end"], f["label"])),
        } for eid, entity in entities.items()],
    }


def chunk_labels(chunk):
    labels = []
    for span in chunk:
        labels.append(span["label"])
        for attribute in span["attributes"]:
            labels.append(attribute)
    return sorted(set(labels))


def reduce_labels(label_sets):
    changed = True
    label_sets = [list(ls) for ls in label_sets]
    common_labels = tuple(set.intersection(*map(set, label_sets)))

    body_parts = list({label for ls in label_sets for label in ls if label.startswith('location_body_part')})
    if not body_parts:
        body_parts.append("location_body_part_breast")
    if len(body_parts) == 1:
        for label_set in label_sets:
            if not any(label.startswith('location_body_part') for label in label_set):
                label_set.append(body_parts[0])
    common_labels = tuple(set.intersection(*map(set, label_sets)))

    lateralities = list({label for ls in label_sets for label in ls if label.startswith('location_laterality')})
    if not lateralities and (
          "entity_type_score_density" in common_labels or "entity_type_score_acr" in common_labels or "proc_diag_mammography" in common_labels) and "location_body_part_breast" in common_labels:
        lateralities.append("location_laterality_bilateral")
    if len(lateralities) == 1:
        for label_set in label_sets:
            if not any(label.startswith('location_laterality') for label in label_set):
                label_set.append(lateralities[0])
    common_labels = tuple(set.intersection(*map(set, label_sets)))

    quadrants = list({label for ls in label_sets for label in ls if label.startswith('location_quadrant')})
    if len(quadrants) == 1:
        for label_set in label_sets:
            if not any(label.startswith('location_quadrant') for label in label_set):
                label_set.append(quadrants[0])
    common_labels = tuple(set.intersection(*map(set, label_sets)))

    temporalities = list({label for ls in label_sets for label in ls if label.startswith('is_')})
    if not temporalities:
        temporalities.append("is_now")
    if len(temporalities) == 1:
        for label_set in label_sets:
            if not any(label.startswith('is_') for label in label_set):
                label_set.append(temporalities[0])
    common_labels = tuple(set.intersection(*map(set, label_sets)))

    new_label_sets = []
    for ls in label_sets:
        if "is_now_and_passed" in ls:
            new_label_sets.append(sorted(set(ls) - {"is_now_and_passed"} | {"is_now"}))
            new_label_sets.append(sorted(set(ls) - {"is_now_and_passed"} | {"is_passed"}))
        else:
            new_label_sets.append(ls)
    label_sets = new_label_sets

    new_label_sets = []
    for ls in label_sets:
        if "is_now" in ls and "is_passed" in ls:
            new_label_sets.append(sorted(set(ls) - {"is_now"}))
            new_label_sets.append(sorted(set(ls) - {"is_passed"}))
        else:
            new_label_sets.append(ls)
    label_sets = new_label_sets

    new_label_sets = []
    for ls in label_sets:
        if "location_laterality_bilateral" in ls:
            new_label_sets.append(sorted(set(ls) - {"location_laterality_bilateral"} | {"location_laterality_left"}))
            new_label_sets.append(sorted(set(ls) - {"location_laterality_bilateral"} | {"location_laterality_right"}))
        else:
            new_label_sets.append(ls)
    label_sets = new_label_sets

    new_label_sets = []
    for ls in label_sets:
        if "location_laterality_left" in ls and "location_laterality_right" in ls:
            new_label_sets.append(sorted(set(ls) - {"location_laterality_right"}))
            new_label_sets.append(sorted(set(ls) - {"location_laterality_left"}))
        else:
            new_label_sets.append(ls)
    label_sets = new_label_sets

    for i in range(20):
        if not changed:
            break
        changed = False
        new_label_sets = []
        for i, ls1 in enumerate(label_sets):
            ls1 = tuple(sorted(ls1))
            flag = False
            for ls2 in label_sets:
                if ls1 == ls2:
                    continue
                ls2 = tuple(sorted(ls2))
                new_ls = tuple(sorted(set(ls1) | set(ls2)))
                if new_ls in enum_outputs:
                    new_label_sets.append(new_ls)
                    flag = True
                    changed = True
            if not flag:
                new_label_sets.append(ls1)
        label_sets = list(set(new_label_sets))
    return label_sets


@mappable
def preprocess_frame_annotations(doc, only_fragments=False):
    new_doc = {**doc, "fragments": [], "entities": []}
    for f in doc["fragments"]:
        if f["label"] == "location_laterality_bilateral":
            new_doc["fragments"].append({**f, "label": "location_laterality_left"})
            new_doc["fragments"].append({**f, "label": "location_laterality_right"})
        elif f["label"] != "is_now_and_passed":
            new_doc["fragments"].append(f)

    if only_fragments:
        return new_doc

    all_chunks = {}
    entities = []
    for entity in doc["entities"]:
        new_entity = {**entity, "chunks": []}
        all_chunks_labels = []
        for chunk in entity["chunks"]:
            chunk_fragments_labels = {fragment["label"] for fragment in chunk["fragments"]}
            chunk_labels = chunk_fragments_labels | set(chunk["labels"])
            chunk_labels = tuple(
                sorted({mapped_label for label in chunk_labels for mapped_label in (set(label_mapping.get(label, (label,))) & chunk_labels) or set(label_mapping.get(label, (label,)))}))

            matching_outputs = [(output, len(set(output) ^ set(chunk_labels))) for output in enum_outputs if set(output) >= set(chunk_labels)]
            if len(matching_outputs) == 0:
                matching_outputs = [(output, len(set(output) ^ set(chunk_labels))) for output in enum_outputs if set(output) & set(chunk_labels)]
            min_match_difference = min(count for output, count in matching_outputs)
            matching_outputs = [output for output, count in matching_outputs if count == min_match_difference]
            assert len(matching_outputs) > 0 and min_match_difference <= 3

            for output in matching_outputs:
                # if show and set(output) != set(chunk_labels):
                #    print("Replaced", ", ".join(a for l, a in abbreviations.items() if l in chunk_labels).ljust(50), "=>", ", ".join(a for l, a in abbreviations.items() if l in output))
                new_entity["chunks"].append({
                    **chunk,
                    "entities": [],
                    "labels": output,
                    "fragments": [
                        {"begin": f["begin"], "end": f["end"], "label": mapped_label}
                        for f in chunk["fragments"]
                        if "is_now_and_passed" != f["label"]
                        for mapped_label in label_mapping.get(f["label"], (f["label"],)) if mapped_label in output
                    ]})

        new_entity["chunks"] = sorted(dedup(new_entity["chunks"], key=lambda c: (tuple((f["begin"], f["end"], f["label"]) for f in c["fragments"]), tuple(sorted(c["labels"])))),
                                      key=lambda c: len(c["fragments"]))
        denested_chunks = []
        entity_labels = reduce_labels([chunk["labels"] for chunk in new_entity["chunks"]])
        for i, chunk in enumerate(sorted(new_entity["chunks"], key=lambda c: -len(c['fragments']))):
            if not any(
                  set((f["begin"], f["end"], f["label"]) for f in chunk["fragments"]) <= set((f["begin"], f["end"], f["label"]) for f in c["fragments"]) and
                  any(set(label_set) >= (set(chunk["labels"]) | set(c["labels"])) for label_set in entity_labels)
                  for c in denested_chunks
            ):
                key = hash((tuple(chunk["labels"]), tuple((f["begin"], f["end"], f["label"]) for f in chunk["fragments"])))
                chunk = all_chunks.setdefault(key, chunk)
                denested_chunks.append(chunk)
                chunk["entities"].append(new_entity)
        new_entity["chunks"] = denested_chunks
        new_doc["entities"].append(new_entity)
    new_entities = []
    for entity in new_doc["entities"]:
        if any("score" in label for chunk in entity["chunks"] for label in chunk["labels"]):
            left_chunks = [{**chunk, 'complete_labels': ['location_laterality_left']} for chunk in entity['chunks'] if not 'location_laterality_right' in chunk['labels']]
            right_chunks = [{**chunk, 'complete_labels': ['location_laterality_right']} for chunk in entity['chunks'] if not 'location_laterality_left' in chunk['labels']]
            if len(left_chunks):
                new_entities.append({**entity, "chunks": left_chunks})
            if len(right_chunks):
                new_entities.append({**entity, "chunks": right_chunks})
        else:
            new_entities.append(entity)
    all_chunks = []
    for entity in new_entities:
        for chunk in entity['chunks']:
            chunk['entities'] = []
            all_chunks.append(chunk)
    for entity in new_entities:
        for chunk in entity['chunks']:
            chunk['entities'].append(entity)
        entity_labels = reduce_labels([chunk["labels"] for chunk in entity["chunks"]])
        entity["labels"] = entity_labels
    new_doc["entities"] = new_entities
    for chunk in all_chunks:
        matching_label_sets = [ls for entity in chunk["entities"] for ls in entity["labels"] if set(chunk["labels"]) <= set(ls)]
        assert len(matching_label_sets) > 0, chunk['labels']
        chunk["complete_labels"] = sorted(set(chunk.get("complete_labels", ())) | set.intersection(*map(set, matching_label_sets)) - set(value_labels) | set(chunk["labels"]))

    new_doc = relink_doc_items(new_doc)
    return new_doc


def relink_doc_items(doc):
    fragment_index = {}
    chunk_index = {}
    doc['fragments'] = dedup(doc['fragments'], key=lambda fragment: (fragment['begin'], fragment['end'], fragment['label']))
    for fragment in doc['fragments']:
        fragment_index[(fragment['begin'], fragment['end'], fragment['label'])] = fragment
    for entity in doc['entities']:
        for chunk in entity['chunks']:
            chunk_index[(tuple(chunk['labels']), tuple(chunk['complete_labels']), tuple(sorted((f['begin'], f['end'], f['label']) for f in chunk['fragments'])))] = chunk
    for entity in doc['entities']:
        entity['chunks'] = [chunk_index[(tuple(chunk['labels']), tuple(chunk['complete_labels']), tuple(sorted((f['begin'], f['end'], f['label']) for f in chunk['fragments'])))]
                            for chunk in entity['chunks']]
        for chunk in entity['chunks']:
            chunk['fragments'] = [fragment_index[key] for key in dedup([(fragment['begin'], fragment['end'], fragment['label']) for fragment in chunk['fragments']], key=str)]
    return doc


def format_chunk(chunk, doc):
    sent_split_regex = r"((?:\s*\n)+\s*|(?:(?<=[a-z0-9)][.\:])\s+))(?=[[:upper:]-])"
    last = None
    fragments = sorted(((f, f["label"]) for f in chunk["fragments"]), key=lambda f: (f[0]["begin"], -f[0]["end"], not f[0]["label"].startswith("entity_type")))
    text = ""
    begin = end = 0

    labels = set(chunk["labels"])

    fragment_labels = {fragment["label"] for fragment in chunk["fragments"]}
    fragment_labels = tuple(sorted({mapped_label for label in fragment_labels for mapped_label in label_mapping.get(label, (label,))} & labels))
    hidden_labels = set(labels) - set(fragment_labels)

    text = (
          text + "> " +
          termcolor.colored(", ".join(abbr for label, abbr in abbreviations.items() if label in fragment_labels), "green").ljust(70) + " + " +
          ", ".join(abbr for label, abbr in abbreviations.items() if label in hidden_labels).ljust(30) + " | " +
          termcolor.colored(", ".join(abbr for label, abbr in abbreviations.items() if label in set(chunk["complete_labels"] if "complete_labels" in chunk else chunk["labels"]) - labels),
                            "yellow") + "\n"
    )

    for fragment, label in fragments:
        begin = fragment["begin"]
        end = fragment["end"]
        newlines = [m for m in regex.finditer(sent_split_regex, doc["text"][last or 0:end])]
        if last is None:
            text = text + (doc["text"][(last or 0) + newlines[-1].end():begin] if len(newlines) else "")
        if last is not None and end <= last:
            continue
        if last is not None and not newlines:
            text = text + doc["text"][last:begin]
            last = fragment["end"]
        elif last is not None:
            text = text + "\n" + (doc["text"][(last or 0) + newlines[-1].end():begin] if len(newlines) else "")
            last = None
        text = text + "[{}]({})".format(termcolor.colored(doc["text"][begin:end], 'red' if label.startswith('entity_type') else 'green'), termcolor.colored(abbreviations.get(label), 'grey'))
        last = fragment["end"]
    newlines = [m for m in regex.finditer(sent_split_regex, doc["text"][begin or end:])]
    text = text + (doc["text"][end:(begin or end) + newlines[0].start()] if len(newlines) else "")
    return text


def format_entity(entity, doc, do_raise=False):
    out = "------------------------------------------------------------" + "\n"
    entity_labels = entity["labels"]
    common_labels = sorted(set.intersection(*map(set, entity_labels)))
    for label_set in entity_labels:
        out = out + termcolor.colored("    > " + ", ".join(
            [abbr for label, abbr in abbreviations.items() if label in common_labels]
            + [abbr for label, abbr in abbreviations.items() if label in sorted(set(label_set) - set(common_labels))]
        ).ljust(60) + "\n", 'yellow')
    out = out + "------------------------------------------------------------" + "\n"

    for chunk in sorted(entity["chunks"], key=lambda chunk: path_extent(chunk["fragments"])):
        labels = chunk["labels"]

        chunk_out = textwrap.indent(format_chunk(chunk, doc), "    ") + "\n\n"

        try:
            fragment_labels = {fragment["label"] for fragment in chunk["fragments"]}
            # assert any(label.startswith("location_body_part") for label in entity_labels)
            has_lat = any(label.startswith("location_laterality") for label in labels)
            is_proc = any(label.startswith("entity_type_proc") for label in labels)
            is_passed = any(label.startswith("is_passed") for label in labels)
            is_now = any(label.startswith("is_now") for label in labels)
            # assert has_lat, "no lat"
            assert any(label.startswith("is_") for label in labels), "no temp"
            # assert any(label.startswith("location_body_part") for label in labels), "location_body_part"
            assert not any(label == "is_now" for label in {fragment["label"] for fragment in chunk["fragments"]}), "is_now in fragments"
            # if is_proc:
            assert any(label.startswith("location_laterality") for label in labels) == any(label.startswith("location_laterality") for label in fragment_labels), "unjustified lat"
            has_hidden_breast = any(label.startswith("location_body_part_breast") for label in labels) != any(label.startswith("location_body_part_breast") for label in fragment_labels)
            assert not has_hidden_breast or any(
                label.startswith("location_quad") or label.startswith("entity_type_score") or label.startswith("proc_diag_mammo") for label in labels), "unjustified breast"

            # assert any(label.startswith("location_laterality") for label in entity_labels) or "is_passed" in entity_labels or "is_future" in entity_labels or "location_body_part_other" in entity_labels
            assert any(label.startswith("entity_type") for label in labels), "not type"
        except Exception as e:
            # print(out)
            if do_raise:
                print(out)
                raise
            out += termcolor.colored("!!!!!!!!!!!!!!!!!   ", "blue") + termcolor.colored(str(e), "red") + termcolor.colored("   !!!!!!!!!!!!!!!\n", "blue")
        out = out + chunk_out
    return out


def format_entities(doc, explode_chunks=False, do_raise=False):
    out = doc["doc_id"] + "\n"
    if explode_chunks:
        entities = [
            {"chunks": [chunk], "labels": [chunk["labels"]]}
            for entity in doc["entities"]
            for chunk in entity["chunks"]
        ]
    else:
        entities = doc["entities"]
    for eid, entity in enumerate(sorted(entities, key=lambda entity: (entity["labels"][0][0], path_extent([fragment for chunk in entity["chunks"] for fragment in chunk["fragments"]])))):
        out += format_entity(entity, doc, do_raise)

    return out


class FrameDataset(BaseDataset):
    def __init__(self, train=None, test=None, val=None, seed=None, prep_fn=None):
        if prep_fn is None:
            prep_fn = lambda x: x
        brat = BRATDataset(train=train, test=test, val=val, seed=seed)
        train_data = list(prep_fn([extract_entities(doc) for doc in brat.train_data])) if brat.train_data is not None else None
        val_data = list(prep_fn([extract_entities(doc) for doc in brat.val_data])) if brat.val_data is not None else None
        test_data = list(prep_fn([extract_entities(doc) for doc in brat.test_data])) if brat.test_data is not None else None
        super().__init__(train_data=train_data, test_data=test_data, val_data=val_data)

    @classmethod
    def statistics(cls, docs):
        stats = defaultdict(lambda: 0)
        if docs is None:
            return {}
        for doc in docs:
            stats["fragment"] += len(dedup(doc["fragments"], key=lambda f: (f["begin"], f["end"], f["label"])))
            stats["chunks"] += sum(len(entity["chunks"]) for entity in doc["entities"])
            stats["entities"] += len(doc["entities"])
            stats["docs"] += 1
        return dict(stats)

    def describe(self):
        return pd.DataFrame({
            "train": self.statistics(self.train_data),
            "val": self.statistics(self.val_data),
            "test": self.statistics(self.test_data),
        }).T


class MixDataset(BaseDataset):
    def __init__(self, datasets, rates=None):
        rates = [rate for rate, d in zip(rates, datasets) if d.train_data is not None] if rates is not None else [1 for d in datasets if d.train_data is not None]
        total = sum(rates)
        rates = [rate / total for rate in rates]
        super().__init__(
            mix(*[loop(d.train_data, shuffle=True) for d in datasets if d.train_data is not None],
                rates=rates),
            list(chain.from_iterable([d.val_data for d in datasets if d.val_data is not None])),
            list(chain.from_iterable([d.test_data for d in datasets if d.test_data is not None])),
        )


def visualize(model, doc, show_scopes=True, show_ner=True, show_text=True, show_square="prediction", base_loss=(), do_cpu=False, do_eval=True):
    from matplotlib import pyplot as plt

    seed_all(42)
    if do_cpu:
        model.cpu()
    else:
        model.cuda()
    if do_eval:
        model.eval()
    else:
        model.train()
    model.zero_grad()

    docs = [doc]
    prep = list(model.preprocessor(docs, chain=True))
    res_idx = 0
    doc = docs[res_idx]
    batch = model.preprocessor.tensorize(prep)

    true_res = model.forward(
        list(model.preprocessor([{"doc_id": doc["doc_id"], "text": doc["text"]}], chain=True)),
        return_loss=base_loss,
        return_predictions=True)

    res = model.forward(prep, return_loss=("span", 'chunk_relation', 'chunk_label', 'normalizer') if 'entities' in doc else (),  # 'local', 'reg'),
                        return_predictions=True)

    def shorten(s, n):
        if len(s) <= n:
            return s
        return s[:(n - 3) // 2] + "..." + s[-(n - 3) // 2:]

    fragments_text = [
        str(fragment_idx).ljust(4)
        + ((" [" + shorten(" ".join(prep[res_idx]['words_text'][res['spans']['flat_spans_begin'][res_idx][fragment_idx].item():res['spans']['flat_spans_end'][res_idx][fragment_idx].item() + 1]),
                           15) + "] ").ljust(19)
           )
        for fragment_idx in range(res['spans']['flat_spans_label'].shape[1])
    ]
    fragments_text_with_scores = [
        fragments_text[fragment_idx] + "({:.2f})".format(res['spans']['flat_spans_logit'][res_idx][fragment_idx].item()).ljust(8)
        for fragment_idx in range(res['spans']['flat_spans_label'].shape[1])
    ]

    if show_text:
        display({k: float(v) for k, v in res.items() if k.endswith('loss') and v is not None})
        print("================================================================")
        print(doc["doc_id"])
        print(prep[res_idx]["original_sample"]["text"])
        print("================================================================")

        for t in fragments_text_with_scores:
            print(t)

        print("================================================================")

    if not doc.get("is_synthetic", False) and show_scopes:
        gold_tags = torch.zeros_like(res['chunks']["scope_logits"][res_idx].t().detach().cpu().sigmoid()).cpu()
        if 'valid_target' in res['chunks']:
            valid_target = (res['chunks']['valid_target'])[res_idx]
            mask = (res['chunks']['mask'])[res_idx]
            closest_target = (res['chunks']['closest_target'])[res_idx]
            loss_mask = (
                (((~res['chunks']["valid_target"]) | res['chunks']["closest_target"]) & res['chunks']["mask"])[res_idx]
            )
            for source_fragment, dest_fragment in (((res['chunks']['valid_target']) & res['chunks']["mask"])[res_idx]).nonzero():
                b, e = res['spans']['flat_spans_begin'][res_idx, dest_fragment].item(), res['spans']['flat_spans_end'][res_idx, dest_fragment].item() + 1
                gold_tags[b:e, source_fragment.item()] = 0.5
            # for source_fragment, dest_fragment in ((closest_target)).nonzero():
            #    b, e = res['spans']['flat_spans_begin'][res_idx, dest_fragment].item(), res['spans']['flat_spans_end'][res_idx, dest_fragment].item()+1
            #    gold_tags[b:e, source_fragment.item()] = 1.5
            for source_fragment, dest_fragment in ((res['chunks']['closest_target'] & res['chunks']["mask"])[res_idx]).nonzero():
                b, e = res['spans']['flat_spans_begin'][res_idx, dest_fragment].item(), res['spans']['flat_spans_end'][res_idx, dest_fragment].item() + 1
                gold_tags[b:e, source_fragment.item()] = 1.5
            for source_fragment, dest_fragment in (((~res['chunks']['valid_target']) & res['chunks']["mask"])[res_idx]).nonzero():
                b, e = res['spans']['flat_spans_begin'][res_idx, dest_fragment].item(), res['spans']['flat_spans_end'][res_idx, dest_fragment].item() + 1
                gold_tags[b:e, source_fragment.item()] = torch.clamp_max(gold_tags[b:e, source_fragment.item()], -1.5)
        else:
            gold_tags = None

        if show_scopes in ("B", "L", "Bh", "Lh"):
            pred_tags = res["chunks"]["scope_tags_logprobs"][..., (
                [2, 4] if show_scopes == "B" else
                [3, 4] if show_scopes == "L" else
                [6, 8] if show_scopes == "Bh" else
                [7, 8]
            )].exp().sum(-1).t().detach().cpu()
        else:
            pred_tags = res["chunks"]["scope_logits"][res_idx, ...].t().detach().cpu().sigmoid()

        for i, (b, e, l) in enumerate(zip(res['spans']['flat_spans_begin'][res_idx].tolist(), res['spans']['flat_spans_end'][res_idx].tolist(), res['spans']['flat_spans_label'][res_idx].tolist())):
            pred_tags[b:e + 1, i] = 1.5

        heatmap = torch.stack([pred_tags, gold_tags] if gold_tags is not None else [pred_tags], dim=-1).view(pred_tags.shape[0], -1)
        fig, ax = plt.subplots(figsize=(200, math.ceil(200 / 358 * len(prep[res_idx]['words_text']))))
        ax.set_xticks(list(range(len(fragments_text) * (2 if gold_tags is not None else 1))))
        ax.xaxis.set_ticks_position('top')
        ax.set_xticklabels([s for f in fragments_text for s in ((f, '') if gold_tags is not None else (f,))], rotation=45, ha='left')
        ax.set_yticks(list(range(len(prep[res_idx]['words_text']))))
        ax.set_yticklabels(["{} ".format(w) + "{}".format(i, sidx).ljust(4) for i, (w, sidx) in enumerate(zip(prep[res_idx]['words_text'], prep[res_idx]['words_sentence_idx']))])
        my_cmap = copy(plt.get_cmap('viridis'))
        my_cmap.set_over('white')
        my_cmap.set_under('black')
        ax.imshow(heatmap, vmax=1, vmin=0, cmap=my_cmap)

    if show_ner:
        gold_span_tags = torch.zeros(res['spans']["crf_tag_logprobs"].shape[2], len(prep[res_idx]['words_text']))
        pred_span_tags = true_res['spans']["crf_tag_logprobs"].cpu()[0, res_idx, ..., 1:].exp().sum(-1).detach()
        # pred_span_tags = true_res['spans']["raw_logits"].cpu()[0, res_idx, ..., 0].sigmoid().detach()
        for i, (b, e, l) in enumerate(zip(res['spans']['flat_spans_begin'][res_idx].tolist(), res['spans']['flat_spans_end'][res_idx].tolist(), res['spans']['flat_spans_label'][res_idx].tolist())):
            gold_span_tags[l, b:e + 1] = 1.5
        for i, (b, e, l) in enumerate(
              zip(true_res['spans']['flat_spans_begin'][res_idx].tolist(), true_res['spans']['flat_spans_end'][res_idx].tolist(), true_res['spans']['flat_spans_label'][res_idx].tolist())):
            pred_span_tags[l, b:e + 1] = 1.5
        heatmap = torch.stack([
            pred_span_tags,
            # true_res['spans']["tag_logprobs"][0, res_idx, ..., 1:].logsumexp(-1).exp().cpu().detach(),
            gold_span_tags,
        ], dim=1).view(-1, gold_span_tags.shape[1]).numpy().T
        # heatmap = model.decoder.span_scorer.crf.decode(true_res['spans']["tag_logits"][res_idx], torch.ones_like(true_res['spans']["tag_logits"][res_idx, ..., 0]).bool()).numpy().T > 0
        fig, ax = plt.subplots(figsize=(100, math.ceil(100 / 358 * len(prep[res_idx]['words_text']))))
        ax.set_xticks(list(range(len(model.preprocessor.vocabularies['fragment_label'].values) * 2)))
        ax.xaxis.set_ticks_position('top')
        ax.set_xticklabels([a for label in model.preprocessor.vocabularies['fragment_label'].values for a in (label,) * 2], rotation=45, ha='left')
        ax.set_yticks(list(range(len(prep[res_idx]['words_text']))))
        ax.set_yticklabels(["{} ".format(w) + "{}".format(i).ljust(4) for i, w in enumerate(prep[res_idx]['words_text'])])
        my_cmap = copy(plt.get_cmap('viridis'))
        my_cmap.set_over('white')
        my_cmap.set_under('black')
        ax.imshow(heatmap, vmax=1.005, vmin=0, cmap=my_cmap)

    if show_square:
        fig, ax = plt.subplots(figsize=(20, 20))
        logits = res["chunks"]["logits"].sigmoid()[0].detach().cpu()
        ax.xaxis.set_ticks_position('top')
        ax.set_xticks(list(range(len(fragments_text))))
        ax.set_xticklabels([s for f in fragments_text for s in (f,)], rotation=45, ha='left')
        ax.set_yticks(list(range(len(fragments_text))))
        ax.yaxis.set_ticks_position('right')
        ax.set_yticklabels([s for f in fragments_text for s in (f,)])
        if show_square == "prediction":
            heatmap = logits  # es["chunks"]["logits"].sigmoid()[0]
            heatmap[((heatmap > 0.5) & ~(heatmap.t() > 0.5)).tril()] = 1.5
            # heatmap[res["chunks"]["links_prediction"][0]] = 1.5
        else:
            heatmap = torch.zeros_like(res["chunks"]["closest_target"][0].float())
            heatmap[res["chunks"]["closest_target"][0]] = 1.
            heatmap[~res["chunks"]["valid_target"][0]] = 0.
            heatmap[~(res["chunks"]["closest_target"][0] | ~res["chunks"]["valid_target"][0])] = 0.5
        my_cmap = copy(plt.get_cmap('viridis'))
        my_cmap.set_over('white')
        my_cmap.set_under('black')
        ax.imshow(
            # res["chunks"]['same_entity_target'][0]
            heatmap
                .cpu().detach().numpy(),
            vmin=0., vmax=1.,
            cmap=my_cmap
        )

    # return true_res['spans']['tag_logits'].cpu().detach()
    return true_res['predictions'][res_idx], {**res, "batch": batch}, {**true_res, "batch": batch}


@mappable
def sentencize_corpus(doc, n=1, yield_rate=1.):
    new_docs = []
    sentences = list(regex_sentencize(doc["text"], r"(?:(?:\s*\n+\s*)|(?:(?<=[a-zA-Z0-9)]\.)\s+))()(?=[A-Z+-=])"))
    for i in range(n):
        last_begin = None
        for begin, end in sentences:
            if doc["text"][begin:end].strip():
                if last_begin is None:
                    last_begin = begin
                try:
                    sliced_doc = slice_document(doc, last_begin, end)
                except OverlappingEntityException:
                    # if cannot split at sentence end
                    pass
                else:
                    if random.random() < yield_rate:
                        new_docs.append(sliced_doc)
                        last_begin = None
        if last_begin is not None:
            new_docs.append(doc)
    return new_docs


bert_tokenizer_cache = {}
regex_tokenizer_cache = {}
