from nlstruct import register
from nlstruct.metrics import DocumentEntityMetric, entity_match_filter
from nlstruct.data_utils import split_spans, regex_tokenize
from tools import *
import numpy as np


def scores_from_match_matrix(match_scores, effective_scores=None, joint_matching=False):
    match_scores = np.asarray(match_scores, dtype=float)
    if effective_scores is None:
        effective_scores = match_scores == 1.
    elif isinstance(effective_scores, (int, float)):
        effective_scores = match_scores >= effective_scores
    effective_scores = np.asarray(effective_scores, dtype=float)
    matched_scores = np.zeros_like(match_scores) - 1.

    if 0 in match_scores.shape:
        return 0, match_scores.shape[0], match_scores.shape[1]

    tp = 0
    for pred_idx in range(match_scores.shape[0]):
        if joint_matching:
            pred_idx = match_scores.max(-1).argmax()
        gold_idx = match_scores[pred_idx].argmax()

        # match_score = match_scores[pred_idx, gold_idx].float()
        match_score = match_scores[pred_idx, gold_idx]
        effective_score = effective_scores[pred_idx, gold_idx]
        matched_scores[pred_idx, gold_idx] = max(matched_scores[pred_idx, gold_idx], effective_score)
        if match_score >= 0 and effective_score > 0:
            assert 0. <= effective_score <= 1.
            tp += effective_score
            match_scores[:, gold_idx] = -1
            match_scores[pred_idx, :] = -1
    return tp, match_scores.shape[0], match_scores.shape[1]


def f1_precision_recall(tp, pc, gc):
    if pc == 0:
        return float(gc == 0), 1., float(gc == 0)
    if gc == 0:
        return float(pc == 0), float(pc == 0), 1.
    return tp * 2 / (pc + gc), tp / pc, tp / gc


@register("frame_metric")
class FrameMetric(DocumentEntityMetric):
    def __init__(self, attributes=None, only_trigger=False, **kwargs):
        super().__init__(**kwargs)
        self.attributes = attributes
        self.only_trigger = only_trigger

    def compare_two_samples(self, pred, gold, return_match_scores=False):
        pred_frames = dedup([
            {
                "fragments": [f for f in frame["fragments"] if entity_match_filter(f["label"], self.attributes or "True")],
                "labels": tuple(sorted([l for l in frame["labels"] if entity_match_filter(l, self.attributes or "True")]))
            }
            for obj in pred["entities"]
            for frame in obj["chunks"]
            if entity_match_filter(frame["labels"], self.filter_entities or "True")
        ], key=str)

        gold_frames = {}
        for obj in gold["entities"]:
            for frame in obj["chunks"]:
                if entity_match_filter(frame["labels"], self.filter_entities or "True"):
                    dedup_frame = {  # **frame,
                        "fragments": sorted([f for f in frame["fragments"] if entity_match_filter(f["label"], self.attributes or "True")], key=str),
                        "labels": tuple(sorted([l for l in frame["labels"] if entity_match_filter(l, self.attributes or "True")])),
                        "complete_labels": [],
                    }
                    key = str(dedup_frame)
                    dedup_frame = gold_frames.setdefault(key, dedup_frame)
                    dedup_frame["complete_labels"].extend(tuple(sorted([l for l in frame["complete_labels"] if entity_match_filter(l, self.attributes or "True")])), )
        gold_frames = list(gold_frames.values())

        words = regex_tokenize(pred["text"], self.word_regex, return_offsets_mapping=True)
        words_begin = words["begin"]
        words_end = words["end"]

        def as_word_indices(fragments):
            new_begins, new_ends = split_spans(
                [fragment["begin"] for fragment in fragments],
                [fragment["end"] for fragment in fragments],
                words_begin,
                words_end,
            )
            return list(zip(new_begins.tolist(), new_ends.tolist()))

        match_scores = []
        effective_scores = []
        if not len(pred_frames):
            match_scores = np.zeros((0, len(gold_frames)))
            effective_scores = np.zeros((0, len(gold_frames)))
        else:
            for pred_idx, pred_frame in enumerate(pred_frames):
                pred_fragments = as_word_indices([fragment for fragment in pred_frame["fragments"] if fragment['label'].startswith('entity') or not self.only_trigger])
                match_scores.append([])
                effective_scores.append([])
                for gold_idx, gold_frame in enumerate(gold_frames):
                    # Text scores
                    gold_fragments = as_word_indices([fragment for fragment in gold_frame["fragments"] if fragment['label'].startswith('entity') or not self.only_trigger])
                    fragment_f1 = f1_precision_recall(
                        *scores_from_match_matrix(np.asarray([
                            [f1_precision_recall(*span_dice_overlap(pred_fragment, gold_fragment))[0] for gold_fragment in gold_fragments]
                            for pred_fragment in pred_fragments
                        ]).reshape(len(pred_fragments), len(gold_fragments)), effective_scores=1e-14)
                    )[0]

                    # Label scores
                    label_precision = len(set(pred_frame["labels"]) & set(gold_frame["complete_labels"])) / len(set(pred_frame["labels"]))
                    label_recall = len(set(pred_frame["labels"]) & set(gold_frame["labels"])) / len(set(gold_frame["labels"]))
                    label_f1 = 2. / (1. / label_precision + 1. / label_recall) if (label_precision > 0 and label_recall > 0) else 0.
                    # rint(label_precision, label_recall)

                    match_scores[-1].append(label_f1 * fragment_f1)
                    effective_scores[-1].append(
                        ((label_f1 >= self.binarize_label_threshold) if self.binarize_label_threshold is not False else label_f1) *
                        ((fragment_f1 >= self.binarize_tag_threshold) if self.binarize_tag_threshold is not False else fragment_f1)
                    )

        # print(np.asarray(text_scores))
        if return_match_scores:
            return (np.asarray(match_scores), np.asarray(effective_scores)), {**pred, "entities": pred_frames}, {**gold, "entities": gold_frames}
        return scores_from_match_matrix(match_scores, effective_scores=effective_scores, joint_matching=self.joint_matching)