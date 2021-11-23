import torch
import transformers
import torch.nn.functional as F

from nlstruct.models.common import *
from nlstruct.models.crf import LinearChainCRF
from nlstruct.models.ner import SpanScorer
from nlstruct.torch_utils import *
from tools import prod

REQUIRED = 1000000
IMPOSSIBLE = -1000000


def make_heads(x, n_heads):
    if isinstance(n_heads, int):
        n_heads = (n_heads,)
    total_heads = prod(n_heads)
    return x.view(*x.shape[:-1], *n_heads, x.shape[-1] // total_heads)  # if isinstance(n_heads, int) else x.view(*x.shape[:-1], *n_heads, x.shape[-1]//n_heads)


def log_and(x, dim):
    if torch.is_tensor(dim):
        return -torch.logaddexp(-x, -dim)
    return -torch.logsumexp(-x, dim)


class ScopeCRF(LinearChainCRF):
    def __init__(self, learnable_transitions=False, allow_hole_juxtaposition=False, do_holes=True, allow_hole_outside_scope=False, with_start_end_transitions=False):
        O, I, B, L, U, Ih, Bh, Lh, Uh = 0, 1, 2, 3, 4, 5, 6, 7, 8

        if do_holes:
            num_tags = 9
        else:
            num_tags = 5
        forbidden_transitions = torch.ones(num_tags, num_tags, dtype=torch.bool)
        forbidden_transitions[O, O] = 0

        forbidden_transitions[O, B] = 0
        forbidden_transitions[O, U] = 0

        forbidden_transitions[B, I] = 0
        forbidden_transitions[B, L] = 0

        forbidden_transitions[I, I] = 0
        forbidden_transitions[I, L] = 0

        forbidden_transitions[L, O] = 0
        forbidden_transitions[U, O] = 0

        if do_holes:
            forbidden_transitions[B, Bh] = 0
            forbidden_transitions[B, Uh] = 0

            forbidden_transitions[I, Bh] = 0
            forbidden_transitions[I, Uh] = 0

            forbidden_transitions[Bh, Lh] = 0
            forbidden_transitions[Bh, Ih] = 0

            forbidden_transitions[Ih, Ih] = 0
            forbidden_transitions[Ih, Lh] = 0

            forbidden_transitions[Uh, I] = 0
            forbidden_transitions[Uh, L] = 0
            forbidden_transitions[Lh, I] = 0
            forbidden_transitions[Lh, L] = 0

            if allow_hole_juxtaposition:
                forbidden_transitions[Bh, Uh] = 0
                forbidden_transitions[Uh, Uh] = 0
                forbidden_transitions[Uh, Lh] = 0
                forbidden_transitions[Lh, Uh] = 0
                forbidden_transitions[Lh, Bh] = 0

            if allow_hole_outside_scope:
                forbidden_transitions[O, Bh] = 0
                forbidden_transitions[O, Uh] = 0
                forbidden_transitions[Uh, O] = 0
                forbidden_transitions[Lh, O] = 0

        start_forbidden_transitions = None
        end_forbidden_transitions = None
        super().__init__(forbidden_transitions,
                         start_forbidden_transitions,
                         end_forbidden_transitions,
                         with_start_end_transitions=with_start_end_transitions,
                         learnable_transitions=learnable_transitions)


class RelationLoss(torch.nn.Module):
    def __init__(self, symmetric=False, weight=1.):
        super().__init__()
        self.symmetric = symmetric
        self.weight = weight

    def forward(self, logits, valid_target, closest_target, mask):
        if self.symmetric:
            valid_target = valid_target | valid_target.transpose(-1, -2)
            closest_target = closest_target | closest_target.transpose(-1, -2)
            mask = multi_dim_triu(mask | mask.transpose(-1, -2))
        loss_mask = ((~valid_target) | closest_target) & mask
        loss = bce_with_logits(logits, closest_target & loss_mask, reduction='none').masked_fill(~loss_mask, 0).sum(-1).sum(-1)
        return loss * self.weight


class GroupedLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, n_groups=1):
        super().__init__()
        self.n_groups = n_groups
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(n_groups, output_size))
        else:
            self.bias = None
        self.weight = torch.nn.Parameter(torch.stack([torch.nn.Linear(input_size, output_size).weight.data.T for _ in range(n_groups)], dim=0))

    def forward(self, x, reshape=True):
        if not reshape:
            x = torch.einsum('...ni,nio->...no', x, self.weight)
            if self.bias is not None:
                x = x + self.bias
            return x
        (*base_shape, dim) = x.shape
        x = x.reshape(*base_shape, self.n_groups, dim // self.n_groups)
        x = torch.einsum('...ni,nio->...no', x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x.reshape(*base_shape, x.shape[-1] * self.n_groups)


@register("relative_attention")
class RelativeAttention(torch.nn.Module):
    def __init__(self,
                 size,
                 n_heads,
                 pos_size=None,
                 query_size=None,
                 key_size=None,
                 value_size=None,
                 head_size=None,
                 position_embedding=None,
                 dropout_p=0.1,
                 same_key_query_proj=False,
                 same_positional_key_query_proj=False,
                 n_coordinates=1,
                 head_bias=True,
                 do_pooling=True,
                 mode=('c2c', 'p2c', 'c2p'),
                 n_additional_heads=0):
        super().__init__()

        if query_size is None:
            query_size = size
        if key_size is None:
            key_size = size
        if value_size is None:
            value_size = key_size
        if pos_size is None:
            pos_size = size
        if head_size is None and key_size is not None:
            assert key_size % n_heads == 0
            head_size = key_size // n_heads
        value_head_size = None
        if do_pooling and size is not None:
            assert size % n_heads == 0
            value_head_size = size // n_heads
        self.n_coordinates = n_coordinates
        self.n_heads = n_heads + n_additional_heads
        self.n_additional_heads = n_additional_heads
        self.mode = mode
        n_query_heads = n_heads + n_additional_heads
        self.content_key_proj = torch.nn.Linear(key_size, n_query_heads * head_size) if key_size else Identity()
        if isinstance(position_embedding, torch.nn.Parameter):
            self.position_embedding = position_embedding
        else:
            self.register_buffer('position_embedding', position_embedding)

        if same_key_query_proj:
            self.content_query_proj = self.content_key_proj
        else:
            self.content_query_proj = torch.nn.Linear(query_size, n_query_heads * head_size) if query_size else Identity()
        if do_pooling:
            self.content_value_proj = torch.nn.Linear(value_size, value_head_size * n_heads)

        self.position_key_proj = GroupedLinear(pos_size // n_coordinates, head_size * n_query_heads // n_coordinates, n_groups=n_coordinates) if pos_size else Identity()
        if same_key_query_proj or same_positional_key_query_proj:
            self.position_query_proj = self.position_key_proj
        else:
            self.position_query_proj = GroupedLinear(pos_size // n_coordinates, head_size * n_query_heads // n_coordinates, n_groups=n_coordinates) if pos_size else Identity()

        self.dropout = torch.nn.Dropout(dropout_p)
        if head_bias:
            self.bias = torch.nn.Parameter(torch.zeros(n_query_heads))
        self.output_size = size

    def forward(self, content_queries, content_keys=None, content_values=None, mask=None, relative_positions=None, no_position_mask=None, base_attn=None):
        n_samples, n_queries = content_queries.shape[:2]
        n_keys = content_queries.shape[1]

        attn = None
        if 0 in content_queries.shape or 0 in content_keys.shape:
            attn = torch.zeros(content_queries.shape[0], content_queries.shape[1], content_keys.shape[1], self.n_heads, device=content_queries.device)
            if content_values is None:
                return attn
            return torch.zeros(content_queries.shape[0], content_queries.shape[1], self.content_value_proj.weight.shape[0], device=content_queries.device), attn

        if content_keys is None:
            content_keys = content_queries

        content_keys = make_heads(self.content_key_proj(self.dropout(content_keys)), self.n_heads)
        content_queries = make_heads(self.content_query_proj(self.dropout(content_queries)), self.n_heads)
        if content_values is not None:
            content_values = make_heads(self.content_value_proj(self.dropout(content_values)), self.n_heads - self.n_additional_heads)

        size = content_queries.shape[-1]
        if 'c2c' in self.mode:
            content_to_content_attn = torch.einsum('nihd,njhd->nijh', content_queries, content_keys) / math.sqrt(size)
        else:
            content_to_content_attn = torch.zeros(n_samples, n_queries, n_keys, self.n_heads, device=content_queries.device)

        if relative_positions is not None and ('p2c' in self.mode or 'c2p' in self.mode):
            position_keys = make_heads(self.position_key_proj(self.dropout(self.position_embedding)), (self.n_coordinates, self.n_heads))
            # position_keys = make_heads(self.dropout(self.position_key_embedding), self.n_heads)
            position_queries = make_heads(self.position_query_proj(self.dropout(self.position_embedding)), (self.n_coordinates, self.n_heads))
            # position_queries = make_heads(self.dropout(self.position_query_embedding), self.n_heads)
            relative_positions = (position_queries.shape[0] // 2 + relative_positions).clamp(0, position_queries.shape[0] - 1)

            content_to_position_attn = torch.einsum('nihxd,zxhd->nizhx', make_heads(content_queries, self.n_coordinates), position_keys)
            content_to_position_attn = gather(content_to_position_attn, index=relative_positions.unsqueeze(-2), dim=2).sum(-1) / math.sqrt(size)
            if no_position_mask is not None:
                content_to_position_attn = content_to_position_attn.masked_fill(no_position_mask[..., None, None], 0)
            position_to_content_attn = torch.einsum('zxhd,njhxd->nzjhx', position_queries, make_heads(content_keys, self.n_coordinates))
            # position_to_content_attn = gather(position_to_content_attn, index=relative_positions.transpose(1, 2).unsqueeze(-2), dim=1).transpose(1, 2).sum(-1) / math.sqrt(size)
            position_to_content_attn = gather(position_to_content_attn, index=relative_positions.unsqueeze(-2), dim=1).sum(-1) / math.sqrt(size)
            if no_position_mask is not None:
                position_to_content_attn = position_to_content_attn.masked_fill(no_position_mask[..., None, None], 0)

            if 'c2p' in self.mode and 'p2c' in self.mode:
                attn = (content_to_content_attn + position_to_content_attn + content_to_position_attn) / math.sqrt(3 if 'c2c' in self.mode else 2)
            elif 'c2p' in self.mode:
                attn = (content_to_content_attn + content_to_position_attn) / math.sqrt(2 if 'c2c' in self.mode else 1)
            elif 'p2c' in self.mode:
                attn = (content_to_content_attn + position_to_content_attn) / math.sqrt(2 if 'c2c' in self.mode else 1)
        else:
            attn = content_to_content_attn

        if base_attn is not None:
            attn = attn + base_attn

        if hasattr(self, 'bias'):
            attn = attn + self.bias
        if mask.ndim == 2:
            mask = mask[:, None, :, None]
        if mask.ndim == 3:
            mask = mask[:, :, :, None]
        if content_values is not None:
            weights = self.dropout(attn[..., self.n_additional_heads:].masked_fill(~mask, IMPOSSIBLE).softmax(-2))
            pooled = torch.einsum('nijh,njhd->nihd', weights, content_values)
            pooled = pooled.reshape(*pooled.shape[:-2], -1)
            return pooled, attn
        return attn


@register("span_embedding")
class SpanEmbedding(torch.nn.Module):
    def __init__(self, input_size, n_labels, mode="max", n_heads=None, do_label_embedding=True, do_value_proj=False, label_dropout_p=0.1):
        super().__init__()
        self.mode = mode
        if mode != "bounds_cat":
            self.pooler = Pooler(mode=mode, input_size=input_size, n_heads=n_heads, do_value_proj=do_value_proj)
        if do_label_embedding:
            self.label_embedding = torch.nn.Embedding(n_labels, input_size).weight
        self.label_dropout_p = label_dropout_p
        self.output_size = input_size

    def forward(self, words_embed, spans_begin, spans_end, spans_label):
        n_samples, n_words, size = words_embed.shape
        if self.mode == "bounds_cat":
            # return words_embed[torch.arange(n_samples)[:, None], spans_begin]
            # noinspection PyArgumentList
            pooled = torch.cat([
                words_embed[..., size // 2:][torch.arange(n_samples)[:, None], spans_begin],
                words_embed[..., :size // 2][torch.arange(n_samples)[:, None], spans_end],
            ], size=-1)
        else:
            pooled = self.pooler(words_embed, (spans_begin, spans_end + 1))

        if hasattr(self, 'label_embedding'):
            label_embeds = torch.einsum('ld,...l->...d', self.label_embedding, spans_label.float())
            if self.training:
                label_embeds = label_embeds.masked_fill(torch.rand_like(label_embeds[..., [0]]) < self.label_dropout_p, 0)
            pooled = pooled + label_embeds

        return pooled


@register("normalizer", do_not_serialize=["multilabels", "ner_label_to_multilabels"])
class Normalizer(torch.nn.Module):
    def __init__(self, input_size, multilabels, ner_label_to_multilabels, pooler_mode="max", classifier_mode="dot"):
        super().__init__()

        if classifier_mode == "dot":
            self.classifier = torch.nn.Linear(input_size, multilabels.shape[1])
        self.pooler = Pooler(mode=pooler_mode)

        self.register_buffer('multilabels', multilabels)  # n_multilabels * n_labels
        self.register_buffer('ner_label_to_multilabels', ner_label_to_multilabels)  # n_ner_label * n_multilabels

    def forward(self, words_embed, spans_begin, spans_end, spans_label, spans_mask, batch=None, return_loss=True):
        pooled = self.pooler(words_embed, (spans_begin, spans_end + 1))

        scores = self.classifier(pooled)
        span_allowed_multilabels = self.ner_label_to_multilabels[spans_label]  # [b]atch_size * n_[m]ultilabel_combination
        multilabel_scores = torch.einsum('bfl,ml->bfm', scores, self.multilabels.float()).masked_fill(~span_allowed_multilabels, IMPOSSIBLE)

        if return_loss:
            gold_outputs = (batch['fragments_label'][..., None, :] == self.multilabels).all(-1)
            assert gold_outputs.any(-1)[spans_mask].all()
            mention_err = torch.einsum('bfm,bfm->bf', -multilabel_scores.log_softmax(-1), gold_outputs.float())[spans_mask]
            loss = mention_err.sum()
            pred = batch['fragments_label']
        else:
            loss = mention_err = gold_outputs = None
            pred = self.multilabels[multilabel_scores.argmax(-1)]  # => [b]atch_size * n_fine_[l]abels

        return {
            "prediction": pred,
            "loss": loss,
            "mention_err": mention_err,
            "multilabel_scores": multilabel_scores,
            "gold_outputs": gold_outputs,
        }


@register("entity_decoder", do_not_serialize=('constrained_outputs', 'normalizer_scheme'))
class EntityDecoder(torch.nn.Module):
    ENSEMBLE = "ensemble_entity_decoder"

    def __init__(self,
                 pos_size=768,
                 max_relative_position=100,

                 head_labels=None,
                 constrained_outputs=None,
                 normalizer_scheme=None,
                 contextualizer=None,

                 span_scorer={},
                 span_embedding=dict(mode='bounds_cat'),

                 use_sentence_coordinates=False,

                 chunk_decoder=dict(
                     scope_head_size=64,
                     do_target_links_count=True,
                     scope_match_temperature=1.,
                 ),
                 normalizer=dict(
                 ),
                 _classifier=None,
                 _preprocessor=None,
                 _encoder=None,
                 ):
        super().__init__()
        input_size = _encoder.output_size

        ner_labels = _preprocessor.vocabularies["fragment_label"].values
        entity_labels = _preprocessor.vocabularies["entity_label"].values

        self.base_supervision = ()

        self.norm = torch.nn.LayerNorm(input_size)
        self.proj = torch.nn.Linear(input_size, input_size)
        self.contextualizer = Contextualizer(**{**contextualizer, "input_size": input_size}) if contextualizer is not None else None

        size = input_size if self.contextualizer is None else self.contextualizer.output_size
        pos_size = size if pos_size is None else pos_size

        self.span_scorer = SpanScorer(**{
            "input_size": size,
            "n_labels": len(ner_labels),
            **span_scorer,
        })
        self.span_embedding = SpanEmbedding(**{"input_size": size, "n_labels": len(entity_labels), **span_embedding})
        position_embedding = torch.nn.Embedding(max_relative_position, pos_size).weight
        self.use_sentence_coordinates = use_sentence_coordinates

        ###########################
        # Label constraints logic #
        ###########################
        constrained_outputs = torch.tensor([[label in output for label in entity_labels] for output in constrained_outputs])
        head_labels = torch.as_tensor([label in head_labels for label in entity_labels])
        allowed_triple_relations = torch.einsum('xi,xj,xk->ijk', constrained_outputs.float(), constrained_outputs.float(), constrained_outputs.float()).bool()
        allowed_a2a2h_relations = allowed_triple_relations & head_labels[None, None, :]
        allowed_a2h_relations = allowed_triple_relations.any(-1) & head_labels[None, :]

        self.chunk_decoder = get_instance({**{
            "module": "chunk_decoder",
            "input_size": self.span_embedding.output_size,
            "position_embedding": position_embedding,
            "allowed_a2h_relations": allowed_a2h_relations,
            "allowed_a2a2h_relations": allowed_a2a2h_relations,
            "constrained_outputs": constrained_outputs,
            **chunk_decoder,
        }})

        multilabels = {}
        ner_label_to_multilabels = [None] * len(ner_labels)

        for ner_label, ner_multilabels in normalizer_scheme.items():
            indices = [multilabels.setdefault(multilabel, len(multilabels)) for multilabel in ner_multilabels]
            ner_label_to_multilabels[ner_labels.index(ner_label)] = indices
        ner_label_to_multilabels = torch.as_tensor([
            [True if idx in multilabel_indices else False for idx in range(len(multilabels))]
            for multilabel_indices in ner_label_to_multilabels
        ]).bool()
        multilabels = torch.as_tensor([
            [label in multilabel for label in entity_labels]
            for multilabel in multilabels
        ]).bool()
        self.normalizer = Normalizer(
            input_size=self.span_embedding.output_size,
            multilabels=multilabels,
            ner_label_to_multilabels=ner_label_to_multilabels,
            **normalizer,
        )

    def fast_params(self):
        return [
            *self.chunk_decoder.fast_params(),
            *self.span_scorer.fast_params(),
        ]

    def on_training_step(self, current, total):
        self.chunk_decoder.on_training_step(current, total)

    def forward(self, words_embed, batch=None, return_loss=False, return_predictions=False):
        loss_dict = {}

        if return_loss is True:
            supervision = {*self.base_supervision, "span", "chunk_relation", "normalizer", "chunk_label", "reg", "coref"}  # , "c2c", "label", "reg")
        elif return_loss is False:
            supervision = set(self.base_supervision)
        else:
            supervision = set(return_loss)
        if isinstance(words_embed, tuple):
            words_embed, lm_embeds = words_embed

        words_mask = batch['words_mask']

        original_words_embed = words_embed
        if self.contextualizer is not None:
            words_embed = self.contextualizer(original_words_embed, words_mask)

        ############################
        # Generate spans           #
        ############################

        spans = self.span_scorer(words_embed, words_mask, {**batch, "fragments_label": batch.get("fragments_ner_label", None)}, force_gold=bool({"local", "span", "label", "normalizer"} & supervision))
        spans_mask, spans_begin, spans_end, spans_ner_label, spans_chunks, chunks_spans = (
            spans["flat_spans_mask"],
            spans["flat_spans_begin"],
            spans["flat_spans_end"],
            spans["flat_spans_label"],
            batch["fragments_chunks"],
            batch["chunks_fragments"])

        normalizer_result = self.normalizer(words_embed, spans_begin, spans_end, spans_ner_label, spans_mask, batch=batch, return_loss=bool({"local", "normalizer", "label"} & supervision))
        spans_label = normalizer_result["prediction"]

        ##################################
        # Transform spans with attention #
        ##################################
        spans_embed = self.span_embedding(words_embed, spans_begin, spans_end, spans_label)

        #######################################
        # Decode chunks (local scope cliques) #
        #######################################
        chunks = self.chunk_decoder(
            words_embed=words_embed,
            words_mask=words_mask,
            spans_embed=spans_embed,
            spans_begin=spans_begin,
            spans_end=spans_end,
            spans_label=spans_label,
            spans_mask=spans_mask,
            batch=batch,
            supervision=supervision,
        )

        #####################
        # Sum up the losses #
        #####################
        #####################
        if "span" in supervision:
            span_loss_dict = self.span_scorer.loss(spans, {**batch, "fragments_label": batch.get("fragments_ner_label", None)})
            del span_loss_dict["loss"]
            loss_dict.update(span_loss_dict)
        if "normalizer" in supervision:
            loss_dict["normalizer_loss"] = normalizer_result["loss"]
        if "chunk_relation" in supervision:
            loss_dict["chunk_relation_loss"] = chunks["chunk_relation_loss"]
            loss_dict["scope_tag_loss"] = chunks["scope_tag_loss"]
            # loss_dict["biaffine_relation_loss"] = chunks["biaffine_relation_loss"]
        if "chunk_label" in supervision:
            loss_dict["chunk_label_loss"] = chunks["chunk_label_loss"]
        loss = sum(sub_loss for sub_loss in loss_dict.values())

        predictions = None
        if return_predictions:
            predictions = [{"fragments": [], "entities": []} for sample in batch["original_sample"]]

            entities_fragments = chunks["chunks_spans"]
            entities_mask = (entities_fragments != -1).any(-1)
            entities_label = chunks["chunks_labels"]
            if 0 not in entities_fragments.shape:
                for sample_idx, entity_idx in entities_mask.nonzero(as_tuple=False).tolist():
                    chunk_labels = entities_label[sample_idx, entity_idx].nonzero(as_tuple=True)[0].tolist()
                    predictions[sample_idx]["entities"].append({
                        "entity_id": len(predictions[sample_idx]),
                        "confidence": 1.,  # entities_confidence[sample_idx, entity_idx].item(),
                        "chunks": [{
                            "chunk_id": len(predictions[sample_idx]),
                            "confidence": 1.,  # entities_confidence[sample_idx, entity_idx].item(),
                            "label": chunk_labels,
                            "fragments": [
                                {
                                    "fragment_id": fragment_idx,
                                    "begin": spans_begin[sample_idx, fragment_idx].item(),
                                    "end": spans_end[sample_idx, fragment_idx].item(),
                                    "label": label,
                                }
                                for fragment_idx in entities_fragments[sample_idx, entity_idx].tolist()
                                if fragment_idx >= 0
                                for label in spans_label[sample_idx, fragment_idx].nonzero(as_tuple=True)[0].tolist()
                                if label in chunk_labels
                            ]
                        }]
                    })

            if 0 not in spans_mask.shape:
                # entities_confidence = entities_label_scores.detach().cpu()[-1].sigmoid().masked_fill(~entities_label, 1).prod(-1)
                # for sample_idx, entity_idx in (~entities_label[..., 0]).masked_fill(~entities_mask.cpu(), False).nonzero(as_tuple=False).tolist():
                for sample_idx, fragment_idx in spans_mask.nonzero(as_tuple=False).tolist():
                    for label in spans_label[sample_idx, fragment_idx].nonzero(as_tuple=True)[0].tolist():
                        predictions[sample_idx]["fragments"].append({
                            "fragment_id": fragment_idx,
                            "confidence": 1.,  # entities_confidence[sample_idx, entity_idx].item(),
                            "begin": spans_begin[sample_idx, fragment_idx].item(),
                            "end": spans_end[sample_idx, fragment_idx].item(),
                            "label": label,
                        })

        return {
            "predictions": predictions,
            **loss_dict,
            "holes_delta_loss": chunks["holes_delta"].detach() if "holes_delta" in chunks else 0,
            "loss": loss,
            "spans": spans,
            "chunks": chunks,
            "words_embed": words_embed,
            "spans_embed": spans_embed,
            "normalized": normalizer_result,
        }


@register("chunk_decoder", do_not_serialize=(
      "position_embedding",
      "allowed_a2h_relations",
      "allowed_a2a2h_relations",
      "constrained_outputs",
))
class ChunkDecoder(torch.nn.Module):
    ENSEMBLE = "ensemble_chunk_decoder"

    def __init__(self,
                 input_size,
                 scope_head_size=64,
                 position_embedding=None,
                 allowed_a2h_relations=None,
                 allowed_a2a2h_relations=None,
                 constrained_outputs=None,
                 do_target_links_count=True,
                 scope_match_temperature=1.,
                 chunk_relation_loss_weight=1,
                 scope_tag_loss_weight=1,
                 label_loss_weight=0.5,
                 do_biaffine=True,
                 do_scopes=True,
                 do_constraints="always",
                 biaffine_scale=0.2,
                 dropout_p=0.2,
                 biaffine_mode="and",
                 label_proj_pooling="max",
                 attention_mode=('c2c', 'c2p', 'p2c'),
                 scope_crf_params={},
                 symmetric_training=False,
                 scope_supervision="word",
                 do_scope_holes=True,
                 ):
        super().__init__()

        if do_constraints is True:
            do_constraints = "always"
        assert do_constraints in ("always", False, "test-only")
        self.do_constraints = do_constraints

        self.register_buffer("allowed_a2h_relations", allowed_a2h_relations)
        self.register_buffer("allowed_a2a2h_relations", allowed_a2a2h_relations)
        self.register_buffer("constrained_outputs", constrained_outputs.float())

        self.label_proj = torch.nn.Linear(input_size, constrained_outputs.shape[1])
        self.label_proj.bias = None
        self.do_target_links_count = do_target_links_count
        self.scope_match_temperature = scope_match_temperature
        self.chunk_relation_loss = RelationLoss(symmetric=False, weight=chunk_relation_loss_weight)
        self.scope_tag_loss_weight = scope_tag_loss_weight
        self.label_loss_weight = label_loss_weight
        self.dropout = torch.nn.Dropout(dropout_p)

        self.symmetric_training = symmetric_training
        self.do_scopes = do_scopes
        self.label_proj_pooling = label_proj_pooling
        if label_proj_pooling == "embed-max":
            self.pre_label_proj = torch.nn.Linear(input_size, input_size)

        self.do_scope_holes = do_scopes and do_scope_holes
        self.scope_supervision = scope_supervision if do_scopes else False
        self.scope_tag_loss_weight = scope_tag_loss_weight if scope_supervision == "word" else 0.
        if self.do_scopes:
            if self.do_scope_holes:
                self.register_buffer('bounds_to_tags', torch.tensor([
                    # [0, 1, 1, 1, 1],
                    [0, 0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1],
                ], dtype=torch.float))
            else:
                self.register_buffer('bounds_to_tags', torch.tensor([
                    # [0, 1, 1, 1, 1],
                    [0, 0, 1, 0, 1],
                    [0, 0, 0, 1, 1],
                ], dtype=torch.float))
            self.scope_bias = torch.nn.Parameter(torch.zeros(self.bounds_to_tags.shape[1], dtype=torch.float))
            self.scope_proj = RelativeAttention(
                size=input_size,
                n_heads=self.bounds_to_tags.shape[1],
                do_pooling=False,
                head_size=scope_head_size,
                pos_size=position_embedding.shape[-1] if position_embedding is not None else None,
                position_embedding=position_embedding,
                dropout_p=dropout_p,
                mode=attention_mode,
                head_bias=True,
            )
            self.scope_pooler = Pooler(mode="mean")
            self.scope_decoder = ScopeCRF(**{**scope_crf_params, "do_holes": do_scope_holes})

        assert biaffine_mode in ("and", "sum", "min")
        self.do_biaffine = do_biaffine
        self.biaffine_mode = biaffine_mode
        self.biaffine_scale = biaffine_scale  # torch.nn.Parameter(torch.tensor(0.2))
        if self.do_biaffine:
            self.biaffine_proj = RelativeAttention(
                size=input_size,
                n_heads=1,
                do_pooling=False,
                head_size=input_size,
                pos_size=position_embedding.shape[-1] if position_embedding is not None else None,
                position_embedding=position_embedding,
                head_bias=True,
                dropout_p=dropout_p,
                mode=attention_mode,
            )
        self.progress = 0.

    def on_training_step(self, step_idx, total_steps):
        self.progress = 2. * float(step_idx) / total_steps

    def fast_params(self):
        return [self.scope_bias] if hasattr(self, 'scope_bias') else []  # *self.scope_decoder.parameters(), self.scope_bias, *self.label_proj.parameters()]

    def forward(self, words_embed, words_mask, spans_embed, spans_begin, spans_end, spans_label, spans_mask, batch=None, supervision=set()):
        device = words_embed.device
        sample_index = torch.arange(len(words_embed), device=device)[:, None]
        n_spans = spans_mask.shape[-1]
        res_dict = {}

        if (self.do_constraints == "test-only" and not self.training) or self.do_constraints == "always":
            allowed_a2h_relations = self.allowed_a2h_relations
            allowed_a2a2h_relations = self.allowed_a2a2h_relations
        else:
            allowed_a2h_relations = self.allowed_a2h_relations | self.allowed_a2h_relations.any(0)
            allowed_a2a2h_relations = self.allowed_a2a2h_relations | self.allowed_a2a2h_relations.any(0).any(0)

        #########################################
        #               CHUNK DECODING          #
        #########################################

        scope_logits = scope_tags_logprobs = scope_bounds_emissions = scope_emissions = scope_tags_logits = None
        spans_positions = (torch.stack([spans_begin], dim=-1), torch.stack([spans_end], dim=-1))
        words_positions = (torch.stack([torch.arange(words_embed.shape[1], device=words_embed.device).view(1, -1).repeat_interleave(len(words_embed), dim=0)], dim=-1),) * 2

        if self.do_scopes:
            # begin and end coordinates of each element (span or word)
            x = words_positions[0][:, None, :] - spans_positions[1][:, :, None]  # span end -> word
            y = words_positions[1][:, None, :] - spans_positions[0][:, :, None]  # span begin -> word
            spans_words_distance = torch.where(
                (x.sign() > 0) & (y.sign() > 0), torch.minimum(x, y),
                torch.where((x.sign() < 0) & (y.sign() < 0), torch.maximum(x, y), 0)
            )
            span_word_attn = self.scope_proj(
                content_queries=spans_embed,
                content_keys=words_embed,
                mask=words_mask,
                relative_positions=spans_words_distance,
            )
            scope_bounds_emissions = torch.cat([
                *[span_word_attn[..., :2].masked_fill(
                    torch.stack([
                        y.squeeze(-1) > 0,  # start tag cannot be after span begin
                        x.squeeze(-1) < 0,  # stop tag cannot be before span end
                    ], dim=-1), IMPOSSIBLE)],
                *([span_word_attn[..., 2:4]] if self.do_scope_holes else []),
            ], dim=-1) if not self.training else span_word_attn[..., :(4 if self.do_scope_holes else 2)]
            scope_emissions = scope_bounds_emissions[spans_mask] @ self.bounds_to_tags + self.scope_bias
            scope_mask = words_mask.unsqueeze(1).expand(*spans_mask.shape, -1)[spans_mask]
            scope_tags_logits = torch.zeros(span_word_attn.shape[:-1], device=scope_emissions.device, dtype=torch.float)
            scope_tags_logprobs = self.scope_decoder.marginal(scope_emissions, scope_mask)
            scope_tags_logits[spans_mask] = (
                  scope_tags_logprobs[..., [1, 2, 3, 4]].logsumexp(-1)
                  - (scope_tags_logprobs[..., [0, 5, 6, 7, 8]].logsumexp(-1) if self.do_scope_holes else scope_tags_logprobs[..., 0])
            )

            scope_logits = self.scope_pooler(
                scope_tags_logits.unsqueeze(-1),
                (spans_begin.unsqueeze(1), spans_end.unsqueeze(1) + 1),
            )[..., 0]

        if self.do_biaffine:
            x = spans_positions[0][:, None, :] - spans_positions[1][:, :, None]
            y = spans_positions[1][:, None, :] - spans_positions[0][:, :, None]
            span_span_distance = torch.where(
                (x.sign() > 0) & (y.sign() > 0), torch.minimum(x, y),
                torch.where((x.sign() < 0) & (y.sign() < 0), torch.maximum(x, y), 0)
            )

            biaffine_logits = self.biaffine_proj(
                content_queries=spans_embed,
                content_keys=spans_embed,
                mask=spans_mask,
                relative_positions=span_span_distance,
            )[..., 0]

        if self.do_biaffine and self.do_scopes:
            if self.biaffine_mode == "and":
                combined_logits = log_and(scope_logits, biaffine_logits)
            elif self.biaffine_mode == "min":
                combined_logits = torch.min(scope_logits, biaffine_logits)
        elif self.do_scopes:
            combined_logits = scope_logits
        elif self.do_biaffine:
            combined_logits = biaffine_logits

        if supervision:
            spans_chunks = batch["fragments_chunks"]
            chunks_spans = batch["chunks_fragments"]

        same_entity_target = None
        if "chunk_relation" in supervision:
            same_entity_target = (
                  (spans_chunks.unsqueeze(1).unsqueeze(-1) == spans_chunks.unsqueeze(2).unsqueeze(-2)) &
                  (spans_chunks.unsqueeze(1).unsqueeze(-1) != -1) &
                  (spans_chunks.unsqueeze(2).unsqueeze(-2) != -1)
            ).any(-1).any(-1)

        relation_mask = spans_mask[:, :, None] & spans_mask[:, None, :]
        has_overlap = (spans_begin.unsqueeze(-1) <= spans_end.unsqueeze(-2)) & (spans_end.unsqueeze(-1) >= spans_begin.unsqueeze(-2))
        matching_logits = combined_logits.detach().masked_fill(~spans_mask.unsqueeze(-1), IMPOSSIBLE)

        ##################################
        # Fragment -> fragment relations #
        ##################################
        mydebug = None
        has_more_target_links = begin_targets = end_targets = hole_targets = pos_targets = scope_target = number_of_holes = None
        if "chunk_relation" in supervision:
            scope_loss_mask = (batch["has_chunks"][..., None] & spans_mask)[spans_mask]
            # allowed_a2h_relations: n_labels * n_labels

            # OLD
            # a2h_constraints = gather(allowed_a2h_relations[spans_label], index=spans_label.unsqueeze(1), dim=2) & relation_mask

            # NEW
            # spans_label: batch_size * n_spans * n_labels (bux) (bvy)
            # -> a2h_constraints: batch_size * n_spans * n_spans (xy)
            a2h_constraints = (torch.einsum(
                'buy,bvy->buv',
                (torch.einsum('bux,xy->buy', spans_label.float(), allowed_a2a2h_relations.any(-1).float())
                 != spans_label.float().sum(-1, keepdim=True)).float(),
                spans_label.float().float()
            ) == 0.) & relation_mask & (spans_label & allowed_a2a2h_relations.any(0).any(0)).any(-1)[:, None, :]

            symmetric_a2h_target = same_entity_target & (a2h_constraints | a2h_constraints.transpose(-1, -2))

            # two attributes can only be linked (and their relation penalized) if one is attached to a head that the other could be linked to
            a2a_constraints = torch.einsum(
                'nih,njh->nij',
                symmetric_a2h_target.float(),
                a2h_constraints.float()
            ).bool() & relation_mask & symmetric_a2h_target.any(-1).unsqueeze(-2)
            a2a_constraints = a2a_constraints | a2a_constraints.transpose(-1, -2)

            symmetric_a2a_target = a2a_constraints & same_entity_target
            symmetric_target = symmetric_a2h_target | symmetric_a2a_target
            symmetric_constraints = a2a_constraints | a2h_constraints | a2h_constraints.transpose(-1, -2)

            if self.do_target_links_count:
                target_links_count = (symmetric_constraints[..., :, None, :] & symmetric_constraints[..., None, :, :] & symmetric_target.unsqueeze(1)).sum(-1)
                has_more_target_links = (target_links_count.transpose(-1, -2) > target_links_count).masked_fill(~symmetric_target, 0).triu()
                matching_logits[has_more_target_links] += 10000.

                holes_per_pair = (
                      same_entity_target[..., None] &
                      (
                          # a < c & c < b
                            ((spans_end[:, :, None, None] < spans_begin[:, None, None, :]) & (spans_end[:, None, None, :] < spans_begin[:, None, :, None])) |
                            # a > c & c > b
                            ((spans_begin[:, :, None, None] > spans_end[:, None, None, :]) & (spans_begin[:, None, None, :] > spans_end[:, None, :, None]))
                      ) &
                      (~same_entity_target & symmetric_constraints)[:, :, None, :]  # c not in same chunk as a
                    # ~symmetric_target[:, :, :, None]  # c not in same chunk as b
                ).long().sum(-1)

                matching_logits[holes_per_pair < holes_per_pair.transpose(-1, -2)] += 1000

            matching_logits += torch.randn_like(matching_logits) * self.scope_match_temperature
            is_sup_than_inverse = multi_dim_triu(((matching_logits - matching_logits.transpose(-1, -2))).sigmoid() >= torch.rand_like(matching_logits))
            closest_target = (
                  is_sup_than_inverse |
                  multi_dim_triu(~is_sup_than_inverse).transpose(-1, -2) |
                  torch.eye(is_sup_than_inverse.shape[-1], dtype=torch.bool, device=matching_logits.device)
            )
            links_prediction = symmetric_target & closest_target

            if self.symmetric_training:
                chunk_relation_loss_data = {
                    "logits": torch.max(combined_logits, combined_logits.transpose(-1, -2), self.progress),
                    "valid_target": symmetric_target | symmetric_target.transpose(-1, -2),
                    "closest_target": links_prediction | links_prediction.transpose(-1, -2),
                    "mask": (symmetric_constraints & ~has_overlap) | (symmetric_constraints & ~has_overlap).transpose(-1, -2),
                }
                if self.do_scopes and self.scope_supervision == "mention":
                    scope_relation_loss_data = {
                        "logits": torch.max(scope_logits, scope_logits.transpose(-1, -2)),
                        "valid_target": symmetric_target | symmetric_target.transpose(-1, -2),
                        "closest_target": links_prediction | links_prediction.transpose(-1, -2),
                        "mask": (symmetric_constraints & ~has_overlap) | (symmetric_constraints & ~has_overlap).transpose(-1, -2),
                    }
            else:
                chunk_relation_loss_data = {
                    "logits": combined_logits,
                    "valid_target": symmetric_target,
                    "closest_target": links_prediction,
                    "mask": symmetric_constraints & ~has_overlap,
                }
                if self.do_scopes and self.scope_supervision == "mention":
                    scope_relation_loss_data = {
                        "logits": scope_logits,
                        "valid_target": symmetric_target,
                        "closest_target": links_prediction,
                        "mask": symmetric_constraints & ~has_overlap,
                    }
            res_dict.update(chunk_relation_loss_data)
            res_dict['chunk_relation_loss'] = (
                  self.chunk_relation_loss(**chunk_relation_loss_data).masked_fill(~batch["has_chunks"], 0.).sum()
                  + (0 if not self.do_scopes or self.scope_supervision != "mention" else self.chunk_relation_loss(**scope_relation_loss_data).masked_fill(~batch["has_chunks"], 0.).sum())
            )

            monolabel_spans_original_idx = None
            monolabel_spans_label = None
            monolabel_links_prediction = None
            if self.do_scopes and self.scope_supervision == "word":
                # Compute which fragment must be directly linked to which other fragment
                [linked_fragments], linked_fragments_mask = multi_dim_nonzero(
                    (links_prediction | torch.eye(same_entity_target.shape[-1], device=device, dtype=torch.bool)[None]) &
                    relation_mask,
                    dim=2)

                positions = torch.arange(words_mask.shape[1], device=device)
                spans_words_mask = (spans_begin[..., None] <= positions) & (spans_end[..., None] >= positions)

                # Compute the target scope tags where we can (will give a partial tag matrix)
                max_scope_begin = gather(spans_begin.unsqueeze(-2), linked_fragments, dim=2).masked_fill(~linked_fragments_mask, 1000000).min(
                    -1).values if 0 not in linked_fragments.shape else torch.zeros_like(spans_begin)
                min_scope_end = gather(spans_end.unsqueeze(-2), linked_fragments, dim=2).masked_fill(~linked_fragments_mask, -1000000).max(
                    -1).values if 0 not in linked_fragments.shape else torch.zeros_like(spans_end)

                [hole_fragments], hole_fragments_mask = multi_dim_nonzero(
                    (~symmetric_target) & symmetric_constraints & ~has_overlap &
                    (spans_begin[..., None, :] >= max_scope_begin[..., :, None]) &
                    (spans_end[..., None, :] <= min_scope_end[..., :, None]),
                    dim=2)
                [not_linked_fragments], not_linked_fragments_mask = multi_dim_nonzero(
                    (~symmetric_target) & symmetric_constraints & ~has_overlap &
                    relation_mask,
                    dim=2)
                # fragments that are outside a given scope => model must not NOT put O tag on them
                # fragments that are linked to a given fragment => model must not NOT put I, B, L or U tag on them
                must_be_link = (
                    gather(spans_words_mask.unsqueeze(-3), linked_fragments[..., None], dim=-2).masked_fill(~linked_fragments_mask[..., None], 0).any(-2)[..., None]
                )
                must_not_be_in_scope = (
                                           gather(spans_words_mask.unsqueeze(-3), not_linked_fragments[..., None], dim=-2).masked_fill(~not_linked_fragments_mask[..., None], 0).any(-2)[..., None]
                                       ) & ~must_be_link
                must_be_hole = (
                      gather(spans_words_mask.unsqueeze(-3), hole_fragments[..., None], dim=-2).masked_fill(~hole_fragments_mask[..., None], 0).any(-2)[..., None] &
                      ~must_be_link.any(-1, keepdim=True)
                )
                # must_not_be_in_scope = must_not_be_in_scope & ~must_be_hole
                must_be_hole_bef = must_be_hole
                # if not self.do_scope_holes:
                #    must_be_hole = torch.zeros_like(must_be_hole)

                must_not_be_begin = (positions > max_scope_begin[..., None])[..., None]
                must_not_be_end = (positions < min_scope_end[..., None])[..., None]
                must_not_be_out = (positions >= max_scope_begin[..., None])[..., None] & (positions <= min_scope_end[..., None])[..., None]
                scope_target = ~(
                      (torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1], device=device, dtype=torch.bool) & (must_not_be_in_scope & ~must_not_be_out))
                      | (~torch.tensor([0, 1, 1, 1, 1, 0, 0, 0, 0], device=device, dtype=torch.bool) & must_be_link)
                      | (~torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], device=device, dtype=torch.bool) & ((must_not_be_in_scope & ~must_not_be_out) if self.do_scope_holes else False))
                      | (torch.tensor([0, 0, 1, 0, 1, 0, 0, 0, 0], device=device, dtype=torch.bool) & must_not_be_begin)
                      | (torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 0], device=device, dtype=torch.bool) & must_not_be_end)
                      | (torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0], device=device, dtype=torch.bool) & must_not_be_out)
                )

                mydebug = torch.stack([
                    must_not_be_in_scope,
                    must_be_link,
                    must_be_hole_bef,
                    must_not_be_begin,
                    must_not_be_end,
                    must_not_be_out,
                ], dim=-1)
                # ~(linked_forbidden | begin_forbidden | end_forbidden | outside_forbidden | hole_forbidden)

                # Compute the scope loss with CRF forward backward algorithm
                if self.scope_supervision == "word":
                    res_dict['scope_tag_loss'] = -self.scope_decoder(
                        scope_emissions,
                        scope_mask,
                        scope_target[spans_mask][..., :(9 if self.do_scope_holes else 5)]
                    )[scope_loss_mask].sum() * self.scope_tag_loss_weight
                else:
                    res_dict['scope_tag_loss'] = 0.

                # if res_dict['scope_tag_loss'] > 10000:

                if self.do_scope_holes:
                    # Infer how many holes does a scope have at least
                    has_scope_info = (must_be_link | must_not_be_in_scope | must_be_hole).squeeze(-1)
                    x = torch.zeros(*scope_target.shape[:-2], int(has_scope_info.sum(-1).max()), dtype=torch.bool, device=device)
                    x[torch.arange(x.shape[-1], device=device) < has_scope_info.sum(-1)[..., None]] = must_be_hole.squeeze(-1)[has_scope_info]
                    number_of_holes = (x != shift(x, dim=-1, n=1)).sum(-1)[spans_mask] // 2
                    # How many more holes did the net predict over the estimated required holes ?
                    holes_delta = (
                        scope_tags_logprobs[..., [6, 8]].logsumexp(-1).masked_fill(~scope_mask, IMPOSSIBLE).logsumexp(-1).exp() - number_of_holes,
                        scope_tags_logprobs[..., [7, 8]].logsumexp(-1).masked_fill(~scope_mask, IMPOSSIBLE).logsumexp(-1).exp() - number_of_holes
                    )
                    # holes_delta : > 0 if missing holes, < 0 if too much holes
                    res_dict["holes_delta"] = -holes_delta[0][scope_loss_mask].sum() - holes_delta[1][scope_loss_mask].sum()

                # Compute the contiguity loss: avoid having more holes than necessary, and try to have crisp scope boundaries
            else:
                res_dict["scope_tag_loss"] = 0.
        else:
            # a2h_constraints = gather(allowed_a2h_relations[spans_label], index=spans_label.unsqueeze(1), dim=2) & relation_mask

            a2h_constraints = (torch.einsum(
                'buy,bvy->buv',
                (torch.einsum('bux,xy->buy', spans_label.float(), allowed_a2a2h_relations.any(-1).float())
                 != spans_label.float().sum(-1, keepdim=True)).float(),
                spans_label.float().float()
            ) == 0.) & relation_mask & (spans_label & allowed_a2a2h_relations.any(0).any(0)).any(-1)[:, None, :]

            a2h_constraints = a2h_constraints | a2h_constraints.transpose(-1, -2)
            a2h_prediction = ((combined_logits > 0) | (combined_logits.transpose(-1, -2) > 0) | has_overlap) & a2h_constraints

            # OLD
            # a2a2h_constraints = torch.einsum(
            #    'nih,njh,nhk->nijk',
            #    a2h_prediction.float(),
            #    a2h_prediction.float(),
            #    F.one_hot(spans_label, num_classes=allowed_a2a2h_relations.shape[-1]).float()
            # ).bool() if 0 not in a2h_prediction.shape else a2h_prediction[..., None] & allowed_a2h_relations.any(-1)
            # a2a_constraints = (
            #                        gather(allowed_a2a2h_relations[spans_label], index=spans_label.unsqueeze(1).unsqueeze(-1), dim=2)  # att 2 att 2 allowed head label
            #                        & a2a2h_constraints
            #                  ).any(-1) & relation_mask

            # a2a2h_constraints: batch_size * n_mention_u * n_mention_v * shared_head_labels
            a2a2h_constraints = torch.einsum(
                'buh,bvh,bhl->buvl',
                a2h_prediction.float(),  # is there a link between mention u to the trigger h
                a2h_prediction.float(),  # is there a link between mention v to the trigger h
                spans_label.float(),  # what are the labels of trigger h
            ).bool() if 0 not in a2h_prediction.shape else a2h_prediction[..., None] & allowed_a2h_relations.any(-1)

            a2a_constraints = ((torch.einsum(
                'buyh,bvy->buvh',
                (torch.einsum('bux,xyh->buyh', spans_label.float(), allowed_a2a2h_relations.float())
                 != spans_label.float().sum(-1, keepdim=True).unsqueeze(-1)).float(),
                spans_label.float()
            ) == 0.) & a2a2h_constraints).any(-1) & relation_mask

            a2a_prediction = ((combined_logits > 0) | (combined_logits.transpose(-1, -2) > 0) | has_overlap) & a2a_constraints
            links_prediction = a2a_prediction | a2h_prediction
            links_prediction = links_prediction | links_prediction.transpose(-1, -2)

            (monolabel_spans_original_idx, monolabel_spans_label), monolabel_spans_mask = multi_dim_nonzero(spans_label & spans_mask[..., None], dim=-2)

            monolabel_links_prediction = smart_gather(
                "sample fragment_i fragment_j, sample @fragment_i -> sample @fragment_i fragment_j",
                links_prediction,
                monolabel_spans_original_idx,
            )
            monolabel_links_prediction = smart_gather(
                "sample fragment_i fragment_j, sample @fragment_j -> sample fragment_i @fragment_j",
                monolabel_links_prediction,
                monolabel_spans_original_idx,
            )
            monolabel_links_prediction = monolabel_links_prediction & smart_gather(
                "sample fragment_i allowed_attributes, sample @allowed_attributes -> sample fragment_i @allowed_attributes",
                allowed_a2a2h_relations.any(-1)[monolabel_spans_label],
                monolabel_spans_label
            ) & monolabel_spans_mask[..., None, :] & monolabel_spans_mask[..., :, None]

            chunks_monolabel_spans = get_cliques(
                monolabel_links_prediction,
                mask=monolabel_spans_mask & (monolabel_links_prediction.any(-1) | monolabel_links_prediction.any(-2)),
                must_contain=allowed_a2a2h_relations.any(0).any(0)[monolabel_spans_label]
            )

            [chunks_spans], chunks_spans_mask = multi_dim_nonzero(smart_gather(
                "sample monolabel_fragment original_fragment_i, sample chunk @monolabel_fragment -> sample chunk @monolabel_fragment original_fragment_i",
                F.one_hot(monolabel_spans_original_idx, num_classes=n_spans).bool(),
                chunks_monolabel_spans
            ).masked_fill(chunks_monolabel_spans.unsqueeze(-1) == -1, False).any(-2), -1)
            chunks_spans[~chunks_spans_mask] = -1
            supported_chunks_labels = (
                smart_gather("sample fragment label, sample chunk @fragment -> sample chunk @fragment label",
                             F.one_hot(monolabel_spans_label, num_classes=allowed_a2a2h_relations.shape[0]).bool(),
                             chunks_monolabel_spans)
                    .masked_fill(chunks_monolabel_spans.unsqueeze(-1) == -1, 0)
                    .bool().any(-2)
            )

        #########################################
        #        CHUNK LABELS PREDICTION        #
        #########################################

        if self.label_proj_pooling == "embed-max":
            label_logits = F.relu(self.pre_label_proj(self.dropout(spans_embed))[  # n_samples * n_spans * n_labels
                                      sample_index.unsqueeze(-1),  # n_samples * n_entities * n_spans
                                      chunks_spans  # n_samples * n_entities * n_spans
                                  ]).masked_fill(chunks_spans.unsqueeze(-1) == -1, IMPOSSIBLE)
            label_logits = self.label_proj(label_logits.max(-2).values if 0 not in label_logits.shape else label_logits.sum(-2))
        else:
            label_logits = self.label_proj(spans_embed)[  # n_samples * n_spans * n_labels
                sample_index.unsqueeze(-1),  # n_samples * n_entities * n_spans
                chunks_spans  # n_samples * n_entities * n_spans
            ].masked_fill(chunks_spans.unsqueeze(-1) == -1, IMPOSSIBLE)
            if self.label_proj_pooling == "max":
                label_logits = label_logits.max(-2).values if 0 not in label_logits.shape else label_logits.sum(-2)
            elif self.label_proj_pooling == "lse":
                label_logits = label_logits.logsumexp(-2)

        if "chunk_label" in supervision:
            chunks_labels = batch['chunks_labels']
            if self.do_constraints != "test-only":
                output_logits = F.linear(label_logits, self.constrained_outputs)
                required_outputs = torch.einsum('ol,ncl->nco', self.constrained_outputs, chunks_labels.float()) >= chunks_labels.float().sum(-1, keepdim=True)
                # all outputs that have at most all labels from the allowed outputs
                forbidden_outputs = torch.einsum('ol,ncl->nco', self.constrained_outputs, (~batch['chunks_allowed_labels']).float()) > 0
                targets = (required_outputs & ~forbidden_outputs)
                assert (targets.long().sum(-1) > 0)[batch['chunks_mask']].all()
                loss = (-output_logits
                        .log_softmax(-1)
                        .masked_fill(~targets, IMPOSSIBLE)
                        .logsumexp(-1)) * self.label_loss_weight
                res_dict["chunk_label_loss"] = loss.masked_fill(~batch['chunks_mask'], 0).sum(-1).masked_fill(~batch["has_chunks"], 0.).sum(-1).sum()
            else:
                # ncl, ncl -> ncl
                loss = bce_with_logits(label_logits, chunks_labels, reduce='none')  # .masked_fill(target_labels | ~required_labels)
                loss = loss.sum(-1) * self.label_loss_weight
                res_dict["chunk_label_loss"] = loss.masked_fill(~batch['chunks_mask'], 0).sum(-1).masked_fill(~batch["has_chunks"], 0.).sum(-1).sum()
        else:
            if True:  # self.force_fragment_labels:
                label_logits = label_logits.masked_fill(supported_chunks_labels, REQUIRED)
            chunks_labels = self.constrained_outputs[
                F.linear(label_logits.detach(), self.constrained_outputs).argmax(-1)
            ].bool().cpu() if self.do_constraints and self.constrained_outputs is not None and 0 not in label_logits.shape else label_logits.masked_fill(supported_chunks_labels, REQUIRED) > 0

        res_dict.update({
            "logits": combined_logits,
            "matching_logits": matching_logits,
            "links_prediction": links_prediction,
            "has_more_target_links": has_more_target_links,
            "chunks_spans": chunks_spans,
            "chunks_labels": chunks_labels,
            "a2h_constraints": a2h_constraints,
            "a2a_constraints": a2a_constraints,
            "mydebug": mydebug,

            "scope_logits": scope_tags_logits,
            "scope_tags_logprobs": scope_tags_logprobs,
            "scope_bounds_emissions": scope_bounds_emissions,
            "scope_emissions": scope_emissions,

            "same_entity_target": same_entity_target,
            "begin_targets": begin_targets,
            "end_targets": end_targets,
            "hole_targets": hole_targets,
            "pos_targets": pos_targets,
            "scope_target": scope_target,
            "number_of_holes": number_of_holes,

            "spans_label": spans_label,
            "monolabel_spans_original_idx": monolabel_spans_original_idx,
            "monolabel_spans_label": monolabel_spans_label,
            "monolabel_links_prediction": monolabel_links_prediction,
        })
        return res_dict


@register("bert")
class BERTEncoder(TextEncoder):
    ENSEMBLE = "ensemble_text_encoder"

    def __init__(self,
                 _bert=None,
                 bert_config=None,
                 path=None,
                 n_layers=1,
                 combine_mode="softmax",
                 bert_dropout_p=None,
                 output_lm_embeds=False,
                 token_dropout_p=0.,
                 dropout_p=0.,
                 word_pooler={"module": "pooler", "mode": "mean"},
                 proj_size=None,
                 freeze_n_layers=-1,
                 do_norm=True,
                 do_cache=False,
                 _preprocessor=None, ):
        super().__init__()
        assert not ("scaled" in combine_mode and do_norm)
        if do_cache:
            assert freeze_n_layers == -1, "Must freeze bert to enable caching: set freeze_n_layers=-1"

        with fork_rng(True):
            if output_lm_embeds:
                self.bert = _bert if _bert is not None else transformers.AutoModelForMaskedLM.from_pretrained(path, config=bert_config)
                if hasattr(self.bert, 'lm_head'):
                    self.bert.lm_head.__class__ = LM_HEAD_CLS_MAPPING[self.bert.lm_head.__class__]
                else:
                    self.bert.cls.predictions.__class__ = LM_HEAD_CLS_MAPPING[self.bert.cls.predictions.__class__]
            else:
                self.bert = _bert if _bert is not None else transformers.AutoModel.from_pretrained(path, config=bert_config)
        self.output_lm_embeds = output_lm_embeds
        self.n_layers = n_layers
        if n_layers > 1:
            with fork_rng(True):
                self.weight = torch.nn.Parameter(torch.zeros(n_layers)) if "softmax" in combine_mode else torch.nn.Parameter(torch.ones(n_layers) / n_layers) if combine_mode == "linear" else None
        with fork_rng(True):
            self.word_pooler = Pooler(**word_pooler) if word_pooler is not None else None
        if "scaled" in combine_mode:
            self.bert_scaling = torch.nn.Parameter(torch.ones(()))
        self.combine_mode = combine_mode

        bert_model = self.bert.bert if hasattr(self.bert, 'bert') else self.bert.roberta if hasattr(self.bert, 'roberta') else self.bert
        bert_output_size = bert_model.embeddings.word_embeddings.weight.shape[1] * (1 if combine_mode != "concat" else n_layers)
        self.bert_output_size = bert_output_size
        if proj_size is not None:
            self.proj = torch.nn.Linear(bert_output_size, proj_size)
            self._output_size = proj_size
        else:
            self.proj = None
            self._output_size = bert_output_size
        self.norm = torch.nn.LayerNorm(self._output_size) if do_norm else Identity()

        if freeze_n_layers < 0:
            freeze_n_layers = len(bert_model.encoder.layer) + 2 + freeze_n_layers
        for module in (bert_model.embeddings, *bert_model.encoder.layer)[:freeze_n_layers]:
            for param in module.parameters():
                param.requires_grad = False
        if bert_dropout_p is not None:
            for module in bert_model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = bert_dropout_p
        self.dropout = torch.nn.Dropout(dropout_p)
        self.token_dropout_p = token_dropout_p
        self.cache = {}
        self.do_cache = do_cache

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Do not save bert if it was frozen. We assume that the freeze_n_layers == -1 was not
        # changed during training, and therefore that the weights are identical to those at init
        state = super().state_dict(destination, prefix, keep_vars)
        if self.freeze_n_layers == -1:
            for name in list(state.keys()):
                if ".bert." in name:
                    del state[name]
        return state

    @property
    def output_embeddings(self):
        bert = self.bert  # .bert if hasattr(self.bert, 'bert') else self.bert.roberta
        return bert.lm_head.decoder.weight if hasattr(bert, 'lm_head') else bert.cls.predictions.weight

    @property
    def output_bias(self):
        bert = self.bert  # .bert if hasattr(self.bert, 'bert') else self.bert.roberta
        return bert.lm_head.decoder.bias if hasattr(bert, 'lm_head') else bert.cls.predictions.bias

    @property
    def bert_config(self):
        return self.bert.config

    def exec_bert(self, tokens, mask, slices_begin=None, slices_end=None):
        res = self.bert.forward(tokens, mask, output_hidden_states=True)
        if self.output_lm_embeds:
            lm_embeds = list(res.logits)
            token_features = res.hidden_states
        else:
            lm_embeds = ()
            token_features = res[2]
        if self.n_layers == 1:
            token_features = token_features[-1].unsqueeze(-2)
        else:
            token_features = torch.stack(token_features[-self.n_layers:], dim=2)

        results = (token_features, *lm_embeds)
        if slices_begin is not None:
            results = multi_dim_slice(results, slices_begin, slices_end)
        return results

    def forward(self, batch):
        tokens, mask = batch["tokens"], batch["tokens_mask"]
        device = mask.device
        if self.training & (self.token_dropout_p > 0):
            tokens[mask & (torch.rand_like(mask, dtype=torch.float) < self.token_dropout_p)] = 32004  # self.bert.config.mask_token_id
        if tokens.ndim == 3 and tokens.shape[1] == 1:
            flat_tokens = tokens.squeeze(1)
            flat_mask = mask.squeeze(1)
            flat_slices_begin = batch['slice_begin'].squeeze(1) if 'slice_begin' in batch else torch.zeros(len(flat_mask), device=device, dtype=torch.long)
            flat_slices_end = batch['slice_end'].squeeze(1) if 'slice_end' in batch else flat_mask.long().sum(1)
            concat_mask = None
        elif tokens.ndim == 3:
            flat_tokens = tokens[mask.any(-1)]
            flat_mask = mask[mask.any(-1)]
            flat_slices_begin = batch['slice_begin'][mask.any(-1)] if 'slice_begin' in batch else torch.zeros(len(flat_mask), device=device, dtype=torch.long)
            flat_slices_end = batch['slice_end'][mask.any(-1)] if 'slice_end' in batch else flat_mask.long().sum(1)
            lengths = (batch['slice_end'] - batch['slice_begin']) if 'slice_begin' in batch else mask.sum(-1)
            concat_mask = (torch.arange(lengths.max(), device=device) < lengths.unsqueeze(-1)) & mask.any(-1, keepdim=True)
        else:
            concat_mask = None
            flat_tokens = tokens
            flat_mask = mask
            flat_slices_begin = batch['slice_begin'] if 'slice_begin' in batch else torch.zeros(len(flat_mask), device=device, dtype=torch.long)
            flat_slices_end = batch['slice_end'] if 'slice_end' in batch else flat_mask.long().sum(1)
        if self.do_cache:
            keys = [hash((tuple(row[:length]), begin, end)) for row, length, begin, end in zip(flat_tokens.tolist(), flat_mask.sum(1).tolist(), flat_slices_begin.tolist(), flat_slices_end.tolist())]

            missing_keys = [key for key in keys if key not in self.cache]
            missing_keys_mask = [key in missing_keys for key in keys]
            if sum(missing_keys_mask) > 0:
                missing_embeds = self.exec_bert(
                    flat_tokens[missing_keys_mask],
                    flat_mask[missing_keys_mask],
                    slices_begin=flat_slices_begin[missing_keys_mask] if flat_slices_begin is not None else None,
                    slices_end=flat_slices_end[missing_keys_mask] if flat_slices_end is not None else None)
                cache_entries = [tuple(t[:length] for t in tensor_set)
                                 for tensor_set, length in zip(zip(*(t.cpu().unbind(0) for t in missing_embeds)), flat_mask[missing_keys_mask].sum(-1).tolist())]
                for key, cache_entry in zip(missing_keys, cache_entries):
                    self.cache[key] = cache_entry
                    if (len(self.cache) % 1000) == 0:
                        print("cache size:", len(self.cache))

            if sum(missing_keys_mask) == len(missing_keys_mask):
                (token_features, *lm_embeds) = missing_embeds
            else:
                (token_features, *lm_embeds) = (pad_embeds(embeds_list).to(device) for embeds_list in zip(*[self.cache[key] for key in keys]))
        else:
            (token_features, *lm_embeds) = self.exec_bert(
                flat_tokens,
                flat_mask,
                slices_begin=flat_slices_begin,
                slices_end=flat_slices_end)

        if self.n_layers == 1:
            token_features = token_features.squeeze(-2)
        elif self.combine_mode != "concat":
            token_features = torch.einsum("stld,l->std", token_features, self.weight.softmax(-1) if "softmax" in self.combine_mode else self.weight)
        else:
            token_features = token_features.view(*token_features.shape[:-2], -1)

        if hasattr(self, 'bert_scaling'):
            token_features = token_features * self.bert_scaling

        token_features = self.dropout(token_features)

        if self.proj is not None:
            token_features = F.gelu(self.proj(token_features))

        token_features = self.norm(token_features)

        if concat_mask is not None and len(lm_embeds) > 0:
            token_features, lm_embeds[0], lm_embeds[1] = rearrange_and_prune([token_features, lm_embeds[0], lm_embeds[1]], concat_mask)[0]
        elif concat_mask is not None and len(lm_embeds) == 0:
            token_features, = rearrange_and_prune([token_features], concat_mask)[0]
        # assert (self.word_pooler(token_features, (batch["words_bert_begin"], batch["words_bert_end"]))[mask.any(1)] == token_features[mask.any(1)]).all()
        # assert (self.word_pooler(lm_embeds, (batch["words_bert_begin"], batch["words_bert_end"]))[mask.any(1)] == lm_embeds[mask.any(1)]).all()
        if self.word_pooler is not None:
            if token_features is not None:
                token_features = self.word_pooler(token_features, (batch["words_bert_begin"], batch["words_bert_end"]))
            if len(lm_embeds) > 0:
                lm_embeds = tuple((
                    self.word_pooler(part, (batch["words_bert_begin"], batch["words_bert_end"]))
                    for part in lm_embeds
                ))

        if self.output_lm_embeds:
            return token_features, lm_embeds
        return token_features
