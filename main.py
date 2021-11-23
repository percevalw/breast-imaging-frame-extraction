import inspect
import string

import fire
import pytorch_lightning as pl
import termcolor
from rich_logger import RichTableLogger

from data import *
from modules import *
from nlstruct.base import InformationExtractor, load_pretrained
from nlstruct.checkpoint import ModelCheckpoint
from preprocess import *


logger_fields = {
    "epoch": {},
    "step": {},

    "chunk_label_loss": {"name": "label_loss"},
    "chunk_relation_loss": {"name": "rel_loss"},
    "scope_tag_loss": {"name": "scope_loss"},

    "(.*)_?loss": {"goal": "lower_is_better", "format": "{:.4f}"},
    "(.*)_(precision|recall|tp)": False,
    "(.*)_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_f1"},
    ".*(precision|recall|f1)": {},
    ".*_lr": {"format": "{:.2e}"},
    "duration": {"format": "{:.0f}", "name": "dur(s)"},
}


def main(
      seed=42,
      max_steps=2000,
      val_check=None,
      augmentation=("sentencize",),
      doc_context=True,

      main_lr=5e-4,
      bert_lr=5e-5,
      batch_size=16,
      encoder_dropout_p=0.5,
      encoder_n_layers=1,
      lower_text=True,
      do_lm_embeds=False,

      scope_match_temperature=1.,
      label_proj_pooling="max",
      hidden_size=400,
      dropout_p=0.2,
      n_heads=8,
      head_size=64,
      max_relative_position=80,
      relation_relative_position_mode=('c2c', 'c2p', 'p2c'),
      bert_name="camembert-base",
      span_pooling="mean",
      span_layers=0,
      lstm_layers=3,
      do_scope_holes=False,
      do_span_on_span_attention=True,
      do_span_on_word_attention=True,
      do_word_on_span_attention=False,
      subset=1.,
      do_constraints=True,

      do_label_embedding=True,
      scope_supervision="word",
      symmetric_training=False,
      chunk_loss_weight=1,
      label_loss_weight=0.5,
      do_relation_heuristics=True,
      lstm_gate_reference="input",
      transformer_gate_reference="input",
      mode="biaffine_min_scope",
      xp_name="xp-name",
):
    if val_check is None:
        val_check = max_steps // 10

    augmentation = tuple(sorted((augmentation,) if isinstance(augmentation, str) else augmentation))

    defaults = {
        k: v.default
        for k, v in inspect.signature(main).parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    for key, value in locals().items():
        if value is not defaults:
            print(key.ljust(40), termcolor.colored(str(value), 'red') if key in defaults and value != defaults[key] else str(value))

    models = []
    results = []

    try:
        memory_model = load_pretrained("mem.pt")
    except:
        memory_model = None

    base_train_data = FrameDataset(
        train='path/to/train',
        prep_fn=compose(preprocess_frame_annotations),
    ).train_data
    if subset != 1.:
        with fork_rng(seed):
            base_train_data = permute(base_train_data)[:int(subset * len(base_train_data))]
            print("SUBSET SIZE", len(base_train_data))
    sentencized_train_data = list(sentencize_corpus(base_train_data, n=5, yield_rate=0.5, chain=True))
    test_data = FrameDataset(
        val='path/to/test',
        prep_fn=preprocess_frame_annotations,
    )

    dataset = MixDataset(
        ([
             BaseDataset(base_train_data, None, None)
         ] if len(base_train_data) > 0 else []) +
        ([
             BaseDataset(sentencized_train_data, None, None),
         ] if "sentencize" in augmentation and len(base_train_data) > 0 else []) +
        ([
             BaseDataset(synthetic_data, None, None),
         ] if "lexicon_sentences" in augmentation else []) +
        [test_data],
        rates=[
            *([10] if len(base_train_data) > 0 else []),
            *([10] if "sentencize" in augmentation and len(base_train_data) > 0 else []),
            *([10] if "lexicon_sentences" in augmentation else []),
            0
        ])

    finetune_bert = True
    model = InformationExtractor(
        seed=seed,

        preprocessor=dict(
            module="frame_preprocessor",
            bert_name=bert_name,  # transformer name
            bert_lower=lower_text,
            split_into_multiple_samples=False,
            sentence_split_regex=r'(?<=(?:\\s*\\n+\\s*)|(?:(?<=[a-zA-Z0-9)]\\.)\\s+))()(?=[A-Z+-=])',  # regex to use to split sentences (must not contain consuming patterns)
            sentence_balance_chars=('()',),  # try to avoid splitting between parentheses
            sentence_entity_overlap="raise",  # raise when an entity spans more than one sentence
            word_regex=r'(?:[\w]+(?:[’\'])?)|<[?-]+>|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]+',
            # regex to use to extract words (will be aligned with bert tokens), leave to None to use wordpieces as is
            substitutions=(  # Apply these regex substitutions on sentences before tokenizing
                ('([Ll]es donn.es administratives, sociales.*?opposition[.])', ' disclaimer '),
                ('={4,}', ' ==== '),
                ('-{4,}', ' ==== '),
                ('_{4,}', ' ==== '),
                (re.escape('<??-??-????>'), 'dd/mm/yyyy'),
                (re.escape('<????-??-??>'), 'yyyy/mm/dd'),
                (r"(?<=[{}\\])(?![ ])".format(string.punctuation), r" "),  # insert a space before punctuations
                (r"(?<![ ])(?=[{}\\])".format(string.punctuation), r" "),  # insert a space after punctuations
                (r"\n", ' _ '),
                ("(?<=[a-zA-Z])(?=[0-9])", r" "),  # insert a space between letters and numbers
                ("(?<=[0-9])(?=[A-Za-z])", r" "),  # insert a space between numbers and letters
            ),

            min_tokens=32,
            join_small_sentence_rate=0.1,
            max_tokens=192,  # split when sentences contain more than 512 tokens
            large_sentences="equal-split",  # for these large sentences, split them in equal sub sentences < 512 tokens
            empty_entities="raise",  # when an entity cannot be mapped to any word, raise
            vocabularies={  # vocabularies to use, call .train() before initializing to fill/complete them automatically from training data
                # "char": dict(module="vocabulary", values=string.punctuation + string.ascii_letters + string.digits, with_unk=True, with_pad=True),
                "fragment_label": dict(module="vocabulary", values=sorted(set(ner_to_multilabel.keys())), with_unk=False, with_pad=False),
                "entity_label": dict(module="vocabulary", values=sorted(set(abbreviations.keys())), with_unk=False, with_pad=False),
            },
            doc_context=doc_context,
            multi_label=True,
            fragment_label_is_entity_label=False,
            filter_entities=None,  # ("entity_type_proc_diag",),#, "entity_type_score_lesion"),
        ),
        dynamic_preprocessing=True,

        encoder=dict(
            module="concat",
            dropout_p=encoder_dropout_p,
            encoders=[
                dict(
                    module="bert",
                    path=bert_name,
                    n_layers=1,
                    freeze_n_layers=0 if finetune_bert is not False else -1,  # freeze 0 layer (including the first embedding layer)
                    bert_dropout_p=None if finetune_bert else 0.,
                    token_dropout_p=0.,
                    output_lm_embeds=do_lm_embeds,
                    combine_mode="scaled_softmax",
                    do_norm=False,
                    do_cache=not finetune_bert,
                    word_pooler=dict(module="pooler", mode="mean"),
                ),
            ],
        ),
        decoder=dict(
            module="entity_decoder",
            contextualizer=dict(
                module="lstm",
                num_layers=lstm_layers,
                hidden_size=hidden_size,
                gate=dict(module="residual_gate", init_value=1., ln_mode="post"),
                bidirectional=True,
                dropout_p=dropout_p,
                gate_reference=lstm_gate_reference,
            ),

            span_scorer=dict(
                module="bitag",
                max_fragments_count=200,
                max_length=20,
                do_tagging="shared_label_unary",
                do_biaffine=False,
                hidden_size=None,
                allow_overlap=False,
                tag_loss_weight=2.,
                learnable_transitions=False,
                eps=1e-8,
            ),
            head_labels=("entity_type_score_acr", "entity_type_score_density", "entity_type_lesion", "entity_type_proc_diag", "entity_type_proc_ther"),
            constrained_outputs=enum_outputs,
            use_sentence_coordinates=False,

            chunk_decoder=dict(
                scope_head_size=hidden_size,
                scope_match_temperature=scope_match_temperature,
                do_target_links_count=do_relation_heuristics,
                do_biaffine="biaffine" in mode,
                do_scopes="scope" in mode,
                dropout_p=0.,
                do_constraints=do_constraints,
                scope_crf_params=dict(allow_hole_outside_scope=False),

                biaffine_mode="and" if "and" in mode else "min" if "min" in mode else "smoothmin" if "smoothmin" in mode else "sum",
                do_scope_holes=do_scope_holes,
                attention_mode=relation_relative_position_mode,

                chunk_relation_loss_weight=chunk_loss_weight,
                scope_tag_loss_weight=chunk_loss_weight,
                label_proj_pooling=label_proj_pooling,
                label_loss_weight=label_loss_weight,

                scope_supervision=scope_supervision,
                symmetric_training=symmetric_training,
            ),
            span_embedding=dict(mode=span_pooling, label_dropout_p=dropout_p, do_label_embedding=do_label_embedding),
            normalizer_scheme=ner_to_multilabel,
            dropout_p=dropout_p,
            pos_size=head_size,
            n_heads=n_heads,
            head_size=head_size,
            max_relative_position=max_relative_position,
            relation_relative_position_mode=relation_relative_position_mode,
        ),
        batch_size=batch_size,

        # Use learning rate schedules (linearly decay with warmup)
        use_lr_schedules=True,
        warmup_rate=0.1,

        gradient_clip_val=50.,

        # Learning rates
        main_lr=main_lr,
        fast_lr=main_lr,
        bert_lr=bert_lr,

        # Optimizer, can be class or str
        optimizer_cls="transformers.AdamW",
        memory_model={
            "min_fit_batch_size": 2,
            "max_fit_batch_size": 8,
            "max_steps_per_batch_size": 25,
            "max_memory": 22,
        } if memory_model is None else memory_model,
        additional_hparams={"augmentation": augmentation, **({} if subset == 1. else {"subset": subset})},
        metrics={
            "span": dict(module="ner_metric", threshold=0.5),
            "chunk": dict(module="frame_metric", only_trigger=False, binarize_tag_threshold=False, binarize_label_threshold=0., joint_matching=True),
            "label": dict(module="frame_metric", only_trigger=True, binarize_tag_threshold=1e-14, binarize_label_threshold=1, joint_matching=True),
        },
    ).train()

    print("SEED =", seed)
    trainer = pl.Trainer(val_check_interval=val_check, max_steps=max_steps, gpus=1,
                         progress_bar_refresh_rate=False, checkpoint_callback=False,  # do not make checkpoints since it slows down the training a lot
                         callbacks=[ModelCheckpoint(path='checkpoints/' + xp_name + '-{hashkey}-{global_step:05d}', only_last=True)],
                         logger=[RichTableLogger(key="epoch", fields=logger_fields)])
    model.trainer = trainer
    trainer.fit(model, dataset)

    return model


if __name__ == "__main__":
    fire.Fire(main)
