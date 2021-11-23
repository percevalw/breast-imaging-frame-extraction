import math
import re
from collections import defaultdict
from itertools import combinations

import numpy as np

from nlstruct.data_utils import regex_multisub_with_spans
from tools import *

abbreviations = {'entity_type_lesion': 'lesion',
                 'entity_type_proc_diag': 'diag',
                 'entity_type_proc_ther': 'ther',
                 'entity_type_score_acr': 'acr',
                 'entity_type_score_density': 'density',

                 'proc_diag_biopsy': 'biopsy',
                 'proc_diag_echography': 'echo',
                 'proc_diag_irm': 'irm',
                 'proc_diag_mammography': 'mammo',
                 'proc_diag_other': 'diag_other',
                 'proc_diag_palpation': 'palpation',
                 'proc_ther_surgery': 'surgery',
                 'proc_ther_other': 'ther_other',
                 'score_acr_0': 'acr_0',
                 'score_acr_1': 'acr_1',
                 'score_acr_2': 'acr_2',
                 'score_acr_3': 'acr_3',
                 'score_acr_4': 'acr_4',
                 'score_acr_4a': 'acr_4a',
                 'score_acr_4b': 'acr_4b',
                 'score_acr_4c': 'acr_4c',
                 'score_acr_5': 'acr_5',
                 'score_acr_6': 'acr_6',
                 'score_density_1': 'dens_1',
                 'score_density_2': 'dens_2',
                 'score_density_3': 'dens_3',
                 'score_density_4': 'dens_4',

                 'location_quadrant_areo': 'areo',
                 'location_quadrant_axi': 'axi',
                 'location_quadrant_qie': 'qie',
                 'location_quadrant_qii': 'qii',
                 'location_quadrant_qse': 'qse',
                 'location_quadrant_qsi': 'qsi',
                 'location_quadrant_uqext': 'uqext',
                 'location_quadrant_uqinf': 'uqinf',
                 'location_quadrant_uqint': 'uqint',
                 'location_quadrant_uqsup': 'uqsup',

                 # 'is_now_and_passed': 'now+passed',
                 'is_now': 'now',
                 'is_future': 'future',
                 'is_passed': 'passed',
                 'location_body_part_breast': 'breast',
                 'location_body_part_other': 'body_other',
                 # 'location_laterality_bilateral': 'left+right',
                 'location_laterality_left': 'left',
                 'location_laterality_right': 'right',

                 'has_angle': 'angle',
                 'has_distance': 'distance',
                 'has_size': 'size', }

terminology = [
    "à [droite](location_laterality_right)",
    "[droite](location_laterality_right)",
    "sur la [droite](location_laterality_right)",
    "[droit](location_laterality_right)",
    "[droits](location_laterality_right)",
    "[droites](location_laterality_right)",

    "à [gauche](location_laterality_left)",
    "[gauche](location_laterality_left)",
    "[gauches](location_laterality_left)",
    "sur la [gauche](location_laterality_left)",

    "[bilatéral](location_laterality_left,location_laterality_right)",
    "[bilatérales](location_laterality_left,location_laterality_right)",
    "[bilatéraux](location_laterality_left,location_laterality_right)",

    "[mammaire](location_body_part_breast)",
    "[sein](location_body_part_breast)",
    "[les seins](location_laterality_left,location_laterality_right,location_body_part_breast)",
    "dans [chaque sein](location_laterality_left,location_laterality_right,location_body_part_breast)",
    "[les deux seins](location_laterality_left,location_laterality_right,location_body_part_breast)",
    "[de chaque coté](location_laterality_left,location_laterality_right)",

    "[rénal](location_body_part_other)",
    "dans l'[épaule](location_body_part_other)",
    "sur le [bras](location_body_part_other)",
    "dans le [foie](location_body_part_other)",
    "[hépatique](location_body_part_other)",
    "[cerveau](location_body_part_other)",
    "[cranien](location_body_part_other)",

    #####################
    # SCORES
    #####################

    "[ACR 0](entity_type_score_acr,score_acr_0)",
    "[ACR 1](entity_type_score_acr,score_acr_1)",
    "[ACR I](entity_type_score_acr,score_acr_1)",
    "[ACR 2](entity_type_score_acr,score_acr_2)",
    "[ACR II](entity_type_score_acr,score_acr_2)",
    "[ACR 3](entity_type_score_acr,score_acr_3)",
    "[ACR III](entity_type_score_acr,score_acr_3)",
    "[ACR 4](entity_type_score_acr,score_acr_4)",
    "[ACR IV](entity_type_score_acr,score_acr_4)",
    "[ACR 4a](entity_type_score_acr,score_acr_4a)",
    "[ACR IVa](entity_type_score_acr,score_acr_4a)",
    "[ACR 4b](entity_type_score_acr,score_acr_4b)",
    "[ACR IVb](entity_type_score_acr,score_acr_4b)",
    "[ACR 4c](entity_type_score_acr,score_acr_4c)",
    "[ACR IVc](entity_type_score_acr,score_acr_4c)",
    "[ACR 5](entity_type_score_acr,score_acr_5)",
    "[ACR V](entity_type_score_acr,score_acr_5)",
    "[ACR 6](entity_type_score_acr,score_acr_6)",
    "[ACR VI](entity_type_score_acr,score_acr_6)",

    "[ACR 1](entity_type_score_acr,score_acr_1) [droite](location_laterality_right)",
    "[ACR 2](entity_type_score_acr,score_acr_2) [droite](location_laterality_right)",
    "[ACR 3](entity_type_score_acr,score_acr_3) [gauche](location_laterality_left)",
    "[ACR 4](entity_type_score_acr,score_acr_4) [droite](location_laterality_right)",
    "[ACR 4a](entity_type_score_acr,score_acr_4a) à [droite](location_laterality_right)",
    "[ACR 4b](entity_type_score_acr,score_acr_4b) [gauche](location_laterality_left)",
    "[ACR 4c](entity_type_score_acr,score_acr_4c) à [droite](location_laterality_right)",
    "[ACR 5](entity_type_score_acr,score_acr_5) sur) [gauche](location_laterality_left)",
    "[ACR 6](entity_type_score_acr,score_acr_6) [droite](location_laterality_right)",

    "[densité [mammaire](location_body_part_breast) de [type 1](score_density_1)](entity_type_score_density)",
    "[densité [mammaire](location_body_part_breast) de [type I](score_density_1)](entity_type_score_density)",
    "[denses de [type I](score_density_1)](entity_type_score_density)",
    "[denses de [type 1](score_density_1)](entity_type_score_density)",
    "[denses de [type A](score_density_1)](entity_type_score_density)",
    "[densité A](score_density_1,entity_type_score_density)",
    "[densité 1](score_density_1,entity_type_score_density)",

    "[densité [mammaire](location_body_part_breast) de [type 2](score_density_2)](entity_type_score_density)",
    "[densité [mammaire](location_body_part_breast) de [type II](score_density_2)](entity_type_score_density)",
    "[denses de [type II](score_density_2)](entity_type_score_density)",
    "[denses de [type 2](score_density_2)](entity_type_score_density)",
    "[denses de [type B](score_density_2)](entity_type_score_density)",
    "[densité B](score_density_2,entity_type_score_density)",
    "[densité 2](score_density_2,entity_type_score_density)",

    "[densité [mammaire](location_body_part_breast) de [type 3](score_density_3)](entity_type_score_density)",
    "[densité [mammaire](location_body_part_breast) de [type III](score_density_3)](entity_type_score_density)",
    "[denses de [type III](score_density_3)](entity_type_score_density)",
    "[denses de [type 3](score_density_3)](entity_type_score_density)",
    "[denses de [type C](score_density_3)](entity_type_score_density)",
    "[densité C](score_density_3,entity_type_score_density)",
    "[densité 3](score_density_3,entity_type_score_density)",

    "[densité [mammaire](location_body_part_breast) de [type 4](score_density_4)](entity_type_score_density)",
    "[densité [mammaire](location_body_part_breast) de [type IV](score_density_4)](entity_type_score_density)",
    "[denses de [type IV](score_density_4)](entity_type_score_density)",
    "[denses de [type 4](score_density_4)](entity_type_score_density)",
    "[denses de [type D](score_density_4)](entity_type_score_density)",
    "[densité D](score_density_4,entity_type_score_density)",
    "[densité 4](score_density_4,entity_type_score_density)",

    #####################
    # QUADRANTS
    #####################

    "dans le [quadrant supéro interne](location_quadrant_qsi)",
    "dans le [qsi](location_quadrant_qsi)",
    "dans le [quadrant qsi](location_quadrant_qsi)",
    "dans le [quad supéro interne](location_quadrant_qsi)",

    "dans le [quadrant supéro externe](location_quadrant_qse)",
    "dans le [qse](location_quadrant_qse)",
    "dans le [quadrant qse](location_quadrant_qse)",
    "dans le [quad supéro externe](location_quadrant_qse)",

    "dans le [quadrant inféro interne](location_quadrant_qii)",
    "dans le [qii](location_quadrant_qii)",
    "dans le [quadrant qii](location_quadrant_qii)",
    "dans le [quad inféro interne](location_quadrant_qii)",

    "dans le [quadrant inféro externe](location_quadrant_qie)",
    "dans le [qie](location_quadrant_qie)",
    "dans le [quadrant qie](location_quadrant_qie)",
    "dans le [quad inféro externe](location_quadrant_qie)",

    "à l'[union des quadrants internes](location_quadrant_uqint)",
    "à l'[uqint](location_quadrant_uqint)",
    "à l'[union qint](location_quadrant_uqint)",
    "à l'[union quad. internes](location_quadrant_uqint)",

    "à l'[union des quadrants supérieurs](location_quadrant_uqsup)",
    "à l'[uqsup](location_quadrant_uqsup)",
    "à l'[union qsup](location_quadrant_uqsup)",
    "à l'[union quad. supérieurs](location_quadrant_uqsup)",

    "à l'[union des quadrants inférieurs](location_quadrant_uqinf)",
    "à l'[uqinf](location_quadrant_uqinf)",
    "à l'[union qinf](location_quadrant_uqinf)",
    "à l'[union quad. inferieurs](location_quadrant_uqinf)",

    "à l'[union des quadrants externes](location_quadrant_uqext)",
    "à l'[uqext](location_quadrant_uqext)",
    "à l'[union qext](location_quadrant_uqext)",
    "à l'[union quad. externes](location_quadrant_uqext)",

    "la [zone para-areolaire](location_quadrant_areo)",
    "l'[aire aréolaire](location_quadrant_areo)",
    "[rétro-aréolaire](location_quadrant_areo)",
    "à l'[union des quatre quadrants](location_quadrant_areo)",

    "dans les [aires axillaires](location_quadrant_axi)",
    "dans les [zones axillaires](location_quadrant_axi)",
    "dans les [prolongements axillaires](location_quadrant_axi)",
    "[axillaires](location_quadrant_axi)",

    #####################
    # LESIONS
    #####################

    "un [nodule](entity_type_lesion)",
    "un [nodule hypoéchogène](entity_type_lesion)",
    "une [calcification kystique](entity_type_lesion)",
    "un [kyste calcifié](entity_type_lesion)",
    "des [nodules](entity_type_lesion)",
    "une [micro-calcification](entity_type_lesion)",
    "des [micro-calcifications](entity_type_lesion)",
    "une [macro-calcification](entity_type_lesion)",
    "des [macro-calcifications](entity_type_lesion)",
    "un [carcinome](entity_type_lesion)",
    "des [carcinomes](entity_type_lesion)",
    "une [néoplasie](entity_type_lesion)",
    "des [néoplasies](entity_type_lesion)",
    "une [lésion](entity_type_lesion)",
    "des [lésions](entity_type_lesion)",
    "une [masse](entity_type_lesion)",
    "des [masses](entity_type_lesion)",
    "un [surcroit d'opacité](entity_type_lesion)",
    "des [surcroits d'opacité](entity_type_lesion)",
    "des [ombilications](entity_type_lesion)",

    "pas de nodule",
    "aucune micro-calcification",
    "pas de macro-calcification",
    "aucune trace de carcinome",
    "pas de carcinomes",
    "pas de néoplasie",
    "néoplasie familiale",
    "carcinome chez la mère",
    "carcinome dans la famille",
    "on n'observe pas néoplasie",
    "pas de lésion",
    "abscence de masse",
    "pas de masse",
    "pas de surcroit d'opacité",

    "pas d'[échographie](proc_diag_echography)",
    "pas de [mammographie](entity_type_proc_diag,proc_diag_mammography)",
    "non [mammographique](proc_diag_mammography)",
    "sans [echographie](proc_diag_echography)",
    "pas d'[IRM](proc_diag_irm)",
    "aucune [imagerie par résonnance médicale](proc_diag_irm)",

    "pas de [chirurgie](proc_ther_surgery)",
    "aucune [ablation](proc_ther_surgery)",

    "sans [radiotherapie](proc_ther_other)",
    "pas de [chimiotherapie](proc_ther_other)",

    #####################
    # MODES
    #####################

    "[mammographie](entity_type_proc_diag,proc_diag_mammography)",
    "[mammographique](entity_type_proc_diag,proc_diag_mammography)",

    "[echographie](entity_type_proc_diag,proc_diag_echography)",
    "[echographique](entity_type_proc_diag,proc_diag_echography)",
    "[echographié](entity_type_proc_diag,proc_diag_echography)",

    "[IRM](entity_type_proc_diag,proc_diag_irm)",
    "[imagerie par résonnance médicale](entity_type_proc_diag,proc_diag_irm)",

    "une [biopsie](entity_type_proc_diag,proc_diag_biopsy)",
    "une [microbiopsie](entity_type_proc_diag,proc_diag_biopsy)",
    "[microbiopsié](entity_type_proc_diag,proc_diag_biopsy)",
    "une [macrobiopsie](entity_type_proc_diag,proc_diag_biopsy)",
    "par [cytoponction](entity_type_proc_diag,proc_diag_biopsy)",

    "une [palpation](entity_type_proc_diag,proc_diag_palpation)",
    "[palpé](entity_type_proc_diag,proc_diag_palpation)",
    "relevé par [compression](entity_type_proc_diag,proc_diag_palpation)",

    "une [craniographie](entity_type_proc_diag,proc_diag_other)",
    "une [scintigraphie](entity_type_proc_diag,proc_diag_other)",
    "une [prise de température](entity_type_proc_diag,proc_diag_other)",
    "un [scanner](entity_type_proc_diag,proc_diag_other)",

    "une [chirurgie](entity_type_proc_ther,proc_ther_surgery)",
    "une [ablation](entity_type_proc_ther,proc_ther_surgery)",

    "une [radiotherapie](entity_type_proc_ther,proc_ther_other)",
    "une [chimiotherapie](entity_type_proc_ther,proc_ther_other)",

    #####################
    # SIZES
    #####################

    "[centimétrique](has_size)",
    *(f"mesurant [{size} cm](has_size)".replace(".", ",") for size in np.arange(0, 5, 0.3)),
    *(f"mesurant [{size} mm](has_size)".replace(".", ",") for size in np.arange(0, 50, 3)),
    *(f"mesuré à [{size} cm](has_size)".replace(".", ",") for size in np.arange(0, 5, 0.3)),
    *(f"mesuré à [{size} centimètres](has_size)".replace(".", ",") for size in np.arange(0, 5, 0.3)),
    *(f"mesuré à [{size} millimetres](has_size)".replace(".", ",") for size in np.arange(0, 50, 3)),

    *(f"à une distance de [{size} cm](has_distance)".replace(".", ",") for size in np.arange(0, 5, 0.3)),
    *(f"à une distance de [{size} mm](has_distance)".replace(".", ",") for size in np.arange(0, 50, 3)),
    *(f"situé à [{size} cm](has_distance)".replace(".", ",") for size in np.arange(0, 5, 0.3)),
    *(f"situé à [{size} mm](has_distance)".replace(".", ",") for size in np.arange(0, 50, 3)),
    *(f"situé à [{size} millimetres](has_distance)".replace(".", ",") for size in np.arange(0, 50, 3)),

    *(f"sur le [rayon horaire de {angle} h](has_angle)" for angle in range(1, 13, 1)),
    *(f"situé à [{angle} h](has_angle)" for angle in range(1, 13, 1)),

    #####################
    # TEMPORALITY
    #####################

    f"[antécédent](is_passed)",
    f"[précédent](is_passed)",
    f"[passé](is_passed)",
    f"en zone antérieure",
    f"en zone postérieure",

    f"[prochainement](is_future)",
    f"[à venir](is_future)",
    f"[pour bientot](is_future)",

    *(f"[dans {n} mois](is_future)" for n in range(1, 12)),
    *(f"[d'ici {n} mois](is_future)" for n in range(1, 12)),
    *(f"[dans {n} ans](is_future)" for n in range(1, 20)),
    *(f"[d'ici {n} ans](is_future)" for n in range(1, 20)),

    *(f"[il y a {n} mois](is_passed)" for n in range(1, 12)),
    *(f"[depuis {n} mois](is_passed)" for n in range(1, 12)),
    *(f"[il y a {n} ans](is_passed)" for n in range(1, 20)),
    *(f"[depuis {n} ans](is_passed)" for n in range(1, 20)),
    *(f"depuis [{str(random.choice(range(1, 30))).rjust(2, '0')}/{str(random.choice(range(1, 12))).rjust(2, '0')}/{random.choice(range(1960, 2030))}](is_passed)" for _ in range(20)),
    f"depuis [<????-??-??>](is_passed)",
    *(f"mis en service le {str(random.choice(range(1, 30))).rjust(2, '0')}/{str(random.choice(range(1, 12))).rjust(2, '0')}/{random.choice(range(1960, 2030))}" for _ in range(20)),
    f"depuis [<????-??-??>](is_passed)",

    *(f"il y a {n} h" for n in range(1, 20)),
    *(f"dans {n} h" for n in range(1, 20)),
    *(f"d'ici [{str(random.choice(range(1, 30))).rjust(2, '0')}/{str(random.choice(range(1, 12))).rjust(2, '0')}/{random.choice(range(1960, 2040))}](is_future)" for _ in range(20)),
    f"d'ici [<????-??-??>](is_future)",
    "évolution du TAC par rapport", "pas de changement de BEPKL ", "dans la bibliographie de l'ACL, on peut trouver de tout", "l'organisation du sahel", "dans la méditerannée", "prise d'aspirine",
]


# for mention in terminology[-1:]:
#    print(regex_multisub_with_spans([r"\[([^\]\[]*)\]\((.*)\)"], [r"\1"], mention,))
#    break

def postprocess_template(text, doc_id):
    spans = []
    text2 = text
    for _ in range(4):
        new_spans = [(m.group(2).split(','), m.start(1), m.end(1), m.end()) for m in re.finditer("\[([^\]\[]+)\]\(([^)]+?)\)", text2)]
        if not new_spans:
            break
        for labels, begin, end, sub_end in new_spans:
            text2 = text2[:begin - 1] + " " * (sub_end - begin + 1) + text2[sub_end:]
            spans.append((labels, begin, end))

    spans = sorted(spans)
    if not len(spans):
        return {
            "doc_id": doc_id,
            "text": text,
            "formatted": text,
            "is_synthetic": True,
            "fragments": [],
            "entities": [],
        }
    labels, begins, ends = zip(*spans)
    new_text, deltas = regex_multisub_with_spans([r"(\]\(([^)]+?)\))", r"(\[)"] * 2, ["", ""] * 2, text, return_deltas=True)
    return {
        "doc_id": doc_id,
        "text": new_text,
        "is_synthetic": True,
        "formatted": text,
        "entities": [],
        "fragments": [
            {"fragment_id": str(fid), "label": label, "begin": begin, "end": end, "text": new_text[begin:end]}
            for fid, (label, begin, end) in
            enumerate([(label, begin, end) for labels, begin, end in zip(labels, deltas.apply(begins, side='left'), deltas.apply(ends, side='right')) for label in labels])
        ]
    }


prompts = ["Le concept est {} .", "cependant, il s'y trouve {} suite au ça", "On appelle cela {} ainsi."]
biprompts = ["Il y a {} et {}.", "On a {}, {} et voila.", "Il faut {} {}."]

label_to_synonym = defaultdict(lambda: [])
for sample in terminology:
    labels = re.findall("\((.*?)\)", sample)
    for label in labels:
        label_to_synonym[label].append(sample)
    if not labels:
        label_to_synonym[None].append(sample)
label_to_synonym = list(label_to_synonym.items())
counts = {label: len(synset) for label, synset in label_to_synonym}
multipliers = {label: math.ceil(max(counts.values()) / count) for label, count in counts.items()}
resampled_terminology = [syn for label, synset in label_to_synonym for syn in synset * multipliers[label]]
synthetic_data_lines = permute(
    [prompt.format(synthetic_sample)
     for doc_id, synthetic_sample in enumerate(resampled_terminology)
     for prompt in prompts] +
    [prompt.format(sample1, sample2)
     for doc_id, (sample1, sample2) in enumerate(zip(resampled_terminology, permute(resampled_terminology)))
     for prompt in biprompts]
)
synthetic_data = []
while len(synthetic_data_lines):
    doc = []
    for i in range(random.choice([1, 2, 3])):
        if len(synthetic_data_lines):
            doc.append(synthetic_data_lines.pop())
    synthetic_data.append(postprocess_template("\n".join(doc), "SYN-{}".format(len(synthetic_data))))

temporality_labels = [
    'is_now',
    'is_passed',
    'is_future',
    #    'is_now_and_passed',
]

location_body_part_labels = [
    # Location
    'location_body_part_breast',
    'location_body_part_other',
]
location_laterality_labels = [
    'location_laterality_left',
    'location_laterality_right',
    #    'location_laterality_bilateral',
]

location_quadrant_labels = [
    ('location_quadrant_qse',),
    ('location_quadrant_qsi',),
    ('location_quadrant_qie',),
    ('location_quadrant_qii',),

    ('location_quadrant_uqext',),
    ('location_quadrant_uqint',),
    ('location_quadrant_uqsup',),
    ('location_quadrant_uqinf',),

    ('location_quadrant_areo',),
    ('location_quadrant_axi',),

    ('location_quadrant_areo', 'location_quadrant_qse'),
    ('location_quadrant_areo', 'location_quadrant_qsi'),
    ('location_quadrant_areo', 'location_quadrant_qie'),
    ('location_quadrant_areo', 'location_quadrant_qii'),
    ('location_quadrant_areo', 'location_quadrant_uqsup'),
    ('location_quadrant_areo', 'location_quadrant_uqext'),
    ('location_quadrant_areo', 'location_quadrant_uqint'),
    ('location_quadrant_areo', 'location_quadrant_uqinf'),

    ('location_quadrant_qse', 'location_quadrant_uqsup'),
    ('location_quadrant_qsi', 'location_quadrant_uqsup'),
    ('location_quadrant_qse', 'location_quadrant_uqext'),
    ('location_quadrant_qie', 'location_quadrant_uqext'),
    ('location_quadrant_qsi', 'location_quadrant_uqint'),
    ('location_quadrant_qii', 'location_quadrant_uqint'),
    ('location_quadrant_qie', 'location_quadrant_uqinf'),
    ('location_quadrant_qii', 'location_quadrant_uqinf'),

    ('location_quadrant_axi', 'location_quadrant_qse',),
]


def all_combinations(l):
    return (comb for n in range(len(l) + 1) for comb in combinations(l, n))


location_labels = [
    *location_body_part_labels,
    *location_laterality_labels,
    *location_quadrant_labels,
]
location_laterality_combinations = [
    [],
    ['location_laterality_left'],
    ['location_laterality_right'],
    # ['location_laterality_bilateral'],
    # ['location_laterality_left', 'location_laterality_bilateral'],
    # ['location_laterality_right', 'location_laterality_bilateral'],
]
lesion_comb_loc = [
    *(("location_body_part_breast", *lat, quad)
      for lat in location_laterality_combinations
      for quad in (None, *location_quadrant_labels)),
    *(("location_body_part_other", *lat)
      for lat in location_laterality_combinations),
    *(lat for lat in location_laterality_combinations),
]
proc_comb_loc = [
    *(("location_body_part_breast", *lat,)
      for lat in location_laterality_combinations),
    *(("location_body_part_other", *lat)
      for lat in location_laterality_combinations),
    *(lat for lat in location_laterality_combinations),
]

# Score ACR
score_acr_labels = [
    'score_acr_0',
    'score_acr_1',
    'score_acr_2',
    'score_acr_3',
    'score_acr_4',
    'score_acr_4a',
    'score_acr_4b',
    'score_acr_4c',
    'score_acr_5',
    'score_acr_6',
]

# Score densité
score_density_labels = [
    'score_density_1',
    'score_density_2',
    'score_density_3',
    'score_density_4',
]

proc_diag_type_labels = [
    ('proc_diag_echography',),
    ('proc_diag_mammography',),
    ('proc_diag_irm',),
    ('proc_diag_biopsy',),
    ('proc_diag_other',),
    ('proc_diag_palpation',),
    (),
]

proc_ther_type_labels = [
    ('proc_ther_surgery',),
    ('proc_ther_other',),
    (),
]

# Entity type
entity_types_labels = [
    'entity_type_lesion',
    'entity_type_score_acr',
    'entity_type_score_density',
    'entity_type_proc_diag',
    'entity_type_proc_ther',
]

# Measure / distance / angle
size_labels = [
    'has_size',
    None,
]
# Has distance
distance_labels = [
    'has_distance',
    None,
]

angle_labels = [
    'has_angle',
    None,
]

enum_labels = list(filter(lambda x: x is not None, [
    *entity_types_labels,
    *proc_diag_type_labels,
    *proc_ther_type_labels,
    *score_acr_labels,
    *score_density_labels,
    *location_body_part_labels,
    *location_laterality_labels,
    *location_quadrant_labels,
    *size_labels,
    *distance_labels,
    *angle_labels,
    *temporality_labels,
]))

enum_outputs = [
                   sorted_tuple('entity_type_score_acr', score_acr, 'location_body_part_breast', *loc, *temp)
                   for score_acr in score_acr_labels
                   for loc in location_laterality_combinations
                   for temp in (('is_now',), ('is_passed',),)
               ] + [
                   sorted_tuple('entity_type_score_density', score_density, 'location_body_part_breast', *loc, *temp)
                   for score_density in (None, *score_density_labels)
                   for loc in location_laterality_combinations
                   for temp in (('is_now',), ('is_passed',))
               ] + [
                   sorted_tuple('entity_type_lesion', *loc, has_size, has_distance, has_angle, *temp)
                   for loc in lesion_comb_loc
                   for has_size in size_labels
                   for has_distance in distance_labels
                   for has_angle in angle_labels
                   for temp in (('is_now',), ('is_passed',), ('is_passed', 'is_now'),
                                # ('is_now_and_passed', 'is_passed', 'is_now')
                                )
               ] + [
                   sorted_tuple('entity_type_proc_diag', *entity_proc_diag_type, 'location_body_part_breast', *lat, *temp)
                   for entity_proc_diag_type in proc_diag_type_labels if "proc_diag_mammography" in entity_proc_diag_type
                   for lat in location_laterality_combinations
                   for temp in (('is_now',), ('is_passed',), ('is_future',))
               ] + [
                   sorted_tuple('entity_type_proc_diag', *entity_proc_diag_type, *loc, *temp)
                   for entity_proc_diag_type in proc_diag_type_labels if "proc_diag_mammography" not in entity_proc_diag_type
                   for loc in proc_comb_loc
                   for temp in (('is_now',), ('is_passed',), ('is_future',))
               ] + [
                   sorted_tuple('entity_type_proc_ther', *entity_proc_ther_type, *loc, *temp)
                   for entity_proc_ther_type in proc_ther_type_labels
                   for loc in proc_comb_loc
                   for temp in (('is_now',), ('is_passed',), ('is_future',))
               ]

value_labels = [
    'has_size',
    'has_distance',
    'has_angle',
]

label_mapping = {
    'is_now_and_passed': ('is_now',),
    'location_laterality_bilateral': ('location_laterality_left', 'location_laterality_right'),
}

# from data import terminology, postprocess_template, regex_multisub_with_spans
import re


def nested_fragments(f, prev=[]):
    if f in prev:
        return prev
    frags = [*prev, f]
    for c in f['contains']:
        frags.extend(nested_fragments(c, frags))
    return dedup(frags, key=id)


def postprocess_terminology_item(text):
    spans = []
    text2 = text
    for _ in range(4):
        new_spans = [(m.group(2).split(','), m.start(1), m.end(1), m.end()) for m in re.finditer("\[([^\]\[]+)\]\(([^)]+?)\)", text2)]
        if not new_spans:
            break
        for labels, begin, end, sub_end in new_spans:
            text2 = text2[:begin - 1] + " " * (sub_end - begin + 1) + text2[sub_end:]
            spans.append((labels, begin, end))

    spans = sorted(spans)
    if not len(spans):
        return {
            "text": text,
            "fragments": [],
            "entities": [],
        }
    labels, begins, ends = zip(*spans)
    new_text, deltas = regex_multisub_with_spans([r"(\]\(([^)]+?)\))", r"(\[)"] * 2, ["", ""] * 2, text, return_deltas=True)
    return {
        "text": new_text,
        "fragments": [
            {"fragment_id": str(fid), "label": label, "begin": begin, "end": end, "text": new_text[begin:end]}
            for fid, (label, begin, end) in
            enumerate([(label, begin, end) for labels, begin, end in zip(labels, deltas.apply(begins, side='left'), deltas.apply(ends, side='right')) for label in labels])
        ]
    }


def replace_in_text_(doc, begin, end, replacement_text, replacement_fragments):
    new_text = doc["text"][:begin] + replacement_text + doc["text"][end:]
    offset = begin + len(replacement_text) - end
    change_list = []
    new_fragments = [{'begin': f['begin'] + begin, 'end': f['end'] + begin, 'label': f['label'], 'chunks': []} for f in replacement_fragments]
    for fragment in list(doc["fragments"]):
        assert fragment["begin"] < fragment["end"]
        if begin <= fragment['begin'] and fragment['end'] <= end:
            change_list.append(fragment)
        else:
            if fragment["begin"] >= end:
                fragment["begin"] += offset
            if fragment["end"] >= end:
                fragment["end"] += offset
            assert fragment["begin"] < fragment["end"]

    changed_chunks = []
    for entity in doc["entities"]:
        for chunk in entity["chunks"]:
            for f in chunk["fragments"]:
                assert f in doc["fragments"], "Mising before {}".format({"begin": f["begin"], "end": f["end"], "label": f["label"]})
    for fragment in change_list:
        same_label_new_fragments = [f for f in new_fragments if f['label'] == fragment['label']]
        for chunk in fragment["chunks"]:
            chunk['fragments'] = [f for f in chunk['fragments'] if f is not fragment] + same_label_new_fragments

            for f in same_label_new_fragments:
                f['chunks'].append(chunk)

        changed_chunks.extend(fragment['chunks'])
        doc['fragments'] = [f for f in doc['fragments'] if f is not fragment]
    for fragment in doc['fragments']:
        fragment['chunks'] = dedup(fragment['chunks'], key=id)
    doc['fragments'].extend(new_fragments)
    for chunk in changed_chunks:
        chunk['fragments'] = dedup(chunk['fragments'], key=id)
    for entity in doc["entities"]:
        for chunk in entity["chunks"]:
            for f in chunk["fragments"]:
                assert f in doc["fragments"], "Mising {}".format({"begin": f["begin"], "end": f["end"], "label": f["label"], 'id': id(f), 'chunk_id': id(chunk)})
    for f in doc["fragments"]:
        assert f["begin"] < f["end"]
    doc['text'] = new_text
    return doc


ner_to_multilabel = {
    'lesion': [
        ('entity_type_lesion',)
    ],
    'diag': [
        ('entity_type_proc_diag', 'proc_diag_biopsy',),
        ('entity_type_proc_diag', 'proc_diag_echography',),
        ('entity_type_proc_diag', 'proc_diag_irm',),
        ('entity_type_proc_diag', 'proc_diag_mammography',),
        ('entity_type_proc_diag', 'proc_diag_other',),
        ('entity_type_proc_diag', 'proc_diag_palpation',),
        ('entity_type_proc_diag',),
        ('proc_diag_biopsy',),
        ('proc_diag_echography',),
        ('proc_diag_irm',),
        ('proc_diag_mammography',),
        ('proc_diag_other',),
        ('proc_diag_palpation',),
    ],
    'ther': [
        ('entity_type_proc_ther', 'proc_ther_surgery',),
        ('entity_type_proc_ther', 'proc_ther_other',),
        ('entity_type_proc_ther',),
        ('proc_ther_surgery',),
        ('proc_ther_other'),
    ],
    'acr': [
        ('entity_type_score_acr', 'score_acr_0',),
        ('entity_type_score_acr', 'score_acr_1',),
        ('entity_type_score_acr', 'score_acr_2',),
        ('entity_type_score_acr', 'score_acr_3',),
        ('entity_type_score_acr', 'score_acr_4',),
        ('entity_type_score_acr', 'score_acr_4a',),
        ('entity_type_score_acr', 'score_acr_4b',),
        ('entity_type_score_acr', 'score_acr_4c',),
        ('entity_type_score_acr', 'score_acr_5',),
        ('entity_type_score_acr', 'score_acr_6',),
    ],
    'density': [
        ('entity_type_score_density', 'score_density_1',),
        ('entity_type_score_density', 'score_density_2',),
        ('entity_type_score_density', 'score_density_3',),
        ('entity_type_score_density', 'score_density_4',),
    ],
    'quadrant': location_quadrant_labels,
    'temporality': [
        ('is_now',),
        ('is_future',),
        ('is_passed',),
    ],
    'organ': [
        ('location_body_part_breast',),
        ('location_body_part_other',),
    ],
    'laterality': [
        ('location_laterality_left',),
        ('location_laterality_right',),
        ('location_laterality_left', 'location_laterality_right',),
    ],
    'angle': [
        ('has_angle',),
    ],
    'distance': [
        ('has_distance',),
    ],
    'size': [
        ('has_size',),
    ]
}

ner_label_mapping = {}
for ner_label, multilabels_sets in ner_to_multilabel.items():
    for multilabel in multilabels_sets:
        for label in multilabel:
            ner_label_mapping[label] = ner_label
