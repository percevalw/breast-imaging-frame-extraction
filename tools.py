import random

def prod(x):
    p = 1
    for a in x:
        p = p * a
    return p

def permute(x):
    x = list(x)
    random.shuffle(x)
    return x

def pick(*lists):
    while True:
        yield [random.choice(l) for l in lists]


def sorted_tuple(*x):
    return tuple(sorted([part
                         for item in x if item is not None
                         for part in (item if isinstance(item, (tuple, list)) else (item,))]))


def dedup(l, key=None):
    return list({(key(item) if key is not None else item): item for item in l}.values())


def compose(*fns):
    def apply(x):
        for fn in fns:
            x = fn(x)
        return x

    return apply


def permute(x):
    x = list(x)
    random.shuffle(x)
    return x


def span_dice_overlap(a, b):
    tp = 0 if (b[0]>a[1] or a[0]>b[1]) else min(a[1], b[1]) - max(a[0], b[0])
    return tp, a[1]-a[0], b[1]-b[0]