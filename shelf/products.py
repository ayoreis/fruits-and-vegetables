"""Plain English names for the Portuguese catalogue names, for prompting vision models."""

import re

# Checked in order, so more specific prefixes come before more general ones.
ENGLISH_NAMES = [
    ("ABACATE", "avocados"),
    ("ABACAXI", "pineapples"),
    ("ABÓBORA", "pumpkins"),
    ("ALFACE", "lettuce"),
    ("ALHO", "garlic"),
    ("ALPERCE", "apricots"),
    ("BANANA", "bananas"),
    ("BATATA DOCE", "sweet potatoes"),
    ("BATATA VERMELHA", "red potatoes"),
    ("BATATA", "potatoes"),
    ("BERINGELA", "aubergines"),
    ("CEBOLA ROXA", "red onions"),
    ("CEBOLINHA", "chives"),
    ("CEBOLA", "onions"),
    ("CEREJA", "cherries"),
    ("CHUCHU", "chayotes"),
    ("CITR", "oranges"),
    ("CLEMENTINA", "clementines"),
    ("COUVE BROCOLO", "broccoli"),
    ("COUVE FLOR", "cauliflowers"),
    ("COUVE ROXA", "red cabbages"),
    ("COUVE PORTUGUESA", "collard greens"),
    ("COUVE BRANCA", "white cabbages"),
    ("CURGETE", "courgettes"),
    ("FEIJÃO VERDE", "green beans"),
    ("GENGIBRE", "ginger"),
    ("GOIABA", "guavas"),
    ("GRANADILHA", "granadillas"),
    ("KIWI GOLD", "golden kiwis"),
    ("KIWI", "kiwis"),
    ("LIMÃO", "lemons"),
    ("LIMA", "limes"),
    ("MAMAO", "papayas"),
    ("PAPAIA", "papayas"),
    ("MANGA", "mangoes"),
    ("MARACUJA", "passion fruits"),
    ("MELANCIA", "watermelons"),
    ("MELAO", "melons"),
    ("MELOA", "melons"),
    ("MORANGO", "strawberries"),
    ("NECTARINA", "nectarines"),
    ("PEPINO", "cucumbers"),
    ("PERA", "pears"),
    ("PESSEGO PARAGUAYO", "flat peaches"),
    ("PESSEGO", "peaches"),
    ("PIMENTO VERDE", "green peppers"),
    ("PIMENTO VERMELHO", "red peppers"),
    ("PITAYA AMARELA", "yellow dragon fruits"),
    ("PITAYA VERMELHA", "red dragon fruits"),
    ("TOMATE CHERRY", "cherry tomatoes"),
    ("TOMATE MINI", "cherry tomatoes"),
    ("MINI TOMATE", "cherry tomatoes"),
    ("TOMATE", "tomatoes"),
    ("UVA BRANCA", "green grapes"),
    ("UVA RED", "red grapes"),
]
APPLES = [("GRANNY", "green apples"), ("GOLDEN", "yellow apples"), ("", "red apples")]

# A weight such as "500 G", "1.5 KG" or "4KG", but not a calibre such as "73/85 KG".
PACK_WEIGHT = re.compile(r"(?<![\w/.])\d+(\.\d+)?\s*(KG|G|GR)\b")


def is_packaged(product_name):
    """Loose products are sold by weight; packaged ones name the pack's weight."""
    return bool(PACK_WEIGHT.search(product_name))


def english_name(product_name):
    if product_name.startswith(("MACA", "MAÇA", "MAÇÂ", "MAÇÃ")):
        return next(name for word, name in APPLES if word in product_name)
    for prefix, name in ENGLISH_NAMES:
        if product_name.startswith(prefix):
            return name
    raise ValueError(f"No English name for {product_name!r}")
