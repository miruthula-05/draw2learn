import re
from collections import Counter


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "he",
    "her",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
}
PRONOUN_WORDS = {"he", "she", "they", "them", "his", "her", "their", "it"}
SETTING_HINTS = {
    "night": ["night", "evening", "moon", "stars"],
    "rain": ["rain", "rainy", "storm", "cloud", "wet"],
    "village": ["village", "road", "path", "market"],
    "school": ["school", "classroom", "teacher", "students"],
    "forest": ["forest", "tree", "woods", "jungle"],
    "home": ["home", "house", "kitchen", "mother", "room"],
    "field": ["field", "farm", "grass", "garden"],
    "river": ["river", "pond", "water", "boat"],
}
EMOTION_KEYWORDS = {
    "sad": ["sad", "cried", "cry", "upset", "hurt", "lost", "alone", "shivering", "worried"],
    "surprised": ["suddenly", "surprised", "shock", "unexpected", "found", "appeared"],
    "hungry": ["hungry", "food", "eat", "ate", "meal"],
    "bored": ["waited", "bored", "quiet", "still"],
    "excited": ["excited", "cheered", "celebrated", "ran", "jumped", "rescued", "barked"],
    "happy": ["happy", "smiled", "hugged", "played", "laughed", "safe", "thank", "relieved"],
}
CHARACTER_HINTS = {
    "boy", "girl", "mother", "father", "maa", "mom", "mummy", "mum", "teacher", "student", "students",
    "friend", "friends", "neighbour", "neighbours", "neighbor", "neighbors", "puppy", "dog", "cat", "rabbit",
    "lion", "tiger", "fox", "frog", "owl", "cow", "elephant", "monkey", "king", "queen", "prince",
    "princess", "circle", "square", "triangle", "rectangle", "star", "badal", "moti"
}
NON_CHARACTER_OBJECT_HINTS = {
    "school", "pit", "rope", "mountain", "mountains", "pattern", "patterns", "road", "lane", "gate",
    "house", "home", "village", "forest", "field", "river", "pond", "water", "boat", "tree", "garden"
}
OBJECT_ALIASES = {
    "moti": {"moti", "puppy", "dog", "pup"},
    "mother": {"mother", "maa", "mom", "mum", "mummy"},
    "badal": {"badal", "boy", "child"},
    "neighbours": {"neighbour", "neighbours", "neighbor", "neighbors", "people"},
    "star": {"star"},
}


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]



def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())



def should_apply_expression(object_name: str) -> bool:
    tokens = [token for token in _tokenize(object_name) if token not in STOPWORDS]
    if not tokens:
        return False
    if any(token in {"neighbour", "neighbours", "neighbor", "neighbors"} for token in tokens):
        return False
    if any(token in NON_CHARACTER_OBJECT_HINTS for token in tokens):
        return False
    if any(token in CHARACTER_HINTS for token in tokens):
        return True
    if all(token[0].isalpha() for token in tokens) and object_name[:1].isupper():
        return True
    return False


BACKGROUND_OBJECT_HINTS = {
    'pit', 'school', 'mountain', 'mountains', 'pattern', 'patterns', 'road', 'lane', 'gate', 'house', 'home',
    'village', 'forest', 'field', 'river', 'pond', 'water', 'boat', 'tree', 'garden'
}


def should_request_child_drawing(object_name: str) -> bool:
    tokens = {token for token in _tokenize(object_name) if token not in STOPWORDS}
    if not tokens:
        return False
    if 'rope' in tokens:
        return True
    if tokens.intersection(BACKGROUND_OBJECT_HINTS):
        return False
    return should_apply_expression(object_name)


def object_alias_tokens(object_name: str) -> set[str]:
    tokens = {token for token in _tokenize(object_name) if token not in STOPWORDS}
    normalized = object_name.strip().lower()
    for token in list(tokens) + [normalized]:
        tokens.update(OBJECT_ALIASES.get(token, set()))
    if "moti" in tokens:
        tokens.update({"moti", "puppy", "dog", "pup"})
    if tokens.intersection({"mother", "maa", "mom", "mum", "mummy"}):
        tokens.update({"mother", "maa", "mom", "mum", "mummy"})
    if "badal" in tokens:
        tokens.update({"badal", "boy", "child"})
    if tokens.intersection({"neighbour", "neighbours", "neighbor", "neighbors"}):
        tokens.update({"neighbour", "neighbours", "neighbor", "neighbors", "people"})
    return {token for token in tokens if token and token not in STOPWORDS}



def suggest_story_objects(text: str, limit: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z']+", text or "")
    if not words:
        return []

    capitalized = []
    frequent_candidates = []
    for word in words:
        if len(word) < 3:
            continue
        lowered = word.lower()
        if lowered in STOPWORDS:
            continue
        if word[0].isupper():
            capitalized.append(word.title())
        frequent_candidates.append(word.title())

    ordered = []
    for item in capitalized:
        if item not in ordered:
            ordered.append(item)

    counts = Counter(frequent_candidates)
    for item, _count in counts.most_common(limit * 2):
        if item not in ordered:
            ordered.append(item)

    return ordered[:limit]



def detect_scene_expression(sentence: str) -> str:
    lowered = sentence.lower()
    for expression, keywords in EMOTION_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return expression
    return "happy"



def detect_scene_setting(sentence: str) -> str:
    lowered = sentence.lower()
    for setting, keywords in SETTING_HINTS.items():
        if any(keyword in lowered for keyword in keywords):
            return setting
    return "village"



def _match_objects(sentence: str, selected_objects: list[str], previous_objects: list[str]) -> list[str]:
    sentence_tokens = set(_tokenize(sentence))
    matched = []

    for object_name in selected_objects:
        direct_tokens = [token for token in _tokenize(object_name) if token not in STOPWORDS]
        object_tokens = object_alias_tokens(object_name)
        if not object_tokens:
            continue
        if direct_tokens and all(token in sentence_tokens for token in direct_tokens):
            matched.append(object_name)
            continue
        if sentence_tokens.intersection(object_tokens):
            matched.append(object_name)

    if sentence_tokens.intersection(PRONOUN_WORDS):
        for item in previous_objects[:2]:
            if item not in matched:
                matched.append(item)

    if not matched and previous_objects:
        carry_keywords = {"then", "next", "after", "later", "suddenly"}
        if sentence_tokens.intersection(carry_keywords):
            matched.extend(previous_objects[:2])

    if not matched and selected_objects:
        matched.append(selected_objects[0])

    ordered = []
    for item in matched:
        if item not in ordered:
            ordered.append(item)
    return ordered[:3]



def build_story_scenes(text: str, selected_objects: list[str]) -> list[dict]:
    scenes = []
    previous_objects = []

    for sentence in split_sentences(text):
        matched_objects = _match_objects(sentence, selected_objects, previous_objects)
        scene = {
            "sentence": sentence,
            "objects": matched_objects,
            "expression": detect_scene_expression(sentence),
            "setting": detect_scene_setting(sentence),
        }
        scenes.append(scene)
        if matched_objects:
            previous_objects = matched_objects

    return scenes