"""Map stable gesture IDs to configurable multilingual phrases."""
DEFAULT_INTENTS = {
    "HELLO": {"en": "Hello", "hi": "नमस्ते", "mr": "नमस्कार"},
    "WATER": {"en": "I need water", "hi": "मुझे पानी चाहिए", "mr": "मला पाणी हवे आहे"},
    "HELP": {"en": "I need help", "hi": "मुझे मदद चाहिए", "mr": "मला मदत हवी आहे"},
    "YES": {"en": "Yes", "hi": "हाँ", "mr": "हो"},
    "NO": {"en": "No", "hi": "नहीं", "mr": "नाही"},
}


def phrase_for(intent, language="en", table=None):
    table = table or DEFAULT_INTENTS
    return table.get(str(intent), {}).get(language, table.get(str(intent), {}).get("en", ""))
