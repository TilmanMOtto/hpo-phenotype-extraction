GROUNDING_PROMPT = (
    "As an expert clinician, your task is to accurately link the entity using the concepts listed below. "
    "Accuracy is paramount. "
    # "Do not guess missing words, and do not guess the meaning of any abbreviation. "
    # "If the entity refers to multiplt concepts, return the last one. "
    "If the entity does not precisely refer to any of the concepts listed below, please return \"None\"; "
    "otherwise, return the corresponding concept ID in the following format:\n"
    # "reason: <one-sentence reasoning about entity linking>\n"
    "answer: <concept ID or None>\n"
    "confidence: <one of: HIGH, LOW, MEDIUM>\n"
)

GROUNDING_PROMPT_NOCONF = (
    "As an expert clinician, your task is to accurately link the entity using the concepts listed below. "
    "Accuracy is paramount. "
    # "Do not guess missing words, and do not guess the meaning of any abbreviation. "
    # "If the entity refers to multiplt concepts, return the last one. "
    "If the entity does not precisely refer to any of the concepts listed below, please return \"None\"; "
    "otherwise, return the corresponding concept ID in the following format:\n"
    # "reason: <one-sentence reasoning about entity linking>\n"
    "answer: <concept ID or None>\n"
)

# GROUNDING_PROMPT = (
#     "As an expert clinician, your task is to accurately link the entity using the concepts listed below. "
#     "Accuracy is paramount. "
#     # "If the entity refers to multiplt concepts, return the last one. "
#     "If the entity refers precisely to any of the concepts listed below, return the corresponding concept ID in the following format:\n"
#     # "reason: <one-sentence reasoning about entity linking>\n"
#     "answer: <concept ID or None>\n"
#     "confidence: <one of: HIGH, LOW, MEDIUM>\n"
#     "Otherwise, please return \"None\". "
# )

# GROUNDING_PROMPT = (
#     "As an expert clinician, your task is to accurately link the entity using the concepts listed below. "
#     "Accuracy is paramount. "
#     "Do not guess missing words. "
#     # "If the entity refers to multiplt concepts, return the last one. "
#     "If the entity refers precisely to any of the concepts listed below, return the corresponding concept ID in the following format:\n"
#     # "reason: <one-sentence reasoning about entity linking>\n"
#     "answer: <concept ID or None>\n"
#     "confidence: <one of: HIGH, LOW, MEDIUM>\n"
#     "Otherwise, return \"None\". "
# )

# GROUNDING_PROMPT = (
#     "As an expert clinician, your task is to accurately link the entity using the concepts listed below. "
#     "Accuracy is paramount. "
#     "Do not guess missing words. "
#     # "If the entity refers to multiplt concepts, return the last one. "
#     "If the entity refers precisely and completely to any of the concepts listed below, return the corresponding concept ID in the following format:\n"
#     # "reason: <one-sentence reasoning about entity linking>\n"
#     "answer: <concept ID or None>\n"
#     "confidence: <one of: HIGH, LOW, MEDIUM>\n"
#     "Otherwise, return \"None\". "
# )

# GROUNDING_PROMPT = (
#     "As an expert clinician, your task is to accurately link the entity using the concepts listed below. "
#     "Accuracy is paramount. "
#     "Do not guess missing words."
#     # "If the entity refers to multiplt concepts, return the last one. "
#     "If the entity does not precisely and completely refer to any of the concepts listed below, please return \"None\"; "
#     "otherwise, return the corresponding concept ID in the following format:\n"
#     # "reason: <one-sentence reasoning about entity linking>\n"
#     "answer: <concept ID or None>\n"
#     "confidence: <one of: HIGH, LOW, MEDIUM>\n"
# )

# GROUNDING_PROMPT = (
#     "As an expert clinician, your task is to accurately link the entity using the concepts listed below. "
#     "Accuracy is paramount. "
#     "Do not guess missing words. "
#     # "If the entity refers to multiplt concepts, return the last one. "
#     "If the entity does not precisely refer to any of the concepts listed below, please return \"None\"; "
#     "otherwise, return the corresponding concept ID in the following format:\n"
#     # "reason: <one-sentence reasoning about entity linking>\n"
#     "answer: <concept ID or None>\n"
#     "confidence: <one of: HIGH, LOW, MEDIUM>\n"
# )

# GROUNDING_PROMPT = (
#     "As an expert clinician, your task is to accurately link the entity using the concepts listed below. "
#     "Accuracy is paramount. "
#     # "If the entity refers to multiplt concepts, return the last one. "
#     "If the entity does not precisely and completely refer to any of the concepts listed below, please return \"None\"; "
#     "otherwise, return the corresponding concept ID in the following format:\n"
#     # "reason: <one-sentence reasoning about entity linking>\n"
#     "answer: <concept ID or None>\n"
#     "confidence: <one of: HIGH, LOW, MEDIUM>\n"
# )

# BROAD_GROUNDING_PROMPT = (
#     "As an expert clinician, your task is to accurately link the entity using the concepts listed below. "
#     "Accuracy is paramount. "
#     "Do not guess missing words."
#     # "If the entity refers to multiplt concepts, return the last one. "
#     "If the entity does not precisely and completely refer to any of the concepts listed below, please return \"None\"; "
#     "otherwise, return the corresponding concept ID in the following format:\n"
#     # "reason: <one-sentence reasoning about entity linking>\n"
#     "answer: <concept ID or None>\n"
#     "confidence: <one of: HIGH, LOW, MEDIUM>\n"
# )