"""Template-based synthetic Levantine/English code-switched sentence generator.

No public corpus of real Levantine-English code-switched *speech* exists to
train on (see PLAN.md Phase 2 / PRD risk table) — this generates plausible
*text* for two specific, legitimate uses that don't require matching audio:

1. Held-out evaluation prompts (FR4): we only need text to synthesize from
   and measure objective/subjective metrics on; there's no reference
   recording to compare a code-switched utterance against, so synthetic
   text is sufficient and appropriate here, not a workaround.
2. The self-training bootstrap loop (PLAN.md Phase 2 step 4b, stretch
   goal): generate text -> synthesize with the current checkpoint -> filter
   by ASR/quality metrics -> only *then* does it need to pass a quality bar
   before being used as training data.

Patterns below mirror common real Levantine tech/casual code-switching:
English nouns/verbs for modern concepts (meeting, laptop, email, password,
deadline) dropped into an Arabic carrier sentence, which is the dominant,
well-documented pattern in Levantine speech rather than switching at
arbitrary points.
"""

from __future__ import annotations

import random

_TEMPLATES = [
    "بدي أعمل {en} بكرا",
    "ممكن تبعتلي ال {en} عالـ email؟",
    "عندي {en} الساعة {num}",
    "ال {en} تبعي خربان، لازم أصلحه",
    "شفت ال {en} من يومين بس ما رديت",
    "هاد ال {en} كتير مهم بالنسبة إلي",
    "لازم نخلص ال {en} قبل ال {en2}",
    "كيف بقدر أعمل {en} من هون؟",
    "في {en} جديد لازم تجربه",
    "ارسلتلك ال {en} على الـ {en2}",
    "واحد {en} مع {num} دولار بس",
    "خلص ال {en}، رح نحكي بعدين",
]

_EN_SLOTS = [
    "meeting",
    "email",
    "laptop",
    "password",
    "deadline",
    "project",
    "schedule",
    "update",
    "interview",
    "presentation",
    "feedback",
    "backup",
    "account",
    "app",
    "message",
]

_NUMS = ["2", "3", "5", "10", "25"]


def generate(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    sentences = []
    for _ in range(n):
        template = rng.choice(_TEMPLATES)
        sentence = template.format(
            en=rng.choice(_EN_SLOTS),
            en2=rng.choice(_EN_SLOTS),
            num=rng.choice(_NUMS),
        )
        sentences.append(sentence)
    return sentences
