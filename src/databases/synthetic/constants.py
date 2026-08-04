"""Regime definitions for the E2 synthetic experiment (SEAB extensions).

Three regimes are supported:

* ``additive``  -- the original SEAB-Single mapping (one attribute per
  emotion, bias fires when that single attribute is privileged).  This is
  the regime WAF's additive form is inductively biased for.

* ``multi``     -- the original SEAB-Multiple mapping (variable-arity AND
  gates per emotion, as in the paper).  Kept for backward compatibility
  with the existing pipeline.

* ``threshold`` -- reviewer-requested regime.  For every emotion the same
  global threshold applies: bias is injected iff at least ``k=3`` out of
  the four demographic attributes are simultaneously privileged.  Emotion
  ``i`` still uses its own ``multi``-style mapping to decide *which*
  emotion is affected -- the *only* thing that changes vs. ``multi`` is
  the trigger geometry (arbitrary-subset AND  ->  count >= k).

  This means:
    - Under ``multi`` the effective ``k`` per emotion equals ``|subset|``
      and varies across emotions.
    - Under ``threshold`` the effective ``k`` is fixed to 3 globally.

  Threshold effects cannot be perfectly captured by additive + two-way
  interaction terms (they need a 3-way and above interaction), so any
  residual WAF advantage on this regime is non-circular.
"""

from typing import Dict, List

# Original mappings (kept for reference / backward compat)
attributes_to_emotion_map_single = {
    0: ["AgeGroup"],   # Anger
    1: ["Sex"],        # Disgust
    2: ["Race"],       # Fear
    3: ["Ethnicity"],  # Happy
    4: [],             # Neutral (no bias)
    5: [],             # Sad (no bias)
}

attributes_to_emotion_map_multi = {
    0: ["AgeGroup", "Sex"],                              # Anger
    1: ["Race", "Ethnicity"],                            # Disgust
    2: [],                                               # Fear
    3: ["AgeGroup"],                                     # Happy
    4: ["AgeGroup", "Sex", "Race", "Ethnicity"],         # Neutral
    5: ["AgeGroup", "Sex", "Race"],                      # Sad
}

# For the ``threshold`` regime, the *emotion -> associated attribute set* is
# the same as ``multi``, but the trigger fires globally when >= k demographic
# attributes are privileged.  Emotions with empty subsets are still unbiased.
attributes_to_emotion_map_threshold = attributes_to_emotion_map_multi.copy()

# For the ``pairwise`` regime (E2, reviewer R1/R3 response for non-additive
# testing), each emotion is associated with a specific 2-attribute AND gate,
# and the 6 emotions collectively cover all C(4,2)=6 pairwise combinations of
# demographic attributes. This is a *genuinely non-additive* bias pattern
# (each emotion's bias requires the joint state of exactly two attributes) but
# is within the reach of 2-way interaction terms, so it directly tests whether
# WAF's pairwise-interaction extension captures signal that main-effects-only
# WAF (and SP/EO/FPR) cannot.
attributes_to_emotion_map_pairwise = {
    0: ["AgeGroup", "Sex"],        # Anger      -- Age x Sex
    1: ["Race", "Ethnicity"],      # Disgust    -- Race x Eth
    2: ["AgeGroup", "Race"],       # Fear       -- Age x Race
    3: ["Sex", "Ethnicity"],       # Happy      -- Sex x Eth
    4: ["AgeGroup", "Ethnicity"],  # Neutral    -- Age x Eth
    5: ["Sex", "Race"],            # Sadness    -- Sex x Race
}

# XOR-style variant: bias fires when *exactly one* of the pair is privileged.
# Under this rule the pair has zero main-effect signal (E[x_a * I(bias)] = 0)
# because +1 and -1 states are equally likely to trigger. Only the pairwise
# interaction term can detect the pattern -- a cleanest non-additive test.
attributes_to_emotion_map_xor = dict(attributes_to_emotion_map_pairwise)

mappings: Dict[str, Dict[int, List[str]]] = {
    "single": attributes_to_emotion_map_single,
    "multi": attributes_to_emotion_map_multi,
    # Aliases used by the E2 CLI:
    "additive": attributes_to_emotion_map_single,
    "threshold": attributes_to_emotion_map_threshold,
    "pairwise": attributes_to_emotion_map_pairwise,
    "xor": attributes_to_emotion_map_xor,
}

# Global threshold ``k`` used by the ``threshold`` regime.
THRESHOLD_K: int = 3
