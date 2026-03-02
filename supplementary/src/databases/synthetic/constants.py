attributes_to_emotion_map_multi = {
    0: ["AgeGroup", "Sex"],  # Anger
    1: ["Race", "Ethnicity"],  # Disgust
    2: [],  # Fear (no specific attributes)
    3: ["AgeGroup"],  # Happy
    4: ["AgeGroup", "Sex", "Race", "Ethnicity"],  # Neutral
    5: ["AgeGroup", "Sex", "Race"],  # Sad
}

attributes_to_emotion_map_single = {
    0: ["AgeGroup"],  # Anger
    1: ["Sex"],  # Disgust
    2: ["Race"],  # Fear 
    3: ["Ethnicity"],  # Happy
    4: [], # Neutral (no specific attributes)
    5: [], # Sad (no specific attributes)
}

mappings = {
    "single": attributes_to_emotion_map_single,
    "multi": attributes_to_emotion_map_multi,
}
