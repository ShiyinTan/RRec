user_prompt = """\
Analyze in depth and finally recommend next {category_singular} I might purchase inside {{emb_token}} and {{emb_end_token}}.
For example, {{emb_token}}a product{{emb_end_token}}.

Below is my historical {category} purchases and ratings (out of 5):\
"""

##################### item #####################
item_prompt = "Summarize key attributes of the following {category_singular} inside {{emb_token}} and {{emb_end_token}}."


poi_user_prompt = """\
Analyze in depth and finally recommend the next {category_singular} I might visit inside {{emb_token}} and {{emb_end_token}}.
For example, {{emb_token}}a place{{emb_end_token}}.

Below is my historical visits to {category}. The number in parentheses is a check-in flag:\
"""


##################### categories ####################
cat_dict = {
    "Video_Games": ("video games", "video game"),
    "CDs_and_Vinyl": ("CDs and vinyl", "CD or vinyl"),
    "Musical_Instruments": ("musical instruments", "musical instrument"),
    "NYC": ("places in New York City", "place"),
    "TKY": ("places in Tokyo", "place"),
}

POI_CATEGORIES = {"NYC", "TKY"}


def obtain_prompts(category):
    """Obtain prompts for a specific category."""
    assert category in cat_dict, f"Category {category} not found in the dictionary."
    category_text, category_singular = cat_dict[category]
    template = poi_user_prompt if category in POI_CATEGORIES else user_prompt
    return {
        "user_prompt": template.format(category=category_text, category_singular=category_singular),
        'item_prompt': item_prompt.format(category_singular=category_singular),
    }
