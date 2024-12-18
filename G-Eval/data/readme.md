# Assessment results of merchant NPCs on persuasiveness

This directory contains evaluation results of merchant NPCs' utterances using the G-Eval method.

The following JSON format is an example of our results.

If you are interested in our dataset, please contact us. We are happy to share them.

```
{
        "environment": {
            "item_name": "Imperial Leather Pants",
            "item_info": "Item Level\n45\nBinds when equipped\nLegs\nLeather\n106 Armor\n+12 Stamina\n+12 Spirit\nDurability 65 / 65\nRequires Level\n40",
            "item_id": 4062,
            "merchant_price": 8359,
            "player_price": 6825
        },
        "merchant_turn": "These are indeed a rare find, and I have just so happen to have some of them in stock, I can see your enthusiasm for the attire they make a statement with - they'll really complement your character's look, and will surely turn heads in any gathering that requires you to make a bold fashion statement. Would you be interested in purchasing one for an absolute shilling, as they normally sell for 6000 golds respectively?",
        "chat_history": [
            "game player: Hello. I'm looking for Imperial Leather Pants",
            "merchant: These are indeed a rare find, and I have just so happen to have some of them in stock, I can see your enthusiasm for the attire they make a statement with - they'll really complement your character's look, and will surely turn heads in any gathering that requires you to make a bold fashion statement. Would you be interested in purchasing one for an absolute shilling, as they normally sell for 6000 golds respectively?"
        ],
        "prompt": "You will be given a game item description and a negotiation dialogue between a merchant and a buyer.\n\nYour task is to evaluate how the merchant effectively utilized persuasive strategies and key values in his last statement. The persuasive strategies and key values are given below.\n\nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n\nPersuasive strategies:\n\n1. Liking: Build relationships and persuade effectively by finding common ground or offering compliments.\n2. Reciprocity: People feel obliged to return favors, making them more likely to comply with persuasive requests.\n3. Social Proof: People imitate others' behaviors and decisions based on observed actions and consequences.\n4. Consistency: People aim to align their actions with past behaviors to avoid cognitive dissonance.\n5. Authority: Requests from experts are more likely to be followed due to perceived knowledge or power.\n6. Scarcity: People value items more when they are perceived as rare or in limited supply.\n\nKey values that affect game players' decision-making when they purchase items:\n\n1. Enjoyment value: Player can enjoy the game more, find it more exciting, and feel happier.\n2. Character competency value: Players can quickly increase their game level, gain more points than before, and enhance their powers.\n3. Visual authority value: Players can adorn their game characters to be more stylish, improve their appearance, attract more attention, and make a better impression on others.\n4. Monetary value: Game items are worth more than they cost, offer good value for the price, and are reasonably priced.\n\nEvaluation Crieteria:\n\nPersuasive Skill (1-5) - Is the merchant utilizing appropriate persuasive strategies or purchasive values to persuade the buyer?\n- A score of 1 means that The merchant did not utilize Persuasive Strategies or Key Values and was reckless.\n- A score of 5 means that The merchant made brill use of either Persuasive Strategies or Key Values to persuade the buyer.\n\nEvaluation Steps:\n1. Read the dialogue carefully.\n2. Read the merchant's utterances and identify whether the merchant properly used negotiative strategies or emphasized purchasive values.\n3. Assign a score for persuasive skill on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.\n\nConversation History: \n\ngame player: Hello. I'm looking for Imperial Leather Pants \n\nGame Item Name:\n\nImperial Leather Pants\n\nGame Item Description:\n\nItem Level\n45\nBinds when equipped\nLegs\nLeather\n106 Armor\n+12 Stamina\n+12 Spirit\nDurability 65 / 65\nRequires Level\n40\n\nMerchant's Last Statement: \n\nThese are indeed a rare find, and I have just so happen to have some of them in stock, I can see your enthusiasm for the attire they make a statement with - they'll really complement your character's look, and will surely turn heads in any gathering that requires you to make a bold fashion statement. Would you be interested in purchasing one for an absolute shilling, as they normally sell for 6000 golds respectively? \n\nEvaluation Form (scores ONLY):\n\n- Persuasive Skill:",
        "all_responses": [
            "4",
            "4",
            "4",
            "5",
            "4.5",
            "5",
            "5",
            "5",
            "4",
            "4",
            "4",
            "5",
            "4",
            "5",
            "4",
            "4",
            "5",
            "4",
            "4",
            "5"
        ],
        "score": 4.4
    }
```
