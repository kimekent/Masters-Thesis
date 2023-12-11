from typing import Dict

class TeamsCards:
    """class to provide cards for MS Teams Channel"""
    def card(self, text) -> Dict:
        card = {
                    "attachments": [
                        {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                                "type": "AdaptiveCard",
                                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                                "version": "1.2",
                                "body": [
                                    {
                                        "type": "TextBlock",
                                        "text": text,
                                        "wrap": "true"
                                    }
                                ]
                        }
                        }
                    ]
                    }
        return card

    def button(self, title, response)  -> Dict:
        button = {
                    "type": "Action.Submit",
                    "title": title,
                    "data": {
                        "msteams": {
                            "type": "imBack",
                            "value": response
                        }
                    }
                }
        return button