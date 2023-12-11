"""
This files contains custom actions which can be used to run custom Python code.
When running the bot on MS Teams replace this file with the content from 'action_teams.py'.
"""
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.events import FollowupAction
# For email action
from email.message import EmailMessage
import ssl
import smtplib
import win32com.client as win32


class ActionQuestionAnswered(Action):
    # Ask user if they would like to create a websupport ticket with their question. This is a backup in case answer was
    # not satisfactory.

    def name(self) -> Text:
        return "action_question_answered"  # action name

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        buttons = []

        buttons.append({"title": "Erstelle ein Websupport-Ticket mit meiner Frage", "payload": "/open_ticket"})

        dispatcher.utter_message(buttons=buttons)

        return []


class ActionEmail(Action):
    # In case bot can't answer the user question and user wants to create a websupport ticket, question is sent via email
    # to websupport. This will automatically create a ticket.

    def name(self) -> Text:
        return "action_email"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get last user utterance to set as body of email.
        user_input = []
        for event in reversed(tracker.events):
            if event.get("event") == "user":
                # Filter out the intent "/open_ticket", which is also saved as user utterance.
                if event['text'] != "/open_ticket":
                    user_input.append(event['text'])
                    if len(user_input) == 1:
                        break

        # Check if the 'new_email_message' slot is set
        email_body = " ".join(user_input)

        try:
            # raise Exception("Forced error for testing") # uncomment to test except block
            # This try block opens Outlook with the body of the email being the question asked.
            outlook = win32.Dispatch("Outlook.Application")  # Starts Outlook application
            new_email = outlook.CreateItem(0)  # Creates new email item
            new_email.To = "mefabe7562@marksia.com"  # for testing purposes a temporary email address is used
            new_email.Body = email_body  # body of email
            new_email.Display(False)

            dispatcher.utter_message(
                "Deine Anfrage wurde erfolgreich in eine Outlook-E-Mail übertragen. Dort kannst du "
                "deine Anfrage noch bearbeiten, bevor du sie dem Websupport schickst.")
            return [SlotSet("website", None), SlotSet("message", None)]

        except Exception as e:
            print(f"Error occurred: {e}")
            # In case of error in try block, start process of sending email from inside bot.
            return [FollowupAction("action_ask_confirmation")]


class ActionAskConfirmation(Action):
    # If bot couldn't create Outlook mail with last user utterance. Email will be sent from within the bot. Before sending
    # the email, user is given the chance to edit email message.
    def name(self) -> Text:
        return "action_ask_confirmation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # This slot gets filled, when user requests to edit ticket before sending it.
        # The new message is saved in 'message'.
        new_email_message = tracker.get_slot("message")

        # Ask for confirmation of the new email message.
        if new_email_message is not None:
            message = f"Das ist der Inhalt des Tickets:\n\n{new_email_message}\n\nMöchtest du den Inhalt des Mails vor dem Abschicken noch bearbeiten?"
        else:
            # If 'new_email_message' is not set, use last user input.
            last_user_input = []
            l_intents = ["/email_not_confirmed", "/email_confirmed", "open_ticket"]
            for event in reversed(tracker.events):
                if event.get("event") == "user":
                    # Filter out intent messages, that are also saved in the user events list.
                    if event['text'] not in l_intents:
                        last_user_input.append(event['text'])
                        if len(last_user_input) == 1:
                            break

            user_input = " ".join(last_user_input)
            message = f"Das ist der Inhalt des Tickets:\n\n{user_input}\n\nMöchtest du den Inhalt des Mails vor dem Abschicken noch bearbeiten?"

        buttons = [{"title": "Ja", "payload": "/email_not_confirmed"},
                   {"title": "Nein", "payload": "/email_confirmed"}]

        dispatcher.utter_message(message)
        dispatcher.utter_message(buttons=buttons)

        return []


class ActionAskNewMessage(Action):
    # If user wants to edit message before sending email from within bot, bot prompts for new message.
    def name(self) -> Text:
        return "action_ask_new_message"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("Bitte gib deine neue Websupport-Nachricht ein.")
        return [SlotSet("send_new_message", "true")]


class ActionEmail(Action):
    # If bot cant open Outlook. The email will be sent from within the bot.
    def name(self) -> Text:
        return "action_backup_mail"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Check if the 'new_email_message' slot is set
        new_email_message = tracker.get_slot("message")

        if new_email_message is not None:
            email_body = new_email_message

        else:
            l_intents = ["/email_not_confirmed", "/email_confirmed", "open_ticket"]
            # Retrieve user inputs from the slot
            user_input = []
            for event in reversed(tracker.events):
                if event.get("event") == "user":
                    # Filter out intent messages, that are also saved in the user events list.
                    if event['text'] not in l_intents:
                        user_input.append(event['text'])
                        if len(user_input) == 1:
                            break
            email_body = " ".join(user_input)

        email_sender = "kimberly.kent.twitter@gmail.com"
        email_password = "mihtepktfliycluz"
        email_receiver = "mefabe7562@marksia.com"
        subject = "subject line"
        website = tracker.get_slot("website")
        body = email_body

        # Modify the body to include "Website:" before the URL
        if website:
            body += f"\nWebsite: {website}"

        em = EmailMessage()
        em["From"] = email_sender
        em["To"] = email_receiver
        em["subject"] = subject
        em.set_content(body)

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(email_sender, email_password)
            smtp.sendmail(email_sender, email_receiver, em.as_string())

        # After sending the email, set the "last_email_time" slot
        return [SlotSet("website", None), SlotSet("message", None)]


class ActionDefaultAskAffirmation(Action):
    """Asks for an affirmation of the intent if NLU threshold is not met."""

    def name(self) -> Text:
        return "action_default_ask_affirmation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get intent ranking
        intent_ranking = tracker.latest_message.get('intent_ranking', [])

        # Check if we have at least two intents
        if len(intent_ranking) >= 2:
            # Get the top two intents
            top_intent = intent_ranking[0].get('name')
            print("top_intet: " + top_intent)
            second_intent = intent_ranking[1].get('name')
            print("second_intet: " + second_intent)
            diff_intent_confidence = intent_ranking[0].get("confidence") - intent_ranking[1].get("confidence")
            print("diff_condf: " + str(diff_intent_confidence))
            if diff_intent_confidence < 0.2:
                if top_intent == "site_delete" and second_intent == "site_delete_en":
                    question = "Möchtest du Informationen zum Löschen einer Website?"
                    buttons = [{"title": "Nein, gib mir Informationen zum Löschen einer Sprachversion",
                                "payload": "/site_delete_en"}]

                elif top_intent == "site_delete_en" and second_intent == "site_delete":
                    question = "Möchtest du Informationen zum Löschen einer englischen Sprachversion einer Website?"
                    buttons = [
                        {"title": "Nein, gib mir Informationen zum Löschen einer Seite", "payload": "/site_delete"}]

                dispatcher.utter_message(text=question)
                dispatcher.utter_message(buttons=buttons)