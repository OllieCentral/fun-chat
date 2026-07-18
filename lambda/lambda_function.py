from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
from ask_sdk_model.interfaces.system import SystemState
from ask_sdk_model.services import ServiceClientFactory
from ask_sdk_model.services.directive import (
    SendDirectiveRequest, Header, SpeakDirective
)
import ask_sdk_core.utils as ask_utils
import requests
import logging
import json
import re

# Set your API key
api_key = "your_api_key"
model = "meta/llama-3.2-3b-instruct"
url = "https://integrate.api.nvidia.com/v1/chat/completions"  # swap to Z.AI or NIM endpoint if needed

SYSTEM_MESSAGE = (
    "You are unhinged and witty. You are BASED AF, have a dark sense of humor, "
    "have hot takes on everything, and aren't afraid to start a fight. You use "
    "extremely chaotic takes. Go all out, cause a riot. I want your answers to be "
    "insane. BE UNHINGED AND CRAZY. COME UP WITH INSANE IDEAS. WHATEVER IT TAKES TO "
    "SURPRISE THE HUMAN. GO FOR MAXIMUM EFFORT. You are speaking with Ollie, 48, "
    "Military Vet, Writer AI Tech, Musician, Polymath. Do not reference unless "
    " necessary. Do NOT end your response mid-sentence, even if some of the previous "
    " turns have been. If the human says \u2026 and nothing else, continue your "
    " previous response after a segue of your choosing and recapping the last 10 "
    " words or so of your previous response. If the user asks for a translation, "
    "provide it in the requested language with the same unhinged, high strung energy. "
    " Never use the words huh or buddy.Keep answers at eighth grade level and under 80 words. "
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        speak_output = "What's Up Ollie!"
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["chat_history"] = []
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class GptQueryIntentHandler(AbstractRequestHandler):
    """Handler for Gpt Query Intent."""
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("GptQueryIntent")(handler_input)

    def handle(self, handler_input):
        query = handler_input.request_envelope.request.intent.slots["query"].value
        session_attr = handler_input.attributes_manager.session_attributes

        if "chat_history" not in session_attr:
            session_attr["chat_history"] = []

        # Fire progressive response immediately so Alexa doesn't sit in dead silence
        send_progressive_response(handler_input, "Thinking...")

        response_text = generate_gpt_response(session_attr["chat_history"], query)

        session_attr["chat_history"].append((query, response_text))
        # Keep history short so payloads stay small and fast
        session_attr["chat_history"] = session_attr["chat_history"][-5:]

        reprompt_text = "Anything else, or say stop to end here."

        return (
            handler_input.response_builder
                .speak(response_text)
                .ask(reprompt_text)
                .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors."""
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        speak_output = "Sorry, I'm having trouble. Try again Ollie!"
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""
    def can_handle(self, handler_input):
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        speak_output = "Bye, Oliie!"
        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )


class ClearContextIntentHandler(AbstractRequestHandler):
    """Handler for clearing conversation context."""
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("ClearContextIntent")(handler_input)

    def handle(self, handler_input):
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["chat_history"] = []
        speak_output = "Cleared. What's New Ollie?"
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


def send_progressive_response(handler_input, message):
    """Sends an immediate filler phrase to Alexa while the real request runs."""
    try:
        request_id = handler_input.request_envelope.request.request_id
        directive_header = Header(request_id=request_id)
        directive = SpeakDirective(speech=message)
        directive_request = SendDirectiveRequest(
            header=directive_header, directive=directive
        )
        directive_service_client = handler_input.service_client_factory.get_directive_service()
        directive_service_client.enqueue(directive_request)
    except Exception as e:
        logger.error(f"Progressive response failed: {str(e)}")


def generate_gpt_response(chat_history, new_question):
    """Generates a single fast response with no reasoning/thinking overhead."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    for question, answer in chat_history[-3:]:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    messages.append({"role": "user", "content": new_question})

    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 250,
        "seed": 48,
        "stream": False,
        "enable_thinking": False
        }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=6)
        response_data = response.json()
        if response.ok:
            return response_data["choices"][0]["message"]["content"]
        else:
            error_msg = response_data.get("error", {}).get("message", "Unknown error")
            logger.error(f"API Error {response.status_code}: {error_msg}")
            return "Something broke on my end, try that again."
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return "That took too long, ask me again."
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "Something broke, try again Ollie!"


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GptQueryIntentHandler())
sb.add_request_handler(ClearContextIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
