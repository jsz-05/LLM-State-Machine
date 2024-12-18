import os
from dotenv import load_dotenv
import openai
from fsm_llm import LLMStateMachine
from fsm_llm.state_models import FSMRun
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

# Initialize the FSM
fsm = LLMStateMachine(initial_state="show_content", end_state="END")

# Global variables to track the learning state
LEARNING_STATE = {
    "current_content_id": 1,  # Tracks the ID of the content the user is currently on
}

# Actions
USER_ACTIONS = ["ua_next", "ua_ask_clarifying_content", "ua_ask_clarifying_example"]
SYSTEM_ACTIONS = ["sa_show_content", "sa_show_example", "sa_show_quiz"]

CONTENT_FILE = "../content/calculus_content.json"
# Function to load content dynamically from a file
def load_content(content_id, file_path=CONTENT_FILE):
    try:
        with open(file_path, "r") as file:
            content_data = json.load(file)  # Assumes content is stored as JSON

        # Loop through the list to find the matching content ID
        for item in content_data:
            if item.get("id") == str(content_id):
                return item.get("content", "Content field not found.")  # Return the 'content' field

        return "Content not found."  # Default if ID not found
    except json.JSONDecodeError:
        return "Error: Failed to decode JSON."
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error loading content: {e}"



EXAMPLE_FILE = "../content/calculus_example.json"
# Function to load content dynamically from a file
def load_example(content_id, file_path=EXAMPLE_FILE):
    try:
        with open(file_path, "r") as file:
            example_data = json.load(file)  # Assumes content is stored as JSON

        # Loop through the list to find the matching content ID
        for item in example_data:
            if item.get("content_id") == str(content_id):
                return item.get("example", "example field not found.")  # Return the 'content' field

        return "Example not found."  # Default if ID not found
    except json.JSONDecodeError:
        return "Error: Failed to decode JSON."
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error loading content: {e}"



# Define the `show_content` state
@fsm.define_state(
    state_key="show_content",
    prompt_template=f"You are a tutor. Present the content for topic {load_content(LEARNING_STATE['current_content_id'])}.",
    transitions={
        "show_content": "If the user wants to move to the next section.",
        "show_example": "If the user asks for an example.",
        "quiz": "If the user asks for a quiz.",
    },
)
async def show_content_state(fsm: LLMStateMachine, response: str):
    if fsm.get_next_state() == "show_example":
        return f"Here's an example for content ID {load_example(LEARNING_STATE['current_content_id'])}."
    elif fsm.get_next_state() == "quiz":
        return f"Let's test your knowledge on content ID {LEARNING_STATE['current_content_id']}."
    LEARNING_STATE['current_content_id']+= 1;
    return f"Here's the content for topic {LEARNING_STATE['current_content_id']}."


# Define the `show_example` state
@fsm.define_state(
    state_key="show_example",
    prompt_template=f"You are a tutor. Provide an example for topic {load_example(LEARNING_STATE['current_content_id'])}.",
    transitions={
        "show_content": "If the user asks for more content.",
        "show_example": "If the user asks for another example.",
        "quiz": "If the user asks for a quiz.",
    },
)
async def show_example_state(fsm: LLMStateMachine, response: str):
    if fsm.get_next_state() == "show_content":
        LEARNING_STATE['current_content_id']+= 1;
        return f"Returning to the content for topic {load_content(LEARNING_STATE['current_content_id'])}."
    elif fsm.get_next_state() == "quiz":
        return f"Let's test your knowledge on content ID {LEARNING_STATE['current_content_id']}."
    return f"Here's another example for content ID {LEARNING_STATE['current_content_id']}."


# Define the `quiz` state
@fsm.define_state(
    state_key="quiz",
    prompt_template="You are a tutor. Create a quiz for topic {LEARNING_STATE['current_content_id']}.",
    transitions={
        "show_content": "If the user asks for more content.",
        "show_example": "If the user asks for another example.",
        "quiz": "If the user wants another quiz.",
    },
)
async def quiz_state(fsm: LLMStateMachine, response: str):
    if fsm.get_next_state() == "show_content":
        return f"Returning to the content for topic {load_content(LEARNING_STATE['current_content_id'])}."
    elif fsm.get_next_state() == "show_example":
        return f"Here's an example for content ID {load_example(LEARNING_STATE['current_content_id'])}."
    return f"Here's your quiz for content ID {LEARNING_STATE['current_content_id']}."


# Define the END state
@fsm.define_state(
    state_key="END",
    prompt_template="The learning session has concluded. Goodbye!",
)
async def end_state(fsm: LLMStateMachine, response: str):
    return "Thank you for learning! Goodbye!"


# Simulated interaction loop
async def main():
    """Simulates a learning session with the tutor agent."""
    import random

    # Create the OpenAI client
    openai_client = openai.AsyncOpenAI()

    print("Tutor: Welcome to the learning session!")
    while not fsm.is_completed():
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            fsm.set_next_state("END")
            break

        # Simulate user and system actions
        user_action = random.choice(USER_ACTIONS) if user_input.strip() == "" else user_input
        if user_action in USER_ACTIONS:
            print(f"[User Action Triggered: {user_action}]")
            if user_action == "ua_next":
                fsm.set_next_state("show_content")
            elif user_action == "ua_ask_clarifying_content":
                fsm.set_next_state("show_content")
            elif user_action == "ua_ask_clarifying_example":
                fsm.set_next_state("show_example")
        else:
            system_action = random.choice(SYSTEM_ACTIONS)
            print(f"[System Action Triggered: {system_action}]")
            if system_action == "sa_show_content":
                fsm.set_next_state("show_content")
            elif system_action == "sa_show_example":
                fsm.set_next_state("show_example")
            elif system_action == "sa_show_quiz":
                fsm.set_next_state("quiz")

        run_state: FSMRun = await fsm.run_state_machine(openai_client, user_input=user_input)
        print(f"Tutor: {run_state.response}")
        print("CURRENT CONTENT ID:", LEARNING_STATE["current_content_id"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
