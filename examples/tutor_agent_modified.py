import os
from dotenv import load_dotenv
import openai
from fsm_llm import LLMStateMachine
from fsm_llm.state_models import FSMRun
import json
import jinja2

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

# Keep track of last user input globally
LAST_USER_INPUT = ""

# Actions
USER_ACTIONS = ["ua_next", "ua_ask_clarifying_content", "ua_ask_clarifying_example"]
SYSTEM_ACTIONS = ["sa_show_content", "sa_show_example", "sa_show_quiz"]

CONTENT_FILE = "content/calculus_content.json"
EXAMPLE_FILE = "content/calculus_example.json"

def load_content(content_id, file_path=CONTENT_FILE):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content_data = json.load(file)
        for item in content_data:
            if item.get("id") == str(content_id):
                return item.get("content", "Content field not found.")
        return "Content not found."
    except json.JSONDecodeError:
        return "Error: Failed to decode JSON."
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error loading content: {e}"

def load_example(content_id, file_path=EXAMPLE_FILE):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            example_data = json.load(file)
        for item in example_data:
            if item.get("content_id") == str(content_id):
                return item.get("example", "Example field not found.")
        return "Example not found."
    except json.JSONDecodeError:
        return "Error: Failed to decode JSON."
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error loading example: {e}"

SHOW_CONTENT_TEMPLATE = """
You are a friendly and helpful calculus tutor.
The user said: "{{ user_input }}"

Current Topic ID: {{ topic_id }}
Content for this topic:
{{ topic_content }}

Explain this content in a helpful way. If the user wants more content, you can move to show_content. 
If they want an example, move to show_example.
If they want a quiz, move to quiz.
If they want to end, move to END.

Include the content above in your explanation to the user.
"""

SHOW_EXAMPLE_TEMPLATE = """
You are a friendly and helpful calculus tutor.
The user said: "{{ user_input }}"

Current Topic ID: {{ topic_id }}
Previously shown content:
{{ topic_content }}

Example for this topic:
{{ topic_example }}

Explain the example and how it relates to the content. If the user wants more content, move to show_content.
If they want another example, move to show_example.
If they want a quiz, move to quiz.
If they want to end, move to END.
"""

QUIZ_TEMPLATE = """
You are a friendly and helpful calculus tutor.
The user said: "{{ user_input }}"

Current Topic ID: {{ topic_id }}
Previously shown content:
{{ topic_content }}

Please create a short quiz related to the above content. Include a few questions and maybe some hints.
If the user wants more content after this, move to show_content.
If they want an example, move to show_example.
If they want another quiz, move to quiz.
If they want to end, move to END.
"""

END_TEMPLATE = "The learning session has concluded. Goodbye!"

def preprocess_prompt_template(processed_prompt: str) -> str:
    """Dynamically fill in user input and content/example before sending to LLM."""
    topic_id = LEARNING_STATE["current_content_id"]
    topic_content = load_content(topic_id, CONTENT_FILE)
    topic_example = load_example(topic_id, EXAMPLE_FILE)
    user_input = LAST_USER_INPUT

    # Use Jinja2 to render the template dynamically
    template = jinja2.Template(processed_prompt)
    rendered = template.render(
        user_input=user_input,
        topic_id=topic_id,
        topic_content=topic_content,
        topic_example=topic_example
    )
    return rendered

# Define the `show_content` state
@fsm.define_state(
    state_key="show_content",
    prompt_template=SHOW_CONTENT_TEMPLATE,
    transitions={
        "show_content": "If the user wants to move to the next section.",
        "show_example": "If the user asks for an example.",
        "quiz": "If the user asks for a quiz.",
        "END": "If the user wants to end the session."
    },
    preprocess_prompt_template=preprocess_prompt_template
)
async def show_content_state(fsm: LLMStateMachine, response: str, will_transition: bool):
    # If we are going to show_content again, increment the content_id
    if will_transition and fsm.get_next_state() == "show_content":
        LEARNING_STATE['current_content_id'] += 1
    # Return the LLM's response directly, which should now contain the content
    return response

# Define the `show_example` state
@fsm.define_state(
    state_key="show_example",
    prompt_template=SHOW_EXAMPLE_TEMPLATE,
    transitions={
        "show_content": "If the user asks for more content.",
        "show_example": "If the user asks for another example.",
        "quiz": "If the user asks for a quiz.",
        "END": "If the user wants to end the session."
    },
    preprocess_prompt_template=preprocess_prompt_template
)
async def show_example_state(fsm: LLMStateMachine, response: str, will_transition: bool):
    if will_transition and fsm.get_next_state() == "show_content":
        LEARNING_STATE['current_content_id'] += 1
    return response

# Define the `quiz` state
@fsm.define_state(
    state_key="quiz",
    prompt_template=QUIZ_TEMPLATE,
    transitions={
        "show_content": "If the user asks for more content.",
        "show_example": "If the user asks for another example.",
        "quiz": "If the user wants another quiz.",
        "END": "If the user wants to end the session."
    },
    preprocess_prompt_template=preprocess_prompt_template
)
async def quiz_state(fsm: LLMStateMachine, response: str, will_transition: bool):
    # If transitioning to show_content, increment
    if will_transition and fsm.get_next_state() == "show_content":
        LEARNING_STATE['current_content_id'] += 1
    return response

# Define the END state
@fsm.define_state(
    state_key="END",
    prompt_template=END_TEMPLATE
)
async def end_state(fsm: LLMStateMachine, response: str):
    return "Thank you for learning! Goodbye!"

# Simulated interaction loop
async def main():
    """Simulates a learning session with the tutor agent."""
    import random
    openai_client = openai.AsyncOpenAI()

    print("Tutor: Welcome to the learning session!")
    while not fsm.is_completed():
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            fsm.set_next_state("END")
            break

        global LAST_USER_INPUT
        LAST_USER_INPUT = user_input

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
