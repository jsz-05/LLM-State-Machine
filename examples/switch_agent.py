import os
from dotenv import load_dotenv
import openai
from core.fsm import LLMStateMachine
from core.state_models import FSMRun

# Global variable to track on-off state
SWITCH_STATE = "OFF"

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

# Create the FSM
fsm = LLMStateMachine(initial_state="START", end_state="END")


# Define the START state
@fsm.define_state(
    state_key="START",
    prompt_template="You are an on-off switcher. Ask the user if they want to turn switch the on or off.",
    transitions={"STATE_ON": "If user wants to turn on the switch", "END": "If user wants to end the conversation"},
)
async def start_state(fsm: LLMStateMachine, response: str, will_transition: bool):
    global SWITCH_STATE
    if will_transition and fsm.get_next_state() == "STATE_ON":
        SWITCH_STATE = "ON"
        print("SWITCH TURNED ON")
    elif will_transition and fsm.get_next_state() == "END":
        return "Goodbye!"
    return response


# Define the STATE_ON state
@fsm.define_state(
    state_key="STATE_ON",
    prompt_template="The switch is now on. Ask the user if they want to turn off the switch or end the conversation.",
    transitions={"START": "If user wants to turn off the switch", "END": "If user wants to end the conversation"},
)
async def state_on(fsm: LLMStateMachine, response: str, will_transition: bool):
    global SWITCH_STATE
    if will_transition and fsm.get_next_state() == "START":
        SWITCH_STATE = "OFF"
        print("SWITCH TURNED OFF")
    elif will_transition and fsm.get_next_state() == "END":
        return "Goodbye!"
    return response


# Define the END state
@fsm.define_state(
    state_key="END",
    prompt_template="Goodbye!",
)
async def end_state(fsm: LLMStateMachine, response: str, will_transition: bool):
    return "Goodbye!"


async def main():
    """Example of a simple on-off switch FSM using LLMStateMachine"""
    # Create the OpenAI client
    openai_client = openai.AsyncOpenAI()

    print("Agent: Hi. I am an on-off switch manager.")
    while not fsm.is_completed():  # Run until FSM reaches the END state
        user_input = input("Your input: ")
        if user_input.lower() in ["quit", "exit"]:
            fsm.set_next_state("END")
            break
        run_state: FSMRun = await fsm.run_state_machine(openai_client, user_input=user_input)
        print(f"Agent: {run_state.response}")
        print("CURRENT SWITCH STATE:", SWITCH_STATE)

    print("Agent: Goodbye.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
