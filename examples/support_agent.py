import os
from dotenv import load_dotenv
from pydantic import BaseModel
import openai
from core.fsm import LLMStateMachine
from core.state_models import FSMRun, DefaultResponse

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

# Create the FSM
fsm = LLMStateMachine(initial_state="START", end_state="END")

# Define the response model for user identification
class UserIdentificationResponse(BaseModel):
    user_name: str
    phone_number: str

class ConfirmationResponse(BaseModel):
    confirmation: str  # Expect "yes" or "no"

# Define the START state
@fsm.define_state(
    state_key="START",
    prompt_template=(
        "You are a customer support bot. Your first task is to ask the user for their "
        "name and phone number. Please ensure the user provides both details before proceeding."
    ),
    response_model=UserIdentificationResponse,
    transitions={"CONFIRM": "Once the user provides their name and phone number"},
)
async def start_state(
    fsm: LLMStateMachine, response: UserIdentificationResponse, will_transition: bool
):
    if will_transition and fsm.get_next_state() == "CONFIRM":
        # Store user details in context
        fsm.set_context_data(
            "verified_user",
            {"user_name": response.user_name, "phone_number": response.phone_number},
        )
        return (
            f"Thank you! You provided the following details:\n"
            f"Name: {response.user_name}\nPhone Number: {response.phone_number}\n"
            f"Is this information correct? (yes/no)"
        )
    return "Please provide your name and phone number."



# Define the CONFIRM state
@fsm.define_state(
    state_key="CONFIRM",
    prompt_template="Please confirm the information you provided. Reply with 'yes' or 'no'.",
    response_model=ConfirmationResponse,
    transitions={
        "IDENTIFIED": "If the user confirms the details are correct",
        "START": "If the user indicates the details are incorrect",
    },
)
async def confirm_state(
    fsm: LLMStateMachine, response: ConfirmationResponse, will_transition: bool
):
    if response.confirmation.lower() == "yes":
        fsm.set_next_state("IDENTIFIED")
        return "Thank you for confirming your details. How can I help you?"
    elif response.confirmation.lower() == "no":
        fsm.set_next_state("START")
        return "Let's try again. Please provide your name and phone number."
    else:
        return "Invalid response. Please reply with 'yes' or 'no'."



# Define the IDENTIFIED state
@fsm.define_state(
    state_key="IDENTIFIED",
    prompt_template=(
        "Thank you for identifying yourself. Is there anything else you need help with?"
    ),
    response_model=DefaultResponse,
    transitions={"END": "When the user indicates the conversation is over"},
)
async def identified_state(fsm: LLMStateMachine, response: DefaultResponse, will_transition: bool):
    if will_transition and fsm.get_next_state() == "END":
        return "Thank you! Have a great day!"
    return response.content or "You have been identified successfully. How can I assist you further?"



# Define the END state
@fsm.define_state(
    state_key="END",
    prompt_template="Thank you! Goodbye.",
    response_model=DefaultResponse,
)
async def end_state(fsm: LLMStateMachine, response: DefaultResponse, will_transition: bool):
    return "Goodbye! If you need further assistance, feel free to reach out again."

async def main():
    openai_client = openai.AsyncOpenAI()

    print("Agent: Hello! I am your customer service assistant. Say something to get started.")
    while not fsm.is_completed():  # Run until FSM reaches the END state
        user_input = input("Your input: ")
        if user_input.lower() in ["quit", "exit"]:
            fsm.set_next_state("END")
            break
        run_state: FSMRun = await fsm.run_state_machine(openai_client, user_input=user_input)
        print(f"Agent: {run_state.response}")

    print("Agent: Conversation ended. Thank you!")



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
