# FSM-based LLM Conversational Agents

This project provides a framework for creating conversational agents using a Finite State Machine (FSM) powered by Large Language Models (LLMs) like OpenAI GPT.
## Features

- Define states and transitions for your agent using a simple decorator.
- Handle dynamic conversation flow with flexible state management.
- Integrates with OpenAI’s GPT models to generate responses based on state context.

## Installation

1. Clone the repo:
   ```
   git clone https://github.com/jsz-05/LLM-State-Machine.git
   cd LLM-State-Machine
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key
     OPENAI_ORGANIZATION=your-organization-id
     ```

## Usage

1. Define your states using the `@fsm.define_state` decorator:
   ```python
   @fsm.define_state(
       state_key="START",
       prompt_template="Welcome! How can I assist you today?",
       transitions={"NEXT_STATE": "When the user wants to continue."}
   )
   async def start_state(fsm: LLMStateMachine, response: str, will_transition: bool):
       # Your logic for the state here
       return response
   ```

2. Run your agent:
   ```python
   if __name__ == "__main__":
       import asyncio
       asyncio.run(main())  # Starts the FSM-based agent
   ```

## Examples

- **Light Switch Agent**: A simple agent that asks the user whether they want to turn a light on or off.
- **Customer Support Agent**: A bot that collects user details and assists with customer queries.
- **Medical Triage Agent**: A complex agent that helps assess if a medical situation is an emergency and collects patient data.

<!-- ## Contributing
Feel free to fork, star, and create pull requests. Contributions are welcome! -->
