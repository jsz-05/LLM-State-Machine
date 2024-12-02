import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type
import openai
from pydantic import BaseModel, ValidationError

from core.utils import _add_transitions, _generate_response_schema
from core.llm_handler import LLMUtilities  # Updated import

from core.state_models import (
    FSMRun,
    FSMState,
    DefaultResponse,
    FSMError,
    ImmediateStateChange,
)

logger = logging.getLogger("llmstatemachine")

class LLMStateMachine:
    """
    Finite State Machine for LLM-driven Agents.
    This class enables the creation of conversational agents by defining states, transitions using structured responses from LLMs.

    Parameters:
    - _state: The current active state of the FSM, representing the ongoing context of the conversation.
    - _initial_state: The starting state of the FSM, used for initialization and resetting.
    - _next_state: The next state to transition to, determined dynamically during execution.
    - _end_state: The terminal state of the FSM, where processing ends (default is "END").
    - _state_registry: A dictionary that stores metadata about all defined states and their transitions.
    - _session_history: A list that records the current session's concise user and assistant interactions.
    - _full_session_history: A comprehensive log of all interactions and state transitions in the session.
    - _running_session_history: A temporary, live record of the conversation for intermediate processing.
    - user_defined_context: A dictionary for storing custom session-specific or user-specific context data.
    - _get_completion: A callable for fetching responses from the LLM, defaulting to `default_get_completion`.
    """

    def __init__(self, initial_state: str, end_state: str = "END"):
        self._state = initial_state
        self._initial_state = initial_state
        self._next_state = None
        self._end_state = end_state
        self._state_registry = {}
        self._session_history = []
        self._full_session_history = []
        self._running_session_history = []
        self.user_defined_context = {}
        self._llm_utils = LLMUtilities()

    def define_state(
        self,
        state_key: str,
        prompt_template: str,
        preprocess_prompt_template: Optional[Callable] = None,
        temperature: float = 0.5,
        transitions: Dict[str, str] = None,
        response_model: Optional[BaseModel] = None,
        preprocess_input: Optional[Callable] = None,
        preprocess_chat: Optional[Callable] = None,
    ):
        """
        Decorator to define and register a state [@fsm.define_state(...)] in the FSM (Finite State Machine).

        This function simplifies the process of associating metadata (such as prompts and transitions) 
        with a Python function that defines the behavior of the state.

        Parameters:
        - state_key (str): A unique identifier for the state.
        - prompt_template (str): Instructions provided to the LLM when this state is active.
        - preprocess_prompt_template (Optional[Callable]): A func to preprocess the sys prompt for the LLM.
        - temperature (float): Determines the randomness of LLM responses, defaults at 0.5.
        - transitions (Dict[str, str], optional): Maps possible next states to their conditions 
                                                       (e.g., {"NEXT_STATE": "If user agrees"}). Defaults to None.
        - response_model (Optional[BaseModel]): A Pydantic model for parsing and validating the LLM's response.
        - preprocess_input (Optional[Callable]): A func to preprocess user input before sending to the LLM.
        - preprocess_chat (Optional[Callable]): A func to preprocess the chat history for the LLM.

        Returns:
        - callable The original function wrapped and registered with the FSM.
        """

        # Empty dictionary for transitions if none is provided
        if transitions is None:
            transitions = {}

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            
            # Register the state in the FSM's registry with the provided metadata
            self._state_registry[state_key] = FSMState(
                key=state_key,
                func=wrapper,
                prompt_template=prompt_template,
                preprocess_prompt_template=preprocess_prompt_template,
                temperature=temperature,
                transitions=transitions,
                response_model=response_model,
                preprocess_input=preprocess_input,
                preprocess_chat=preprocess_chat,
            )
            return wrapper
        return decorator

    async def run_state_machine(
        self,
        async_openai_instance: openai.AsyncOpenAI,
        user_input: str,
        model: str = "gpt-4o",
        *args,
        **kwargs,
    ) -> FSMRun:
        """
        Executes a single step of the Finite State Machine (FSM) using the provided user input.
        Processes the current state, generates a response via the LLM, and transitions the FSM.

        Parameters:
        - async_openai_instance (openai.AsyncOpenAI): OpenAI client for API calls.
        - user_input (str): User input for the FSM.
        - model (str): The LLM model to use for generating responses (default: "gpt-4o").

        Returns:
        - FSMRun: A structured representation of the FSM's state, chat history, and response.
        """

        current_state: FSMState = self._state_registry.get(self._state)
        if not current_state:
            raise FSMError(f"State '{self._state}' not found in the state registry.")

        if current_state.preprocess_input:
            user_input = current_state.preprocess_input(user_input, self) or user_input

        # Prepare chat history to use as context
        chat_history_copy = self._session_history.copy()
        full_session_history_copy = self._full_session_history.copy()
        chat_history_copy.append({"role": "user", "content": user_input})
        full_session_history_copy.append({"role": "user", "content": user_input})

        # Generate response using our LLMUtilities
        response_schema = _generate_response_schema(
            current_state.response_model, 
            current_state.transitions, 
            current_state.key
        )
        
        # Generate a response using the LLM
        response_data = await self._llm_utils.get_completion(
            async_openai_instance,
            chat_history_copy,
            response_schema,
            model,
            current_state
        )

        # Extract response and next state
        next_state_key = response_data.get("next_state_key", current_state.key)
        raw_response = response_data.get("response")

        # Validate response
        if current_state.response_model:
            try:
                parsed_response = current_state.response_model(**raw_response)
            except ValidationError as error:
                raise FSMError(f"Error parsing response: {error}")
        else:
            parsed_response = raw_response.get("content", raw_response)

        # Validate and update next state
        if next_state_key not in self._state_registry:
            next_state_key = current_state.key
        self._next_state = next_state_key

        # Execute state logic
        function_context = {
            "fsm": self,
            "response": parsed_response,
            "will_transition": self._state != self._next_state,
            **kwargs,
        }
        final_response = await current_state.func(**function_context)

        # Handle immediate state changes if needed
        if isinstance(final_response, ImmediateStateChange):
            self._state = final_response.next_state
            return await self.run_state_machine(
                async_openai_instance, 
                final_response.input, 
                model, 
                *args, 
                **kwargs
            )

        # Finalize response (or assigns fallback)
        final_response_str = final_response or parsed_response

        # Update histories
        chat_history_copy.append({"role": "assistant", "content": final_response_str})
        full_session_history_copy.append({"role": "assistant", "content": final_response_str})
        self._session_history = chat_history_copy
        self._full_session_history = full_session_history_copy

        # Transition state
        previous_state = self._state
        self._state = self._next_state
        self._next_state = None

        return FSMRun(
            state=self._state,
            chat_history=chat_history_copy,
            context_data=self.user_defined_context,
            response_raw=response_data,
            response=final_response_str,
        )
    
    def reset(self):
        """Resets the FSM to its initial state."""
        self._state = self._initial_state
        self._next_state = None
        self._session_history = []
        self.user_defined_context = {}

    def get_curr_state(self):
        """Returns the current state of the FSM."""
        return self._state

    def get_next_state(self):
        """Returns the next state of the FSM."""
        return self._next_state

    def set_next_state(self, next_state: str):
        """Sets the next state of the FSM."""
        self._next_state = next_state

    def get_running_session_history(self):
        """Returns the current running chat history."""
        return self._running_session_history

    def set_running_session_history(self, chat_history: list):
        """Sets the current running chat history."""
        self._running_session_history = chat_history

    def get_full_session_history(self):
        """Returns the full chat history of the FSM."""
        return self._full_session_history

    def set_context_data(self, key: str, value: Any):
        """Sets a key-value pair into the user-defined context."""
        self.user_defined_context[key] = value

    def set_context_data_dict(self, data: Dict[str, Any]):
        """Sets multiple key-value pairs into the user-defined context."""
        self.user_defined_context.update(data)

    def get_context_data(self, key: str, default: Any = None):
        """Gets a value from the user-defined context, with a default value."""
        return self.user_defined_context.get(key, default)

    def get_full_context_data(self):
        """Returns the full user-defined context."""
        return self.user_defined_context

    def is_completed(self):
        """Checks if the FSM has reached its final state."""
        return self._state == self._end_state

    def is_urgent_shift(self):
        """Checks if the FSM is in an urgent shift state."""
        return self._is_urgent_shift

