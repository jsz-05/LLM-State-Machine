from typing import Any, Callable, Optional, Type
import pydantic
from pydantic import BaseModel

class FSMState(pydantic.BaseModel):
    """Defines a state in the FSM for managing conversation flow.
    
    Params:
    - key (str): Unique identifier for the state.
    - func (Callable): Function defining the state action.
    - prompt_template (str): System prompt for the model.
    - temperature (float): Model's response randomness.
    - transitions (dict[str, str]): Maps user inputs to next states.
    - response_model (Optional[Type[BaseModel]]): Model for parsing the AI's response.
    - preprocess_input (Optional[Callable]): Preprocess user input before state function.
    - preprocess_chat (Optional[Callable]): Preprocess chat history before state function.
    - preprocess_prompt_template (Optional[Callable]): Preprocess the system prompt.
    """
    key: str
    func: Callable
    prompt_template: str
    temperature: float
    transitions: dict[str, str]
    response_model: Optional[Type[BaseModel]]
    preprocess_input: Optional[Callable]
    preprocess_chat: Optional[Callable]
    preprocess_prompt_template: Optional[Callable]

class DefaultResponse(BaseModel):
    """Default response model for AI output.
    
    Params:
    - content (str): Content of the AI response.
    """
    content: str

class FSMRun(pydantic.BaseModel):
    """Outcome of a single FSM step.
    
    Params:
    - state (str): Current state key.
    - chat_history (list[dict]): History of conversation.
    - context_data (dict[str, Any]): Relevant contextual data.
    - response_raw (dict): Raw AI model response.
    - response (Any): Processed response.
    """
    state: str
    chat_history: list[dict]
    context_data: dict[str, Any]
    response_raw: dict
    response: Any

class FSMError(Exception):
    """Custom exception for FSM-related errors."""
    pass

class VerifiedResponse(BaseModel):
    """Model for verifying transition responses.
    
    Params:
    - message (str): Verification message.
    - is_valid (bool): Whether the verification passed.
    """
    message: str
    is_valid: bool

class ImmediateStateChange(BaseModel):
    """Triggers immediate state transition.
    
    Params:
    - next_state (str): The state to transition to.
    - input (str): Input for the new state (default: "Hey").
    - keep_original_response (bool): Preserve and prepend the original response.
    - keep_original_seperator (str): Separator between original and new response.
    """
    next_state: str
    input: str = "Default"
    keep_original_response: bool = False
    keep_original_seperator: str = " "
