import json
from typing import Callable, Dict, Any, Literal, Optional, Type, Union
from pydantic import BaseModel, create_model

from core.state_models import FSMState, DefaultResponse


def _generate_response_schema(
    current_state_model: Union[Type[BaseModel], None],
    transitions: dict[str, str],
    default_state: str,
) -> Type[BaseModel]:
    """Create a response model based on the current state model and transitions, this will be used for structured_response openai param."""

    # Extract the transition keys as a tuple for the Literal type
    transition_keys = tuple([default_state] + list(transitions.keys()))

    next_state_key_type = Literal.__getitem__(transition_keys)

    if not current_state_model:
        current_state_model = DefaultResponse

    # Dynamically create the model with response and next_state_key fields
    return create_model(
        "EnclosedResponse",
        response=(current_state_model, ...),
        next_state_key=(next_state_key_type, ...),
    )


def _add_transitions(prompt_template: str, fsm_state: FSMState) -> str:
    """Add transitions to the system prompt."""
    prompt_template += f"\n\nYou are currently in {fsm_state.key} and based on user input, you can transition to the following states (with conditions defined):"
    for key, value in fsm_state.transitions.items():
        prompt_template += f"\n- {key}: {value}"

    prompt_template += "\n\nIn response add the state you want to transition to.. (or leave blank to stay in the current state)"
    return prompt_template


def wrap_into_json_response(data: BaseModel, next_state: str) -> BaseModel:
    dict_res = {"response": data.model_dump(), "next_state_key": next_state}

    return json.dumps(dict_res)
