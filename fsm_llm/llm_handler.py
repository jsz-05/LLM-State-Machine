from typing import Callable, Dict, Any, Optional, Type
from pydantic import BaseModel
import openai
import jinja2
from .state_models import FSMError, FSMState
from .utils import _add_transitions

class LLMUtilities:
    """Handles all LLM-related operations including prompt processing and API calls."""
    
    @staticmethod
    async def get_completion(
        async_openai_instance: openai.AsyncOpenAI,
        chat_history: list,
        response_model: Type[BaseModel],
        llm_model: str,
        current_state: Optional[FSMState] = None,
    ) -> dict:
        """Get completion from LLM with optional state-specific processing"""
        if current_state:
            # Process state-specific prompt and chat history
            processed_prompt = LLMUtilities.process_prompt_template(
                current_state.prompt_template,
                getattr(current_state, 'user_defined_context', {}),
                current_state.preprocess_prompt_template
            )
            processed_prompt = _add_transitions(processed_prompt, current_state)
            
            system_message = {"role": "system", "content": processed_prompt}
            chat_history = [system_message] + chat_history
            
            if current_state.preprocess_chat:
                chat_history = current_state.preprocess_chat(chat_history)
        
        # Execute LLM call
        completion = await async_openai_instance.beta.chat.completions.parse(
            model=llm_model,
            messages=chat_history,
            response_format=response_model,
        )
        
        message = completion.choices[0].message
        if not message.parsed:
            raise FSMError(f"Error in parsing the completion: {message.refusal}")
            
        return message.parsed.model_dump()

    @staticmethod
    def process_prompt_template(
        prompt_template: str,
        context: Dict[str, Any],
        preprocess_prompt_template: Optional[Callable] = None,
    ) -> str:
        """Process the system prompt with Jinja2 templates and optional pre-processing"""
        # Pre-process system prompt with Jinja2
        template = jinja2.Template(prompt_template)
        processed_prompt = template.render(context)

        if preprocess_prompt_template:
            processed_prompt = (
                preprocess_prompt_template(processed_prompt) 
                or processed_prompt
            )

        return processed_prompt

    @staticmethod
    def process_chat_history(
        chat_history: list,
        preprocess_chat: Optional[Callable] = None,
        fsm_instance = None,
    ) -> list:
        """Process chat history with optional pre-processing function"""
        if preprocess_chat:
            chat_history = preprocess_chat(chat_history, fsm_instance)
        return chat_history