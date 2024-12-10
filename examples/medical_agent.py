import os
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv
from fsm_llm.fsm import LLMStateMachine
from fsm_llm.state_models import FSMRun, DefaultResponse, ImmediateStateChange

# Load environment variables
load_dotenv()

class Severity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class VitalSigns(BaseModel):
    blood_pressure: Optional[str]
    heart_rate: Optional[int]
    temperature: Optional[float]
    oxygen_saturation: Optional[int]
    respiratory_rate: Optional[int]

class Symptom(BaseModel):
    name: str
    severity: Severity
    duration: str
    description: str

class PatientInfo(BaseModel):
    name: str
    age: int
    gender: str
    existing_conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)

class EmergencyAssessment(BaseModel):
    is_emergency: bool
    reasoning: str
    recommended_action: str

class DrugInteraction(BaseModel):
    severity: Severity
    description: str
    recommendation: str

class SymptomAssessment(BaseModel):
    symptoms: List[Symptom]
    potential_causes: List[str]
    risk_factors: List[str]
    additional_questions: List[str]

class TreatmentPlan(BaseModel):
    primary_recommendations: List[str]
    lifestyle_modifications: List[str]
    follow_up_timeline: str
    warning_signs: List[str]
    emergency_conditions: List[str]

# Initialize FSM
fsm = LLMStateMachine(initial_state="INITIAL_TRIAGE", end_state="END")

@fsm.define_state(
    state_key="INITIAL_TRIAGE",
    prompt_template="""
    You are an advanced medical triage system. First, assess if this is an immediate emergency requiring urgent care.
    Look for red flags such as:
    - Chest pain, difficulty breathing, severe bleeding
    - Stroke symptoms (FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency)
    - Severe allergic reactions
    - Loss of consciousness
    - Severe head injuries
    
    Based on the user's initial complaint, determine if immediate emergency response is needed.
    """,
    response_model=EmergencyAssessment,
    transitions={
        "EMERGENCY": "If immediate medical attention is required",
        "GATHER_PATIENT_INFO": "If situation is not immediately life-threatening",
    }
)
async def initial_triage(
    fsm: LLMStateMachine,
    response: EmergencyAssessment,
    will_transition: bool
) -> str:
    fsm.set_context_data("emergency_assessment", response.model_dump())
    
    if response.is_emergency:
        return ImmediateStateChange(
            next_state="EMERGENCY",
            input="Emergency situation detected",
            keep_original_response=True
        )
    
    return (
        f"Assessment: {response.reasoning}\n"
        f"Recommended Action: {response.recommended_action}\n"
        "Since this isn't an immediate emergency, I'll need to gather some information about you. "
        "Please provide your name, age, gender, any existing medical conditions, "
        "current medications, and allergies."
    )

@fsm.define_state(
    state_key="EMERGENCY",
    prompt_template="""
    EMERGENCY PROTOCOL ACTIVATED
    Provide clear, urgent instructions while emergency services are contacted.
    Review the situation and provide immediate first-aid guidance if appropriate.
    """,
    response_model=DefaultResponse,
    transitions={"END": "After emergency instructions are provided"}
)
async def emergency_handler(
    fsm: LLMStateMachine,
    response: DefaultResponse,
    will_transition: bool
) -> str:
    emergency_data = fsm.get_context_data("emergency_assessment")
    return (
        "🚨 EMERGENCY SITUATION DETECTED 🚨\n"
        f"{response.content}\n\n"
        "Please call emergency services immediately (911 in the US).\n"
        "Stay on the line while help is dispatched."
    )

@fsm.define_state(
    state_key="GATHER_PATIENT_INFO",
    prompt_template="""
    Parse the user's information into structured patient data.
    Prompt for any missing critical information.
    Look for any red flags in the patient's history or medications.
    """,
    response_model=PatientInfo,
    transitions={
        "DRUG_INTERACTION_CHECK": "When all critical patient information is gathered",
        "EMERGENCY": "If any red flags are detected in patient history",
    }
)
async def gather_patient_info(
    fsm: LLMStateMachine,
    response: PatientInfo,
    will_transition: bool
) -> str:
    fsm.set_context_data("patient_info", response.model_dump())
    
    return (
        "Thank you. I've recorded your information. "
        "Now, please describe the symptoms you're experiencing, "
        "including when they started and how severe they are."
    )

@fsm.define_state(
    state_key="DRUG_INTERACTION_CHECK",
    prompt_template="""
    Analyze the patient's current medications for potential interactions.
    Consider both existing conditions and reported symptoms.
    Flag any concerning combinations or contraindications.
    """,
    response_model=DrugInteraction,
    transitions={
        "SYMPTOM_ASSESSMENT": "If no severe interactions are found",
        "EMERGENCY": "If dangerous drug interactions are detected",
    }
)
async def check_drug_interactions(
    fsm: LLMStateMachine,
    response: DrugInteraction,
    will_transition: bool
) -> str:
    fsm.set_context_data("drug_interactions", response.model_dump())
    
    if response.severity == Severity.CRITICAL:
        return ImmediateStateChange(
            next_state="EMERGENCY",
            input=f"Critical drug interaction detected: {response.description}"
        )
    
    interaction_msg = (
        f"Medication Analysis:\n"
        f"Severity: {response.severity}\n"
        f"Details: {response.description}\n"
        f"Recommendation: {response.recommendation}\n\n"
    )
    
    return interaction_msg + "Now, let's assess your symptoms in detail."

@fsm.define_state(
    state_key="SYMPTOM_ASSESSMENT",
    prompt_template="""
    Analyze the reported symptoms considering:
    - Patient's age, gender, and medical history
    - Symptom severity and duration
    - Potential interactions with existing conditions
    - Risk factors and warning signs
    
    Generate a structured assessment and identify any patterns or concerning combinations.
    """,
    response_model=SymptomAssessment,
    transitions={
        "GENERATE_PLAN": "If symptoms are well understood and non-emergency",
        "EMERGENCY": "If symptoms suggest a serious condition",
    }
)
async def assess_symptoms(
    fsm: LLMStateMachine,
    response: SymptomAssessment,
    will_transition: bool
) -> str:
    fsm.set_context_data("symptom_assessment", response.model_dump())
    
    # Check for high-severity symptoms
    if any(s.severity == Severity.CRITICAL for s in response.symptoms):
        return ImmediateStateChange(
            next_state="EMERGENCY",
            input="Critical symptoms detected"
        )
    
    assessment = "Symptom Assessment:\n"
    for symptom in response.symptoms:
        assessment += f"- {symptom.name} ({symptom.severity}): {symptom.description}\n"
    
    assessment += "\nPotential Causes:\n"
    for cause in response.potential_causes:
        assessment += f"- {cause}\n"
    
    if response.additional_questions:
        assessment += "\nI need some additional information:\n"
        for question in response.additional_questions:
            assessment += f"- {question}\n"
    
    return assessment

@fsm.define_state(
    state_key="GENERATE_PLAN",
    prompt_template="""
    Based on the complete assessment, generate a comprehensive care plan.
    Consider:
    - Patient's specific circumstances and limitations
    - Interaction with existing conditions and medications
    - Clear follow-up timeline and monitoring plans
    - Specific warning signs to watch for
    """,
    response_model=TreatmentPlan,
    transitions={
        "END": "After plan is generated and explained",
        "EMERGENCY": "If complications arise during plan generation",
    }
)
async def generate_treatment_plan(
    fsm: LLMStateMachine,
    response: TreatmentPlan,
    will_transition: bool
) -> str:
    fsm.set_context_data("treatment_plan", response.model_dump())
    
    plan = "Based on our assessment, here's your care plan:\n\n"
    
    plan += "Recommendations:\n"
    for rec in response.primary_recommendations:
        plan += f"- {rec}\n"
    
    plan += "\nLifestyle Modifications:\n"
    for mod in response.lifestyle_modifications:
        plan += f"- {mod}\n"
    
    plan += f"\nFollow-up Timeline: {response.follow_up_timeline}\n"
    
    plan += "\nWarning Signs (Seek immediate care if you experience):\n"
    for warning in response.warning_signs:
        plan += f"- {warning}\n"
    
    plan += "\nEmergency Conditions:\n"
    for condition in response.emergency_conditions:
        plan += f"- {condition}\n"
    
    # Generate audit log
    audit_log = {
        "timestamp": datetime.now().isoformat(),
        "patient_info": fsm.get_context_data("patient_info"),
        "emergency_assessment": fsm.get_context_data("emergency_assessment"),
        "drug_interactions": fsm.get_context_data("drug_interactions"),
        "symptom_assessment": fsm.get_context_data("symptom_assessment"),
        "treatment_plan": response.model_dump(),
    }
    fsm.set_context_data("audit_log", audit_log)
    
    return plan

@fsm.define_state(
    state_key="END",
    prompt_template="Provide final instructions and documentation.",
    response_model=DefaultResponse,
)
async def end_state(
    fsm: LLMStateMachine,
    response: DefaultResponse,
    will_transition: bool
) -> str:
    audit_log = fsm.get_context_data("audit_log")
    # Here you could save the audit log to a secure medical record system
    
    return (
        "Your assessment is complete. Please follow the provided care plan "
        "and don't hesitate to seek emergency care if warning signs develop. "
        "A record of this assessment has been saved for future reference."
    )

async def main():
    openai_client = openai.AsyncOpenAI()
    print("Medical Triage System Initialized")
    print("Please describe your medical concern:")
    
    while not fsm.is_completed():
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        run_state: FSMRun = await fsm.run_state_machine(
            openai_client,
            user_input=user_input,
            model="gpt-4o-mini"
        )
        print(f"System: {run_state.response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())