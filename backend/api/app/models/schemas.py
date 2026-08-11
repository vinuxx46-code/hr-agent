from pydantic import BaseModel
from typing import Optional, List

class InterviewRequest(BaseModel):
    history: list
    message: str
    image: Optional[str] = None
    resumeContext: Optional[dict] = None


class InterviewStartRequest(BaseModel):
    resumeContext: dict
    candidateName: str = "Candidate"
    jobRole: str = "Software Engineer"


class InterviewAnswerRequest(BaseModel):
    sessionId: str
    questionIndex: int
    answer: str
    proctoringEvents: List[dict] = []


class InterviewFinishRequest(BaseModel):
    sessionId: str
    proctoringEvents: List[dict] = []


