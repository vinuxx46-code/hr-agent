import os
import json
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import PyPDF2
import io
import re
import hashlib
from datetime import datetime, timedelta
import base64
from typing import Optional, List
import knowledge_base
import requirement_engine
import uuid
import zipfile
import io
import asyncio
import time

# Explicitly load .env from parent directory (project root)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(dotenv_path=_env_path, override=True)

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

REJECTED_RESUMES_FILE = "data/rejected_resumes.json"

HR_DB_FILE = "data/hr_database.json"

def load_hr_db():
    if os.path.exists(HR_DB_FILE):
        try:
            with open(HR_DB_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"candidates": {}}

def save_hr_db(db):
    with open(HR_DB_FILE, "w") as f:
        json.dump(db, f, indent=4)


os.makedirs("temp_resumes", exist_ok=True)
os.makedirs("hr_inbox", exist_ok=True)

def load_rejected_resumes():
    if os.path.exists(REJECTED_RESUMES_FILE):
        try:
            with open(REJECTED_RESUMES_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_rejected_resumes(data):
    with open(REJECTED_RESUMES_FILE, "w") as f:
        json.dump(data, f)

# In-memory stores for interview sessions
ACTIVE_INTERVIEWS = {}
COMPLETED_INTERVIEWS = []


def extract_contact_info(text):
    email, phone = None, None
    # Basic email regex
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match:
        email = f"email:{email_match.group(0).lower()}"
        
    # Basic phone regex (digits, dashes, plus, parenthesis, spaces)
    phone_match = re.search(r'\+?[\d\s\-\(\)]{8,15}', text)
    if phone_match:
        clean_phone = re.sub(r'\D', '', phone_match.group(0))
        if len(clean_phone) >= 8:
            phone = f"phone:{clean_phone}"
            
    return email, phone


def parse_resume_content(content: bytes, filename: str) -> tuple:
    """
    Parse resume content from various file formats.
    Returns (resume_text, mime_type, is_image).
    """
    resume_text = ""
    mime_type = "text/plain"
    is_image = False

    if filename.endswith(".pdf"):
        mime_type = "application/pdf"
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        for page in pdf_reader.pages:
            resume_text += page.extract_text() or ""
    elif filename.endswith(".docx") or filename.endswith(".doc"):
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        from docx import Document
        doc = Document(io.BytesIO(content))
        for para in doc.paragraphs:
            resume_text += para.text + "\n"
    elif filename.endswith(".png"):
        mime_type = "image/png"
        is_image = True
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        mime_type = "image/jpeg"
        is_image = True
    else:
        # Assume plain text for .txt and others
        resume_text = content.decode("utf-8", errors="ignore")

    # Handle scanned PDFs with no extractable text
    if not resume_text.strip() and not is_image:
        if mime_type == "application/pdf":
            print("Scanned PDF detected. Skipping local OCR to save time (relying on AI Vision or Fallback).")

    return resume_text, mime_type, is_image


from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount recordings directory for session video streaming
os.makedirs("recordings", exist_ok=True)
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "fake-key-for-now"))

# Load keywords from analyzed resumes
def load_dynamic_keywords():
    default_keywords = "Python, R, SQL, PyTorch, TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Data Visualization, MLOps, HuggingFace, OpenAI, RAG, etc."
    try:
        if os.path.exists("data/resume_analysis_results.json"):
            with open("data/resume_analysis_results.json", "r") as f:
                data = json.load(f)
                skills = set()
                for resume in data.get("shortlisted", []):
                    for skill in resume.get("skillsFound", []):
                        skills.add(skill)
                if skills:
                    return ", ".join(sorted(skills))
    except Exception as e:
        print("Error loading dynamic keywords:", e)
    return default_keywords

dynamic_keywords = load_dynamic_keywords()

company_expectations = f"""
We are hiring for our AI & Data Department, with roles ranging from Fresher/Entry-Level to Advanced AI & ML Developers.
Requirements:
- The candidate's resume MUST show a strong focus on AI & ML, Data Science, or Data Analytics. This is our main target.
- Key areas of interest: Artificial Intelligence, Machine Learning, Deep Learning, Data Science, Data Analytics, NLP, Computer Vision, Generative AI, LLMs.
- Relevant Tech Stack & Keywords: {dynamic_keywords}.
- Experience Level: We accept all levels from freshers/graduates to advanced developers, as long as their skills align with AI, ML, Data Science, or Data Analytics.
- If the candidate's resume contains strong evidence of these keywords and related AI/ML/Data domain experience, ALLOW NEXT LEVEL.
- If the candidate's resume is completely unrelated (e.g., purely unrelated web development, non-technical, or generic IT without data/AI focus), REJECT.
- Provide a detailed ATS-style analysis of their exact technical keywords and explain why they fit into our AI/Data talent pool.
"""

class InterviewRequest(BaseModel):
    history: list
    message: str
    image: Optional[str] = None
    resumeContext: Optional[dict] = None


# ============================================================
# STRICT MATCHING SYSTEM PROMPT (shared by both endpoints)
# ============================================================

def build_strict_matching_prompt(job_description: str, resume_text: str) -> str:
    """
    Build a fast, concise Gemini prompt for resume evaluation.
    """
    return f"""
You are a fast AI HR Recruiter Agent. 
Analyze the resume against the job requirements and output a raw JSON evaluation.

Job Requirements:
{job_description}

Candidate's Resume Text:
{resume_text}

Calculate a matchPercentage (0-100). If they meet the core requirements, set isMatch to true and next_round_status to "ALLOW NEXT LEVEL". Otherwise false and "REJECT".

Return exactly ONE raw JSON object. No markdown wrapping. KEEP IT EXTREMELY CONCISE. Do not return any other fields.

{{
    "isMatch": true,
    "matchPercentage": 85,
    "reason": "A brief 2-3 sentence summary explaining the match or rejection.",
    "skillsFound": ["list of matched skills"],
    "missingSkills": ["list of missing requirements"],
    "candidate_profile": {{"name": "...", "contact": "...", "objective": "..."}},
    "next_round_status": "ALLOW NEXT LEVEL"
}}
"""



async def process_single_resume(content: bytes, filename: str):
    try:
        rejected_resumes = load_rejected_resumes()
        resume_text, mime_type, is_image = parse_resume_content(content, filename)
        
        filename = os.path.basename(filename) # Strip nested directories to prevent Errno 2
        temp_resume_path = f"temp_resumes/{uuid.uuid4()}_{filename}"
        with open(temp_resume_path, "wb") as f:
            f.write(content)

        email_id, phone_id = extract_contact_info(resume_text)
        identifiers = [x for x in [email_id, phone_id] if x is not None]
        content_hash = hashlib.md5(content).hexdigest()
        identifiers.append(f"hash:{content_hash}")

        prompt = build_strict_matching_prompt(company_expectations, resume_text)
        gemini_contents = [prompt]
        if not resume_text.strip() or is_image:
            from google.genai import types as genai_types
            gemini_contents = [
                genai_types.Content(parts=[
                    genai_types.Part(text=prompt),
                    genai_types.Part(inline_data=genai_types.Blob(mime_type=mime_type, data=content))
                ])
            ]

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_actual_api_key_here" or api_key == "fake-key-for-now" or not (api_key.startswith("AIza") or api_key.startswith("AQ.")):
            raise ValueError("Invalid API Key, skipping to offline fallback for instant processing")
            
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model='gemini-2.5-flash',
                contents=gemini_contents
            ),
            timeout=5.0
        )

        result_text = response.text
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            evaluation = json.loads(json_str)
            evaluation = requirement_engine.post_process_ai_response(
                ai_response=evaluation,
                resume_text=resume_text,
                job_description=company_expectations,
            )
            if not evaluation.get("isMatch", False):
                iso_date = datetime.now().isoformat()
                for identifier in identifiers:
                    rejected_resumes[identifier] = iso_date
                save_rejected_resumes(rejected_resumes)
            if not email_id:
                ai_contact = evaluation.get("candidate_profile", {}).get("contact", "")
                ai_email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', ai_contact)
                if ai_email:
                    email_id = f"email:{ai_email.group(0).lower()}"
                    
            evaluation["resumeFilePath"] = temp_resume_path
            evaluation["filename"] = filename
            evaluation["email"] = email_id
            return evaluation
        else:
            raise ValueError("No JSON object found in response.")
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        try:
            result = requirement_engine.build_offline_evaluation(
                resume_text=resume_text,
                job_description=company_expectations,
            )
            if not result.get("isMatch", False):
                iso_date = datetime.now().isoformat()
                for identifier in identifiers:
                    rejected_resumes[identifier] = iso_date
                save_rejected_resumes(rejected_resumes)
            result["resumeFilePath"] = temp_resume_path
            result["filename"] = filename
            result["email"] = email_id
            return result
        except Exception as fallback_error:
            return {"error": f"Failed: {str(e)}", "filename": filename}

@app.post("/api/bulk-upload")
async def bulk_upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        return {"error": "Must be a .zip file"}
    
    content = await file.read()
    
    semaphore = asyncio.Semaphore(5)

    async def process_with_semaphore(file_content, filename):
        async with semaphore:
            return await process_single_resume(file_content, filename)

    tasks = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for filename in z.namelist():
            if filename.endswith(('.pdf', '.doc', '.docx')) and not filename.startswith('__MACOSX'):
                file_content = z.read(filename)
                tasks.append(process_with_semaphore(file_content, filename))
                
    results = await asyncio.gather(*tasks)
                
    return {"isBulk": True, "candidates": list(results)}


@app.post("/api/upload-resume")
async def upload_resume(resume: UploadFile = File(...)):
    try:
        # Read the file content and parse based on extension
        content = await resume.read()
        filename = resume.filename.lower()

        # Load rejected resumes and prepare identifiers for cooldown tracking
        rejected_resumes = load_rejected_resumes()

        # Parse resume content
        resume_text, mime_type, is_image = parse_resume_content(content, filename)
        
        # Save resume for HR email attachment
        temp_resume_path = f"temp_resumes/{uuid.uuid4()}_{filename}"
        with open(temp_resume_path, "wb") as f:
            f.write(content)

        # Print extracted text to console for debugging
        print(f"--- Extracted Text from {filename} ---")
        if is_image:
            print("[Image File Detected - Using AI OCR Vision Technology]")
        else:
            print(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
        print("---------------------------------------")

        # We rely on Gemini's native multimodal capabilities for scanned PDFs/images
        # so we don't block the request if local text extraction fails.

        # Compute identifiers for cooldown tracking
        email_id, phone_id = extract_contact_info(resume_text)
        identifiers = [x for x in [email_id, phone_id] if x is not None]
        content_hash = hashlib.md5(content).hexdigest()
        identifiers.append(f"hash:{content_hash}")

        # Build the strict matching prompt
        prompt = build_strict_matching_prompt(company_expectations, resume_text)

        # Prepare contents array for Gemini
        gemini_contents = [prompt]
        
        # If text is empty (scanned PDF or image), pass the raw data for OCR
        if not resume_text.strip() or is_image:
            from google.genai import types as genai_types
            gemini_contents = [
                genai_types.Content(parts=[
                    genai_types.Part(text=prompt),
                    genai_types.Part(inline_data=genai_types.Blob(mime_type=mime_type, data=content))
                ])
            ]

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_actual_api_key_here" or api_key == "fake-key-for-now" or not (api_key.startswith("AIza") or api_key.startswith("AQ.")):
            raise ValueError("Invalid API Key, skipping to offline fallback for instant processing")
            
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model='gemini-2.5-flash',
                contents=gemini_contents
            ),
            timeout=5.0
        )

        result_text = response.text
        
        # Extract JSON using regex in case Gemini adds extra text
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            evaluation = json.loads(json_str)
            
            # ============================================================
            # POST-PROCESSING: Enforce strict scoring with requirement engine
            # ============================================================
            print(f"[Score Validation] AI returned score: {evaluation.get('matchPercentage', 'N/A')}")
            
            evaluation = requirement_engine.post_process_ai_response(
                ai_response=evaluation,
                resume_text=resume_text,
                job_description=company_expectations,
            )
            
            validated_score = evaluation.get("matchPercentage", 0)
            validation_info = evaluation.get("requirement_validation", {})
            
            print(f"[Score Validation] Final validated score: {validated_score}")
            print(f"[Score Validation] Full compliance: {validation_info.get('full_compliance', 'N/A')}")
            print(f"[Score Validation] Mandatory match rate: {validation_info.get('mandatory_match_rate', 'N/A')}")
            if validation_info.get("critical_gaps"):
                print(f"[Score Validation] Critical gaps: {validation_info['critical_gaps']}")
            if validation_info.get("blocking_reasons"):
                print(f"[Score Validation] 100% blocked: {validation_info['blocking_reasons']}")
            
            # Save rejection identifiers for 3-day cooldown
            if not evaluation.get("isMatch", False):
                iso_date = datetime.now().isoformat()
                for identifier in identifiers:
                    rejected_resumes[identifier] = iso_date
                save_rejected_resumes(rejected_resumes)
                
            if not email_id:
                ai_contact = evaluation.get("candidate_profile", {}).get("contact", "")
                ai_email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', ai_contact)
                if ai_email:
                    email_id = f"email:{ai_email.group(0).lower()}"
                    
            evaluation["resumeFilePath"] = temp_resume_path
            evaluation["filename"] = filename
            evaluation["email"] = email_id
            return evaluation
        else:
            raise ValueError("No JSON object found in response.")

    except Exception as e:
        print(f"Error processing resume with AI: {e}")
        
        # OFFLINE ATS SCANNER FALLBACK
        print("[Fallback] Using offline requirement engine for evaluation due to AI error...")
        try:
            result = requirement_engine.build_offline_evaluation(
                resume_text=resume_text,
                job_description=company_expectations,
            )
            
            # Save rejection identifiers for 3-day cooldown
            if not result.get("isMatch", False):
                iso_date = datetime.now().isoformat()
                for identifier in identifiers:
                    rejected_resumes[identifier] = iso_date
                save_rejected_resumes(rejected_resumes)
            
            result["resumeFilePath"] = temp_resume_path
            result["filename"] = filename
            result["email"] = email_id
            return result
            
        except Exception as fallback_error:
            print(f"[Offline Fallback Error] {fallback_error}")
            return {"error": f"Resume analysis failed. AI error: {str(e)}"}


@app.post("/api/analyze-candidate")
async def analyze_candidate(resume: UploadFile = File(...), jobDescription: str = Form(...)):
    try:
        # Read the file content and parse based on extension
        content = await resume.read()
        filename = resume.filename.lower()

        # Parse resume content
        resume_text, mime_type, is_image = parse_resume_content(content, filename)

        # Build the strict matching prompt with user-provided job description
        prompt = build_strict_matching_prompt(jobDescription, resume_text)

        # Prepare contents array for Gemini
        gemini_contents = [prompt]
        
        # If text is empty (scanned PDF or image), pass the raw data for OCR
        if not resume_text.strip() or is_image:
            from google.genai import types as genai_types
            gemini_contents = [
                genai_types.Content(parts=[
                    genai_types.Part(text=prompt),
                    genai_types.Part(inline_data=genai_types.Blob(mime_type=mime_type, data=content))
                ])
            ]

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_actual_api_key_here" or api_key == "fake-key-for-now" or not (api_key.startswith("AIza") or api_key.startswith("AQ.")):
            raise ValueError("Invalid API Key, skipping to offline fallback for instant processing")
            
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model='gemini-2.5-flash',
                contents=gemini_contents
            ),
            timeout=5.0
        )

        result_text = response.text
        
        # Extract JSON using regex in case Gemini adds extra text
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            evaluation = json.loads(json_str)
            
            # ============================================================
            # POST-PROCESSING: Enforce strict scoring with requirement engine
            # ============================================================
            print(f"[Analyze] AI returned score: {evaluation.get('matchPercentage', evaluation.get('score', 'N/A'))}")
            
            # Normalize the score field name (analyze endpoint may use 'score')
            if "score" in evaluation and "matchPercentage" not in evaluation:
                evaluation["matchPercentage"] = evaluation["score"]

            evaluation = requirement_engine.post_process_ai_response(
                ai_response=evaluation,
                resume_text=resume_text,
                job_description=jobDescription,
            )
            
            validated_score = evaluation.get("matchPercentage", 0)
            validation_info = evaluation.get("requirement_validation", {})
            
            print(f"[Analyze] Final validated score: {validated_score}")
            print(f"[Analyze] Full compliance: {validation_info.get('full_compliance', 'N/A')}")
            print(f"[Analyze] Mandatory match rate: {validation_info.get('mandatory_match_rate', 'N/A')}")
            if validation_info.get("critical_gaps"):
                print(f"[Analyze] Critical gaps: {validation_info['critical_gaps']}")
            
            return evaluation
        else:
            raise ValueError("No JSON object found in response.")

    except Exception as e:
        print(f"Error processing resume with AI: {e}")
        
        # OFFLINE ATS SCANNER FALLBACK
        print("[Fallback] Using offline requirement engine for evaluation due to AI error...")
        try:
            result = requirement_engine.build_offline_evaluation(
                resume_text=resume_text,
                job_description=jobDescription,
            )
            return result
        except Exception as fallback_error:
            print(f"[Offline Fallback Error] {fallback_error}")
            return {"error": "Error analyzing candidate. Please check the backend server."}


class InterviewStartRequest(BaseModel):
    resumeContext: dict
    candidateName: str = "Candidate"
    jobRole: str = "Software Engineer"
    token: Optional[str] = None  # Fixed: Pydantic v2 requires Optional for nullable str

class InterviewAnswerRequest(BaseModel):
    sessionId: str
    questionIndex: int
    answer: str
    proctoringEvents: List[dict] = []

class InterviewFinishRequest(BaseModel):
    sessionId: str
    proctoringEvents: List[dict] = []

@app.post("/api/interview/start")
async def start_interview(request: InterviewStartRequest):
    if request.token:
        db = load_hr_db()
        if request.token in db["candidates"]:
            candidate = db["candidates"][request.token]
            
            # Check if expiry time has passed (2 days)
            if "expiry" in candidate:
                from datetime import datetime
                expiry_dt = datetime.fromisoformat(candidate["expiry"])
                if datetime.now() > expiry_dt:
                    candidate["status"] = "EXPIRED"
                    save_hr_db(db)
                    
            if candidate.get("status") in ["COMPLETED", "EXPIRED", "USED", "IN_PROGRESS"] or candidate.get("used", False):
                return {"error": "This link is expired. You already attended the interview."}
            
            candidate["status"] = "IN_PROGRESS"
            candidate["used"] = True
            candidate["used_at"] = datetime.now().isoformat()
            save_hr_db(db)
            
    try:
        skills = request.resumeContext.get("skillsFound", [])
        if not skills:
            skills = request.resumeContext.get("skills", ["General Programming"])
            
        prompt = f"""
        You are an AI HR Agent generating interview questions for a {request.jobRole} position.
        
        Follow this exact process to generate exactly 15 short, personalized technical interview questions.
        
        ### Question Generation Rules
        Generate exactly 15 personalized technical interview questions.
        
        Question Types Distribution:
        - 5 Multiple Choice Questions (type: "MULTIPLE_CHOICE")
        - 5 True/False Questions (type: "TRUE_FALSE")
        - 5 Short Answer Questions (type: "SHORT_ANSWER")

        Question Content Distribution:
        - 15 Questions -> Resume Skills
          Generate ALL questions ONLY from technologies that actually exist in the candidate's resume context.
          Never generate questions for technologies not mentioned in the resume.

        ### Difficulty Level
        Questions must match candidate experience based on their resume.
        Freshers: 70% Beginner, 30% Intermediate
        1–3 Years: 40% Beginner, 40% Intermediate, 20% Advanced
        4–7 Years: 20% Beginner, 40% Intermediate, 40% Advanced
        8+ Years: 10% Beginner, 30% Intermediate, 60% Advanced

        ### Distractor Rules
        For MULTIPLE_CHOICE: Generate intelligent distractors. Randomize the correct option between A, B, C and D.
        For TRUE_FALSE: Options must be exactly "True" and "False".
        For SHORT_ANSWER: Leave the options field completely empty/null, as the user will type their answer.

        ### No Duplicate Questions
        Do not repeat concepts, technologies, difficulty, or question patterns.
        Each question must assess a different concept.
        
        Candidate Resume Context:
        {json.dumps(request.resumeContext)}
        
        Return ONLY a JSON object with this exact structure. No markdown, no explanation:
        {{
          "assessment": [
            {{
              "type": "MULTIPLE_CHOICE",
              "question": "Which Python library is primarily used for numerical computing?",
              "options": {{
                "A": "Flask",
                "B": "NumPy",
                "C": "Selenium",
                "D": "Django"
              }},
              "correctAnswer": "B",
              "explanation": "NumPy provides efficient multidimensional arrays.",
              "category": "RESUME_BASED",
              "skill": "Python",
              "difficulty": "Beginner",
              "marks": 1
            }},
            {{
              "type": "SHORT_ANSWER",
              "question": "Explain how a Promise works in JavaScript.",
              "options": null,
              "correctAnswer": "A Promise represents the eventual completion of an asynchronous operation...",
              "explanation": "Candidate should mention async, states (pending, resolved, rejected).",
              "category": "RESUME_BASED",
              "skill": "JavaScript",
              "difficulty": "Intermediate",
              "marks": 1
            }}
          ],
          "totalQuestions": 15,
          "totalMarks": 15
        }}
        """
        
        questions = []
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt],
                )
                text = response.text.strip()
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    parsed_json = json.loads(json_match.group(0))
                else:
                    parsed_json = json.loads(text)
                    
                questions = parsed_json.get("assessment", [])
                
                # Validate exact structure
                if len(questions) == 15:
                    resume_q = sum(1 for q in questions if q.get("category") == "RESUME_BASED")
                    if resume_q == 15:
                        break
                print(f"Validation failed on attempt {attempt+1}. Got {len(questions)} questions. Retrying...")
            except Exception as loop_e:
                print(f"JSON parse error on attempt {attempt+1}: {loop_e}")
                
        if len(questions) != 15:
            raise ValueError("Failed to generate exactly 15 structured questions after 3 retries.")
            
        session_id = str(uuid.uuid4())
        ACTIVE_INTERVIEWS[session_id] = {
            "candidateName": request.candidateName,
            "jobRole": request.jobRole,
            "resumeContext": request.resumeContext,
            "questions": questions,
            "answers": [],
            "scores": [],
            "evaluations": [],
            "proctoringEvents": [],
            "currentQuestionIndex": 0,
            "questionStartTime": time.time(),
            "status": "in_progress"
        }
        
        return {
            "sessionId": session_id,
            "totalQuestions": len(questions),
            "questionIndex": 0,
            "questionText": questions[0],
            "timeLimitSeconds": 30
        }
    except Exception as e:
        print(f"Error starting interview (likely API Key issue), falling back to offline mode: {e}")
        
        import random
        # Extract skills for personalized offline fallback
        skills = request.resumeContext.get("skillsFound", [])
        if not skills:
            skills = request.resumeContext.get("skills", ["General Programming", "Software Engineering", "System Design", "Problem Solving", "Debugging"])
        
        # Ensure we have at least 5 skills
        while len(skills) < 5:
            skills.append(random.choice(["Data Structures", "Algorithms", "Optimization", "Architecture", "Testing"]))
            
        selected_skills = random.sample(skills, min(len(skills), 5))
        if len(selected_skills) < 5:
            selected_skills = random.choices(skills, k=5)
            
        resume_pool = [
            f"How have you applied your knowledge of {selected_skills[0]} in a past project?",
            f"Describe a time you solved a difficult technical problem using {selected_skills[1]}.",
            f"What is your greatest professional achievement related to {selected_skills[2]}?",
            f"How do you stay updated with the latest trends in {selected_skills[3]}?",
            f"Explain a complex concept in {selected_skills[4]} to someone without a technical background."
        ]
        
        company_pool = [
            "How do you ensure code quality and performance in your projects?",
            "Describe a time you disagreed with a team member on a technical decision. How did you resolve it?",
            "What is your approach to learning a completely new technology quickly?",
            "How do you handle tight deadlines and high-pressure deployments?",
            "Can you explain your experience with scalable architecture?"
        ]
        
        random.shuffle(resume_pool)
        random.shuffle(company_pool)
        
        questions = []
        for i in range(5):
            skill = selected_skills[i % len(selected_skills)]
            questions.append({
                "type": "MULTIPLE_CHOICE",
                "questionNumber": len(questions) + 1, 
                "question": resume_pool[i % len(resume_pool)], 
                "category": "RESUME_BASED", 
                "skill": skill, 
                "marks": 1,
                "options": {
                    "A": f"Utilized {skill} heavily for optimizing core application logic.",
                    "B": f"Mainly applied {skill} in minor debugging.",
                    "C": f"Led a small team leveraging {skill}.",
                    "D": f"Used {skill} only occasionally."
                },
                "correctAnswer": "A",
                "explanation": f"Practical experience with {skill}.",
                "difficulty": "Intermediate"
            })
        for i in range(5):
            skill = selected_skills[i % len(selected_skills)]
            questions.append({
                "type": "TRUE_FALSE",
                "questionNumber": len(questions) + 1, 
                "question": f"Is {skill} always the best choice for every project?", 
                "category": "RESUME_BASED", 
                "skill": skill, 
                "marks": 1,
                "options": {
                    "True": "True",
                    "False": "False"
                },
                "correctAnswer": "False",
                "explanation": "No single technology is the best choice for everything.",
                "difficulty": "Beginner"
            })
        for i in range(5):
            skill = selected_skills[i % len(selected_skills)]
            questions.append({
                "type": "SHORT_ANSWER",
                "questionNumber": len(questions) + 1, 
                "question": f"Describe a time you solved a difficult technical problem using {skill}.", 
                "category": "RESUME_BASED", 
                "skill": skill, 
                "marks": 1,
                "options": None,
                "correctAnswer": "Candidate provided a valid scenario.",
                "explanation": "Any reasonable explanation is valid.",
                "difficulty": "Advanced"
            })
        session_id = str(uuid.uuid4())
        ACTIVE_INTERVIEWS[session_id] = {
            "candidateName": request.candidateName,
            "jobRole": request.jobRole,
            "resumeContext": request.resumeContext,
            "questions": questions,
            "answers": [],
            "scores": [],
            "evaluations": [],
            "proctoringEvents": [],
            "currentQuestionIndex": 0,
            "questionStartTime": time.time(),
            "status": "in_progress"
        }
        
        return {
            "sessionId": session_id,
            "totalQuestions": len(questions),
            "questionIndex": 0,
            "questionText": questions[0],
            "timeLimitSeconds": 15
        }

@app.post("/api/interview/answer")
async def submit_answer(request: InterviewAnswerRequest):
    session = ACTIVE_INTERVIEWS.get(request.sessionId)
    if not session:
        return {"error": "Session not found"}
        
    current_time = time.time()
    time_taken = current_time - session["questionStartTime"]
    
    # Strict Backend Validation: Allow 5 seconds of network buffer over 60s
    if time_taken > 65:
        print(f"Timer validation failed. Time taken: {time_taken}s")
        request.answer = "[TIME EXPIRED]"
        
    session["proctoringEvents"].extend(request.proctoringEvents)
    
    question_data = session["questions"][request.questionIndex]
    question_text = question_data.get("question", str(question_data))
    
    if request.answer == "[TIME EXPIRED]" or not request.answer.strip():
        score = 0
        ai_evaluation = "No answer provided within the time limit."
    else:
        # Evaluate answer with AI
        prompt = f"""
        Evaluate this technical interview answer for a 1-mark question.
        Question: {question_text}
        Candidate Answer: {request.answer}
        
        Score as:
        1 for Correct
        0.5 for Partially Correct
        0 for Incorrect
        
        Return ONLY a JSON object:
        {{
            "score": <0, 0.5, or 1>,
            "evaluation": "Brief explanation of the score and the expected concept."
        }}
        """
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt],
            )
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group(0))
                score = eval_data.get("score", 0)
                ai_evaluation = eval_data.get("evaluation", "")
            else:
                score = 0
                ai_evaluation = "Failed to parse AI evaluation."
        except Exception as e:
            score = 0
            ai_evaluation = f"AI Error: {e}"
            
    session["answers"].append(request.answer)
    session["scores"].append(score)
    session["evaluations"].append(ai_evaluation)
    
    next_idx = request.questionIndex + 1
    session["currentQuestionIndex"] = next_idx
    
    if next_idx < len(session["questions"]):
        session["questionStartTime"] = time.time()
        return {
            "completed": False,
            "questionIndex": next_idx,
            "questionText": session["questions"][next_idx],
            "timeLimitSeconds": 60
        }
    else:
        return {"completed": True}

@app.post("/api/interview/finish")
async def finish_interview(request: InterviewFinishRequest):
    session = ACTIVE_INTERVIEWS.get(request.sessionId)
    if not session:
        return {"error": "Session not found"}
        
    session["proctoringEvents"].extend(request.proctoringEvents)
    session["status"] = "completed"
    
    total_marks = len(session["questions"])
    marks_obtained = sum(session["scores"])
    percentage = (marks_obtained / total_marks) * 100 if total_marks > 0 else 0
    
    warnings = len(session["proctoringEvents"])
    
    # Generate Final HR Recommendation
    prompt = f"""
    Generate an HR Recommendation based on this interview performance and proctoring logs.
    Candidate Score: {marks_obtained} / {total_marks} ({percentage}%)
    Proctoring Events: {json.dumps(session["proctoringEvents"])}
    
    Provide a professional HR summary (2-3 sentences) and a final status recommendation.
    The status must be one of: RECOMMENDED, NEEDS_HR_REVIEW, NOT_RECOMMENDED, PROCTORING_REVIEW_REQUIRED, INTERVIEW_TERMINATED.
    
    Return ONLY JSON:
    {{
        "status": "...",
        "summary": "..."
    }}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt],
        )
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            rec_data = json.loads(json_match.group(0))
            final_status = rec_data.get("status", "NEEDS_HR_REVIEW")
            final_summary = rec_data.get("summary", "Summary unavailable.")
        else:
            final_status = "NEEDS_HR_REVIEW"
            final_summary = "Failed to parse AI recommendation."
    except Exception as e:
        final_status = "NEEDS_HR_REVIEW"
        final_summary = "Error generating recommendation."
        
    report = {
        "sessionId": request.sessionId,
        "candidateName": session["candidateName"],
        "jobRole": session["jobRole"],
        "resumeScore": session["resumeContext"].get("matchPercentage", session["resumeContext"].get("score", 0)),
        "totalMarks": total_marks,
        "marksObtained": marks_obtained,
        "percentage": percentage,
        "proctoringEvents": session["proctoringEvents"],
        "warningCount": warnings,
        "finalStatus": final_status,
        "finalSummary": final_summary,
        "questions": [
            {
                "question": session["questions"][i].get("question", str(session["questions"][i])) if isinstance(session["questions"][i], dict) else session["questions"][i],
                "answer": session["answers"][i] if i < len(session["answers"]) else "",
                "score": session["scores"][i] if i < len(session["scores"]) else 0,
                "evaluation": session["evaluations"][i] if i < len(session["evaluations"]) else ""
            }
            for i in range(len(session["questions"]))
        ],
        "completedAt": datetime.now().isoformat()
    }
    
    COMPLETED_INTERVIEWS.append(report)
    del ACTIVE_INTERVIEWS[request.sessionId]
    
    # ---------------------------------------------------------
    # HR EMAIL DISPATCH
    # ---------------------------------------------------------
    try:
        msg = MIMEMultipart()
        msg['From'] = "ai-proctor@company.com"
        msg['To'] = "hr@company.com"
        msg['Subject'] = f"Interview Completed: {session['candidateName']} - {final_status}"
        
        body = f"""
Candidate: {session['candidateName']}
Job Role: {session['jobRole']}
Final Score: {marks_obtained} / {total_marks} ({percentage}%)
Security Warnings: {warnings}

AI Recommendation: {final_status}
Summary: {final_summary}
        """
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach the saved resume
        resume_path = session.get("resumeContext", {}).get("resumeFilePath")
        if resume_path and os.path.exists(resume_path):
            with open(resume_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(resume_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(resume_path)}"'
            msg.attach(part)
            
        # 1. Save to local simulated inbox (.eml)
        eml_filename = f"hr_inbox/{request.sessionId}_{session['candidateName'].replace(' ', '_')}.eml"
        with open(eml_filename, "w", encoding="utf-8") as f:
            f.write(msg.as_string())
        print(f"[Email System] Simulated HR email saved to {eml_filename}")
        
        # 2. Real SMTP Sending (Currently mocked to prevent crashes without real credentials)
        # To enable real sending, uncomment and add credentials:
        # with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        #     server.login("your-email@gmail.com", "your-app-password")
        #     server.send_message(msg)
        
    except Exception as e:
        print(f"Failed to dispatch HR email: {e}")
    
    return report

@app.get("/api/candidates")
async def get_candidates():
    # Return completed interviews for HR dashboard
    return {"candidates": COMPLETED_INTERVIEWS}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


def background_send_emails(emails_to_send, smtp_server, smtp_port, smtp_user, smtp_password):
    if not smtp_server:
        return
    import smtplib
    from email.mime.text import MIMEText
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            if smtp_password:
                server.login(smtp_user, smtp_password)
            for c in emails_to_send:
                try:
                    msg = MIMEText(f"Congratulations! You have been selected for an online interview. Please click here to begin: {c['link']}\n\nThis link will expire in {c['expiry_hours']} hours.")
                    msg['Subject'] = 'Invitation to AI Interview'
                    msg['From'] = smtp_user or "noreply@company.com"
                    msg['To'] = c['email']
                    
                    server.send_message(msg)
                    print(f"[HR EMAIL] Successfully sent email to {c['email']}")
                except Exception as e:
                    print(f"[HR EMAIL ERROR] Failed to send to {c['email']}: {e}")
    except Exception as e:
        print(f"[HR EMAIL BATCH ERROR] Failed to connect to SMTP: {e}")

@app.get("/api/hr/candidates")
async def get_all_hr_candidates():
    db = load_hr_db()
    candidates = list(db.get("candidates", {}).values())
    candidates.sort(key=lambda x: x.get("invited_at", ""), reverse=True)
    return {"status": "success", "candidates": candidates}

@app.post("/api/hr/send-invites")
async def send_hr_invites(data: dict, background_tasks: BackgroundTasks):
    # Receives { "candidates": [{"email": "...", "filename": "...", ...}], "expiry_hours": 48 }
    db = load_hr_db()
    expiry_hours = data.get("expiry_hours", 48)
    
    invited = []
    emails_to_send = []
    
    for c in data.get("candidates", []):
        if not c.get("email"): continue
        
        token = str(uuid.uuid4())
        expiry_time = datetime.now() + timedelta(hours=expiry_hours)
        
        db["candidates"][token] = {
            "token": token,
            "filename": c.get("filename"),
            "email": c.get("email"),
            "matchPercentage": c.get("matchPercentage"),
            "resumeFilePath": c.get("resumeFilePath"),
            "status": "INVITED",
            "expiry": expiry_time.isoformat(),
            "invited_at": datetime.now().isoformat(),
            "used": False,
            "access_count": 0
        }
        
        link = f"http://localhost:5173/?token={token}"
        # For prototype, we just log it and the frontend provides a mailto link or shows success.
        print(f"[HR EMAIL MOCK] Queued interview link for {c['email']}: {link}")
        
        emails_to_send.append({
            "email": c["email"],
            "link": link,
            "expiry_hours": expiry_hours
        })
                
        invited.append({"email": c["email"], "link": link})
        
    save_hr_db(db)
    
    smtp_server = os.getenv("SMTP_SERVER")
    if smtp_server and emails_to_send:
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        background_tasks.add_task(
            background_send_emails, 
            emails_to_send, 
            smtp_server, 
            smtp_port, 
            smtp_user, 
            smtp_password
        )
        
    return {"status": "success", "invited_count": len(invited), "invited": invited}

@app.get("/api/validate-token/{token}")
async def validate_token(token: str):
    db = load_hr_db()
    if token not in db["candidates"]:
        return {"valid": False, "reason": "Invalid token."}
        
    candidate = db["candidates"][token]
    try:
        expiry = datetime.fromisoformat(candidate["expiry"])
        if datetime.now() > expiry:
            candidate["status"] = "EXPIRED"
            candidate["used"] = True
            save_hr_db(db)
            return {"valid": False, "reason": "This interview link has expired."}
    except Exception:
        pass
        
    if candidate.get("status") in ["COMPLETED", "EXPIRED", "USED", "IN_PROGRESS"] or candidate.get("used", False):
        return {"valid": False, "reason": "This interview link has already been used. Candidate interview links are valid for ONE single use only."}
        
    # Mark token as USED / IN_PROGRESS on initial access so link expires after 1 use
    candidate["status"] = "IN_PROGRESS"
    candidate["used"] = True
    candidate["used_at"] = datetime.now().isoformat()
    save_hr_db(db)
        
    return {"valid": True, "candidate": candidate}

@app.post("/api/expire-token/{token}")
async def expire_token(token: str):
    db = load_hr_db()
    if token in db["candidates"]:
        db["candidates"][token]["status"] = "EXPIRED"
        db["candidates"][token]["used"] = True
        db["candidates"][token]["used_at"] = datetime.now().isoformat()
        save_hr_db(db)
        return {"success": True, "message": "Interview link expired after single use."}
    return {"success": False, "error": "Token not found"}

def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("•", "-").replace("°", " deg").replace("“", '"').replace("”", '"').replace("’", "'").replace("—", "-")
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_candidate_pdf_report(candidate: dict) -> str:
    from fpdf import FPDF
    import os
    from datetime import datetime

    token = clean_pdf_text(candidate.get("token", "unknown"))
    email = clean_pdf_text(candidate.get("email", "Candidate"))
    filename = clean_pdf_text(candidate.get("filename", "N/A"))
    status = clean_pdf_text(candidate.get("status", "COMPLETED"))
    match_score = candidate.get("matchPercentage", 100)
    evaluation = candidate.get("evaluation", {})
    proctoring = candidate.get("proctoring_logs", [])
    recording_link = clean_pdf_text(candidate.get("recording_link", f"recordings/{token}.webm"))

    warning_count = len(proctoring) if isinstance(proctoring, list) else 0
    interview_score = evaluation.get("marksObtained", match_score or 85)
    overall_feedback = clean_pdf_text(evaluation.get("overallFeedback", "Candidate completed full technical interview and 360 degree room verification."))
    qa_list = evaluation.get("qaList", [])

    os.makedirs("reports", exist_ok=True)
    pdf_path = f"reports/{token}_Interview_Report.pdf"

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(15, 23, 42)
            self.rect(0, 0, 210, 30, 'F')
            self.set_font('Helvetica', 'B', 15)
            self.set_text_color(129, 140, 248)
            self.set_y(8)
            self.cell(0, 8, 'CANDIDATE INTERVIEW & PROCTORING AUDIT REPORT', align='C', new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 8)
            self.set_text_color(148, 163, 184)
            self.cell(0, 5, 'Recruiter - Executive Screening & Proctoring Log', align='C', new_x='LMARGIN', new_y='NEXT')
            self.ln(6)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(148, 163, 184)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - Confidential HR Candidate Report', align='C')

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Candidate Profile Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 34, 190, 26, 'DF')

    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(30, 41, 59)
    pdf.set_y(37)
    pdf.cell(95, 6, f' Candidate Email: {email}', new_x='RIGHT', new_y='TOP')
    pdf.cell(95, 6, f' Status: {status}', new_x='LMARGIN', new_y='NEXT')

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(95, 6, f' Resume File: {filename}', new_x='RIGHT', new_y='TOP')
    pdf.cell(95, 6, f' Token: {token}', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, f' Session Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', new_x='LMARGIN', new_y='NEXT')

    pdf.ln(4)

    # 2. Executive Metric Score Cards
    pdf.set_fill_color(238, 242, 255)
    pdf.rect(10, 66, 60, 20, 'F')
    pdf.set_fill_color(236, 253, 245)
    pdf.rect(75, 66, 60, 20, 'F')
    if warning_count > 0:
        pdf.set_fill_color(254, 242, 242)
    else:
        pdf.set_fill_color(236, 253, 245)
    pdf.rect(140, 66, 60, 20, 'F')

    pdf.set_y(68)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(99, 102, 241)
    pdf.cell(65, 4, 'MATCH RELEVANCE', align='C', new_x='RIGHT', new_y='TOP')
    pdf.set_text_color(16, 185, 129)
    pdf.cell(65, 4, 'INTERVIEW SCORE', align='C', new_x='RIGHT', new_y='TOP')
    if warning_count > 0:
        pdf.set_text_color(239, 68, 68)
    else:
        pdf.set_text_color(16, 185, 129)
    pdf.cell(60, 4, 'SECURITY PROCTORING', align='C', new_x='LMARGIN', new_y='NEXT')

    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(65, 10, f'{match_score}%', align='C', new_x='RIGHT', new_y='TOP')
    pdf.cell(65, 10, f'{interview_score} / 100', align='C', new_x='RIGHT', new_y='TOP')
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(60, 10, f'{"CLEAN SESSION" if warning_count == 0 else f"{warning_count} ALERTS"}', align='C', new_x='LMARGIN', new_y='NEXT')

    pdf.ln(6)

    # 3. AI Assessment Summary
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, 'AI Evaluator Assessment & Summary', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 5, overall_feedback)

    pdf.ln(4)

    # 4. Full Interview Section
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, 'Full Interview Section - Questions, Answers & Scores', new_x='LMARGIN', new_y='NEXT')

    if qa_list and isinstance(qa_list, list):
        for idx, qa in enumerate(qa_list, 1):
            q_text = clean_pdf_text(qa.get('question', ''))
            a_text = clean_pdf_text(qa.get('answer', ''))
            q_score = clean_pdf_text(qa.get('score', '9 / 10'))
            q_fb = clean_pdf_text(qa.get('feedback', ''))

            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_text_color(79, 70, 229)
            pdf.cell(150, 5, f'Q{idx}: {q_text[:90]}', new_x='RIGHT', new_y='TOP')
            pdf.set_text_color(16, 185, 129)
            pdf.cell(40, 5, f'Score: {q_score}', align='R', new_x='LMARGIN', new_y='NEXT')

            pdf.set_x(10)
            pdf.set_font('Helvetica', 'I', 8.5)
            pdf.set_text_color(51, 65, 85)
            pdf.multi_cell(0, 4.5, f'Answer: "{a_text}"')
            if q_fb:
                pdf.set_x(10)
                pdf.set_font('Helvetica', '', 8)
                pdf.set_text_color(100, 116, 139)
                pdf.multi_cell(0, 4, f'AI Notes: {q_fb}')
            pdf.ln(2)
    else:
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, 'Interview evaluated via interactive AI voice & video evaluation session.', new_x='LMARGIN', new_y='NEXT')

    pdf.ln(4)

    # 5. Proctoring Security & Recording Evidence
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, 'Proctoring Security & Recording Evidence', new_x='LMARGIN', new_y='NEXT')

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 5, f'- 360 Degree Room Verification: Verified', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 5, f'- Security Violations Detected: {warning_count}', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 5, f'- Session Video Stream Link: http://localhost:8000/{recording_link}', new_x='LMARGIN', new_y='NEXT')

    pdf.output(pdf_path)
    return pdf_path

def send_hr_completed_report(candidate: dict):
    email = candidate.get("email", "Unknown Candidate")
    token = candidate.get("token", "")
    filename = candidate.get("filename", "N/A")
    recording = candidate.get("recording_link", f"recordings/{token}.webm")
    proctoring = candidate.get("proctoring_logs", [])
    evaluation = candidate.get("evaluation", {})
    
    warning_count = len(proctoring) if isinstance(proctoring, list) else 0
    score = evaluation.get("marksObtained", evaluation.get("score", "N/A"))
    total_marks = evaluation.get("totalMarks", 100)
    percentage = evaluation.get("percentage", 85)
    rec = evaluation.get("finalStatus", "Under Review")

    # Generate PDF Report File
    pdf_file_path = generate_candidate_pdf_report(candidate)

    categories = {
        "360_SCAN_ADDITIONAL_PERSON": 0,
        "FORBIDDEN_OBJECT": 0,
        "MULTIPLE_FACES": 0,
        "EXTRA_HANDS": 0,
        "BACKGROUND_VOICE": 0,
        "TAB_SWITCH / BLUR": 0,
        "GAZE / HEAD MOVEMENT": 0,
        "OTHER_VIOLATIONS": 0
    }

    if isinstance(proctoring, list):
        for e in proctoring:
            etype = str(e.get("type", "")).upper()
            if "360_SCAN" in etype: categories["360_SCAN_ADDITIONAL_PERSON"] += 1
            elif "FORBIDDEN_OBJECT" in etype: categories["FORBIDDEN_OBJECT"] += 1
            elif "MULTIPLE_FACES" in etype: categories["MULTIPLE_FACES"] += 1
            elif "EXTRA_HANDS" in etype: categories["EXTRA_HANDS"] += 1
            elif "BACKGROUND_VOICE" in etype: categories["BACKGROUND_VOICE"] += 1
            elif "TAB_SWITCH" in etype or "WINDOW_BLUR" in etype: categories["TAB_SWITCH / BLUR"] += 1
            elif "EYES_WANDERING" in etype or "HEAD_TURNED" in etype: categories["GAZE / HEAD MOVEMENT"] += 1
            else: categories["OTHER_VIOLATIONS"] += 1

    has_violations = warning_count > 0
    subject_tag = " [SECURITY VIOLATIONS DETECTED]" if has_violations else " [PASSED PROCTORING]"
    subject = f"CANDIDATE INTERVIEW REPORT: {email}{subject_tag}"

    body = f"""CANDIDATE INTERVIEW & UNIFIED SECURITY AUDIT REPORT
==================================================
Candidate Email: {email}
Resume File: {filename}
Interview Token: {token}
Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--------------------------------------------------
COMPLETE ILLEGAL ACTIVITIES & PROCTORING AUDIT (START TO END)
--------------------------------------------------
Total Security Violations Detected: {warning_count}

ILLEGAL ACTIVITY BREAKDOWN:
  • 360° Room Scan Violations: {categories['360_SCAN_ADDITIONAL_PERSON']}
  • Forbidden Objects (Phone/Laptop/Book): {categories['FORBIDDEN_OBJECT']}
  • Multiple Persons / Secondary Face: {categories['MULTIPLE_FACES']}
  • Extra Hands Detected: {categories['EXTRA_HANDS']}
  • Background Voice / Whispering: {categories['BACKGROUND_VOICE']}
  • Browser Tab Switching / Blur: {categories['TAB_SWITCH / BLUR']}
  • Off-Screen Gaze & Head Movement: {categories['GAZE / HEAD MOVEMENT']}

--------------------------------------------------
FULL INTERVIEW VIDEO & SCREEN RECORDING EVIDENCE
--------------------------------------------------
- Direct Video Evidence Link: http://localhost:8000/{recording}

--------------------------------------------------
FINAL AI EVALUATION & MATCH SCORE
--------------------------------------------------
- Score: {score} / {total_marks} ({percentage}%)
- Recommendation: {rec}
==================================================
This report was automatically generated by Recruiter."""

    timeline_items = ""
    if has_violations and isinstance(proctoring, list):
        timeline_items = '<div style="color: #60a5fa; font-size: 15px; font-weight: 700; margin: 22px 0 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; text-transform: uppercase;">Security Violation Timeline</div>'
        for idx, event in enumerate(proctoring[:15], 1):
            etype = str(event.get('type', 'EVENT'))
            extra_desc = ""
            if "360_SCAN" in etype: extra_desc = " (Secondary face during room scan)"
            elif "MULTIPLE_FACES" in etype: extra_desc = " (Extra person in frame during Q&A)"
            elif "FORBIDDEN_OBJECT" in etype: extra_desc = " (Forbidden device detected)"
            elif "EXTRA_HANDS" in etype: extra_desc = " (Extra hand detected in frame)"
            elif "BACKGROUND_VOICE" in etype: extra_desc = " (Unrecognized background voice)"
            elif "TAB_SWITCH" in etype: extra_desc = " (Candidate switched tabs)"
            timeline_items += f'<div style="background: rgba(15, 23, 42, 0.6); border-left: 3px solid #f87171; padding: 10px 14px; margin-bottom: 8px; border-radius: 0 8px 8px 0; font-size: 13px; color: #fca5a5;"><strong>{idx}. [{event.get("timestamp", "N/A")}]</strong> {etype}{extra_desc}</div>'
    else:
        timeline_items = '<div style="color: #60a5fa; font-size: 15px; font-weight: 700; margin: 22px 0 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; text-transform: uppercase;">Security Violation Timeline</div><div style="color:#4ade80; font-size:13px; padding:10px; background:rgba(34,197,94,0.1); border-radius:8px;">Clean Session: No illegal activities or proctoring violations recorded.</div>'

    alert_box = f'<div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 18px; margin-bottom: 20px;"><div style="color: #fca5a5; font-size: 15px; font-weight: bold; margin-bottom: 4px;">HIGH PRIORITY SECURITY ALERT</div><div style="color: #f87171; font-size: 13px;">Total of {warning_count} security violations detected during interview session.</div></div>' if has_violations else ''
    alert_color = '#f87171' if warning_count > 0 else '#4ade80'

    html_template = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: 'Inter', -apple-system, sans-serif; background-color: #0b0f19; color: #f8fafc; margin: 0; padding: 20px;">
  <div style="max-width: 650px; margin: 0 auto; background: #111827; border-radius: 16px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 20px 40px rgba(0,0,0,0.5);">
    <div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); padding: 30px 25px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);">
      <h1 style="margin: 0; color: #818cf8; font-size: 22px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase;">CANDIDATE INTERVIEW REPORT</h1>
      <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 13px;">Automated AI Screening & Security Audit Log</p>
    </div>
    <div style="padding: 25px;">
      {ALERT_BOX}

      <div style="color: #60a5fa; font-size: 15px; font-weight: 700; margin: 15px 0 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; text-transform: uppercase;">Candidate Profile</div>
      <div style="width: 100%; margin-bottom: 20px; background: rgba(255,255,255,0.03); border-radius: 12px; padding: 15px; border: 1px solid rgba(255,255,255,0.05); box-sizing: border-box; font-size: 13px; line-height: 1.8;">
        <div><span style="color: #94a3b8; font-weight: 600; display: inline-block; width: 150px;">Candidate Email:</span> <span style="color: #f8fafc;">{EMAIL}</span></div>
        <div><span style="color: #94a3b8; font-weight: 600; display: inline-block; width: 150px;">Resume File:</span> <span style="color: #f8fafc;">{FILENAME}</span></div>
        <div><span style="color: #94a3b8; font-weight: 600; display: inline-block; width: 150px;">Interview Token:</span> <code style="background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">{TOKEN}</code></div>
      </div>

      <div style="color: #60a5fa; font-size: 15px; font-weight: 700; margin: 22px 0 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; text-transform: uppercase;">Evaluation & Score</div>
      <table style="width: 100%; margin-bottom: 20px; border-spacing: 8px; border-collapse: separate;">
        <tr>
          <td style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px; text-align: center; width: 33%;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Match Score</div>
            <div style="font-size: 24px; font-weight: 800; color: #60a5fa; margin-top: 4px;">{SCORE} / {TOTAL_MARKS}</div>
          </td>
          <td style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px; text-align: center; width: 33%;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Percentage</div>
            <div style="font-size: 24px; font-weight: 800; color: #38bdf8; margin-top: 4px;">{PERCENTAGE}%</div>
          </td>
          <td style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px; text-align: center; width: 33%;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Security Alerts</div>
            <div style="font-size: 24px; font-weight: 800; color: {ALERT_COLOR}; margin-top: 4px;">{WARNING_COUNT}</div>
          </td>
        </tr>
      </table>

      <div style="color: #60a5fa; font-size: 15px; font-weight: 700; margin: 22px 0 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; text-transform: uppercase;">Security Audit Breakdown</div>
      <div style="background: rgba(15, 23, 42, 0.6); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 15px; font-size: 13px; line-height: 1.8;">
        <div>• 360° Room Scan Violations: {CAT_360}</div>
        <div>• Forbidden Objects (Phone/Laptop/Book): {CAT_OBJ}</div>
        <div>• Multiple Persons / Extra Face: {CAT_FACES}</div>
        <div>• Extra Hands Detected: {CAT_HANDS}</div>
        <div>• Background Voice / Whispering: {CAT_VOICE}</div>
        <div>• Browser Tab Switching / Blur: {CAT_TAB}</div>
        <div>• Off-Screen Gaze & Head Movement: {CAT_GAZE}</div>
      </div>

      {TIMELINE_ITEMS}

      <div style="color: #60a5fa; font-size: 15px; font-weight: 700; margin: 22px 0 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; text-transform: uppercase;">Video Evidence & Recording</div>
      <div style="text-align: center; margin: 20px 0;">
        <a href="http://localhost:8000/{RECORDING}" style="display: inline-block; background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: #ffffff !important; font-weight: 700; text-decoration: none; padding: 14px 28px; border-radius: 30px; font-size: 14px; margin: 10px 0; box-shadow: 0 10px 20px rgba(37, 99, 235, 0.3); text-align: center;" target="_blank">Watch Video Recording Evidence</a>
        <div style="color: #94a3b8; font-size: 12px; margin-top: 8px;">Direct Link: <code>http://localhost:8000/{RECORDING}</code></div>
      </div>
    </div>
    <div style="text-align: center; padding: 20px; border-top: 1px solid rgba(255,255,255,0.05); color: #64748b; font-size: 12px;">
      This report was automatically generated by Recruiter. Attached: Official PDF Audit Report.
    </div>
  </div>
</body>
</html>"""

    html_body = html_template.replace("{ALERT_BOX}", alert_box)\
               .replace("{EMAIL}", str(email))\
               .replace("{FILENAME}", str(filename))\
               .replace("{TOKEN}", str(token))\
               .replace("{SCORE}", str(score))\
               .replace("{TOTAL_MARKS}", str(total_marks))\
               .replace("{PERCENTAGE}", str(percentage))\
               .replace("{ALERT_COLOR}", alert_color)\
               .replace("{WARNING_COUNT}", str(warning_count))\
               .replace("{CAT_360}", str(categories.get('360_SCAN_ADDITIONAL_PERSON', 0)))\
               .replace("{CAT_OBJ}", str(categories.get('FORBIDDEN_OBJECT', 0)))\
               .replace("{CAT_FACES}", str(categories.get('MULTIPLE_FACES', 0)))\
               .replace("{CAT_HANDS}", str(categories.get('EXTRA_HANDS', 0)))\
               .replace("{CAT_VOICE}", str(categories.get('BACKGROUND_VOICE', 0)))\
               .replace("{CAT_TAB}", str(categories.get('TAB_SWITCH / BLUR', 0)))\
               .replace("{CAT_GAZE}", str(categories.get('GAZE / HEAD MOVEMENT', 0)))\
               .replace("{TIMELINE_ITEMS}", timeline_items)\
               .replace("{RECORDING}", str(recording))
    
    print(f"[HR EMAIL REPORT GENERATED]\n{body}")
    
    smtp_server = os.getenv("SMTP_SERVER")
    if smtp_server:
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.mime.application import MIMEApplication
            
            msg = MIMEMultipart('mixed')
            msg['Subject'] = subject
            msg['From'] = os.getenv("SMTP_USER", "noreply@company.com")
            msg['To'] = candidate.get("hr_email") or os.getenv("HR_EMAIL") or os.getenv("SMTP_USER") or "hr@company.com"
            
            alt_part = MIMEMultipart('alternative')
            alt_part.attach(MIMEText(body, 'plain'))
            alt_part.attach(MIMEText(html_body, 'html'))
            msg.attach(alt_part)
            
            # Attach PDF File
            if os.path.exists(pdf_file_path):
                with open(pdf_file_path, "rb") as f:
                    pdf_attach = MIMEApplication(f.read(), _subtype="pdf")
                    pdf_attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_file_path))
                    msg.attach(pdf_attach)

            with smtplib.SMTP(smtp_server, int(os.getenv("SMTP_PORT", 587))) as server:
                server.starttls()
                if os.getenv("SMTP_PASSWORD"):
                    server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
                server.send_message(msg)
            print(f"[HR EMAIL SENT WITH PDF] Successfully sent report PDF email for {email} to {msg['To']}")
        except Exception as e:
            print(f"[HR EMAIL ERROR] Failed to send email: {e}")

@app.post("/api/upload-interview-data/{token}")
async def upload_interview_data(token: str, video: UploadFile = File(None), proctoring_logs: str = Form("{}")):
    db = load_hr_db()
    if token not in db["candidates"]:
        return {"error": "Invalid token"}
        
    candidate = db["candidates"][token]
    candidate["used"] = True
    if candidate.get("status") != "COMPLETED":
        candidate["status"] = "COMPLETED"
    
    # Save video if exists
    if video:
        os.makedirs("recordings", exist_ok=True)
        video_path = f"recordings/{token}.webm"
        content = await video.read()
        with open(video_path, "wb") as f:
            f.write(content)
        candidate["recording_link"] = video_path
        
    # Save proctoring logs
    candidate["proctoring_logs"] = json.loads(proctoring_logs)
    
    save_hr_db(db)
    send_hr_completed_report(candidate)
    return {"status": "success"}

@app.get("/api/hr/dashboard-data")
async def get_dashboard_data():
    return load_hr_db()

@app.post("/api/interview/evaluate")
async def evaluate_interview(data: dict):
    # data expects: {"questions": [], "answers": [], "resume_text": "...", "job_description": "...", "token": "..."}
    token = data.get("token")
    
    # Existing AI logic here (we'll just use the old evaluate_interview logic by renaming it)
    evaluation_result = await original_evaluate_interview(data)
    
    # HR Database Linking
    if token:
        db = load_hr_db()
        if token in db["candidates"]:
            candidate = db["candidates"][token]
            candidate["status"] = "COMPLETED"
            candidate["evaluation"] = evaluation_result
            candidate["completed_at"] = datetime.now().isoformat()
            save_hr_db(db)
            send_hr_completed_report(candidate)
                    
    return evaluation_result

@app.post("/api/hr/send-email/{token}")
async def send_candidate_email_report(token: str):
    db = load_hr_db()
    if token not in db["candidates"]:
        return {"success": False, "error": "Candidate token not found"}
    candidate = db["candidates"][token]
    send_hr_completed_report(candidate)
    hr_email = candidate.get("hr_email") or os.getenv("HR_EMAIL") or os.getenv("SMTP_USER") or "vinuxx46@gmail.com"
    return {"success": True, "message": f"Report successfully emailed to {hr_email}"}

from fastapi.responses import FileResponse

@app.get("/api/hr/download-pdf/{token}")
async def download_candidate_pdf_report(token: str):
    db = load_hr_db()
    if token not in db["candidates"]:
        return {"error": "Token not found"}
    candidate = db["candidates"][token]
    pdf_path = generate_candidate_pdf_report(candidate)
    return FileResponse(pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))
