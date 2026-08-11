import os
import json
from fastapi import FastAPI, UploadFile, File, Form
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

load_dotenv()

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
            print("Scanned PDF detected. Passing to Local OCR.")
            try:
                import fitz
                import pytesseract
                from PIL import Image

                pdf_document = fitz.open(stream=content, filetype="pdf")
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img)
                    resume_text += page_text + "\n"
                print("Local OCR Completed.")
            except Exception as ocr_e:
                print(f"Local OCR Failed: {ocr_e}")

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
            gemini_contents.append({
                "mime_type": mime_type,
                "data": content
            })

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=gemini_contents,
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
    results = []
    
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for filename in z.namelist():
            if filename.endswith(('.pdf', '.doc', '.docx')) and not filename.startswith('__MACOSX'):
                file_content = z.read(filename)
                # Process synchronously to avoid rate limits, or use asyncio.gather for parallel
                res = await process_single_resume(file_content, filename)
                results.append(res)
                
    return {"isBulk": True, "candidates": results}


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
            gemini_contents.append({
                "mime_type": mime_type,
                "data": content
            })

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=gemini_contents,
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
            gemini_contents.append({
                "mime_type": mime_type,
                "data": content
            })

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=gemini_contents,
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
        return {"error": "Error analyzing candidate. Please check the backend server."}


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

@app.post("/api/interview/start")
async def start_interview(request: InterviewStartRequest):
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


@app.post("/api/hr/send-invites")
async def send_hr_invites(data: dict):
    # Receives { "candidates": [{"email": "...", "filename": "...", ...}], "expiry_hours": 48 }
    db = load_hr_db()
    expiry_hours = data.get("expiry_hours", 48)
    
    invited = []
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
            "invited_at": datetime.now().isoformat()
        }
        
        # In a real app, we would send an SMTP email here.
        # For prototype, we just log it and the frontend provides a mailto link or shows success.
        print(f"[HR EMAIL MOCK] Sent interview link to {c['email']}: http://localhost:5173/?token={token}")
        
        # Real SMTP implementation
        smtp_server = os.getenv("SMTP_SERVER")
        if smtp_server:
            try:
                import smtplib
                from email.mime.text import MIMEText
                
                msg = MIMEText(f"Congratulations! You have been selected for an online interview. Please click here to begin: http://localhost:5173/?token={token}\n\nThis link will expire in {expiry_hours} hours.")
                msg['Subject'] = 'Invitation to AI Interview'
                msg['From'] = os.getenv("SMTP_USER", "noreply@company.com")
                msg['To'] = c['email']
                
                with smtplib.SMTP(smtp_server, int(os.getenv("SMTP_PORT", 587))) as server:
                    server.starttls()
                    if os.getenv("SMTP_PASSWORD"):
                        server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
                    server.send_message(msg)
                    print(f"[HR EMAIL] Successfully sent email to {c['email']}")
            except Exception as e:
                print(f"[HR EMAIL ERROR] Failed to send to {c['email']}: {e}")
                
        invited.append({"email": c["email"], "link": f"http://localhost:5173/?token={token}"})
        
    save_hr_db(db)
    return {"status": "success", "invited_count": len(invited), "invited": invited}

@app.get("/api/validate-token/{token}")
async def validate_token(token: str):
    db = load_hr_db()
    if token not in db["candidates"]:
        return {"valid": False, "reason": "Invalid token."}
        
    candidate = db["candidates"][token]
    expiry = datetime.fromisoformat(candidate["expiry"])
    if datetime.now() > expiry:
        candidate["status"] = "EXPIRED"
        save_hr_db(db)
        return {"valid": False, "reason": "This interview link has expired."}
        
    if candidate.get("status") == "COMPLETED":
        return {"valid": False, "reason": "Interview already completed."}
        
    return {"valid": True, "candidate": candidate}

@app.post("/api/upload-interview-data/{token}")
async def upload_interview_data(token: str, video: UploadFile = File(None), proctoring_logs: str = Form("{}")):
    db = load_hr_db()
    if token not in db["candidates"]:
        return {"error": "Invalid token"}
        
    candidate = db["candidates"][token]
    
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
            
            # Mock HR Final Email
            print(f"[HR EMAIL MOCK] Candidate {candidate.get('email')} completed interview. Results saved to Dashboard.")
            
            smtp_server = os.getenv("SMTP_SERVER")
            if smtp_server:
                try:
                    import smtplib
                    from email.mime.text import MIMEText
                    
                    msg = MIMEText(f"Candidate {candidate.get('email')} has completed their interview.\n\nView the Recruiter Dashboard to see the recording and evaluation report.")
                    msg['Subject'] = 'Interview Completed: ' + candidate.get('email', 'Candidate')
                    msg['From'] = os.getenv("SMTP_USER", "noreply@company.com")
                    msg['To'] = os.getenv("HR_EMAIL", "hr@company.com")
                    
                    with smtplib.SMTP(smtp_server, int(os.getenv("SMTP_PORT", 587))) as server:
                        server.starttls()
                        if os.getenv("SMTP_PASSWORD"):
                            server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
                        server.send_message(msg)
                except Exception as e:
                    pass
                    
    return evaluation_result
