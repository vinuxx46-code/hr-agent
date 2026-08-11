import os
import json
import re
import PyPDF2
from google import genai

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


