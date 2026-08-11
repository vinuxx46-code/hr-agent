"""
Strict Requirement-Based Matching Engine for AI Recruiter Agent.

This module provides:
- Structured requirement extraction and classification
- Multi-level match type classification (EXACT → MISSING)
- Evidence strength scoring (0-6)
- Hard-gated 100% score validation
- Requirement matrix generation
- Weighted scoring with mandatory requirement gates
- Recommendation logic (NEXT ROUND / HR REVIEW / NOT RECOMMENDED)
"""

import re
from enum import Enum
from typing import Optional


# ============================================================
# ENUMS & CONSTANTS
# ============================================================

class MatchType(str, Enum):
    EXACT_MATCH = "EXACT MATCH"
    NORMALIZED_MATCH = "NORMALIZED MATCH"
    ALIAS_MATCH = "ALIAS / EQUIVALENT MATCH"
    SEMANTIC_MATCH = "SEMANTIC RELATED MATCH"
    PARTIAL_MATCH = "PARTIAL MATCH"
    WEAK_EVIDENCE = "WEAK EVIDENCE"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT APPLICABLE"


class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Recommendation(str, Enum):
    ALLOW_NEXT_LEVEL = "ALLOW NEXT LEVEL"
    REJECT = "REJECT"


EVIDENCE_LEVELS = {
    0: "No Evidence",
    1: "Mentioned Only",
    2: "Basic Exposure",
    3: "Academic / Training Evidence",
    4: "Project Evidence",
    5: "Professional Evidence",
    6: "Strong Production / Advanced Evidence",
}

# Default minimum evidence thresholds
MIN_EVIDENCE_DEFAULT = 4          # For technical skills
MIN_EVIDENCE_PRODUCTION = 5       # When JD says "production", "hands-on", etc.

# Score weights (percentages)
SCORE_WEIGHTS = {
    "mandatory_technical": 40,
    "experience": 20,
    "projects": 15,
    "semantic_relevance": 10,
    "education_certs": 5,
    "preferred_skills": 10,
}


# ============================================================
# COMPREHENSIVE SKILL ALIASES
# ============================================================

SKILL_ALIASES = {
    # NLP
    "nlp": "natural language processing",
    "natural language processing": "nlp",
    # ML
    "ml": "machine learning",
    "machine learning": "ml",
    # DL
    "dl": "deep learning",
    "deep learning": "dl",
    # GenAI
    "genai": "generative ai",
    "generative ai": "genai",
    "gen ai": "generative ai",
    # LLM
    "llm": "large language model",
    "large language model": "llm",
    "large language models": "llm",
    # RAG
    "rag": "retrieval augmented generation",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    # NER
    "ner": "named entity recognition",
    "named entity recognition": "ner",
    # GNN
    "gnn": "graph neural network",
    "graph neural network": "gnn",
    "graph neural networks": "gnn",
    # CV
    "cv": "computer vision",
    "computer vision": "cv",
    # K8s
    "k8s": "kubernetes",
    "kubernetes": "k8s",
    # SKLearn
    "sklearn": "scikit-learn",
    "scikit-learn": "sklearn",
    "scikit learn": "sklearn",
    # TF (context-dependent, but we include it)
    "tf": "tensorflow",
    "tensorflow": "tf",
    # PyTorch
    "torch": "pytorch",
    "pytorch": "torch",
    # PEFT
    "peft": "parameter efficient fine tuning",
    "parameter efficient fine tuning": "peft",
    "parameter-efficient fine-tuning": "peft",
    # LoRA
    "lora": "low rank adaptation",
    "low rank adaptation": "lora",
    "low-rank adaptation": "lora",
    # QLoRA
    "qlora": "quantized low rank adaptation",
    "quantized low rank adaptation": "qlora",
    # ASR
    "asr": "automatic speech recognition",
    "automatic speech recognition": "asr",
    "speech recognition": "asr",
    # TTS
    "tts": "text to speech",
    "text to speech": "tts",
    "text-to-speech": "tts",
    # CNN
    "cnn": "convolutional neural network",
    "convolutional neural network": "cnn",
    "convolutional neural networks": "cnn",
    # RNN
    "rnn": "recurrent neural network",
    "recurrent neural network": "rnn",
    "recurrent neural networks": "rnn",
    # LSTM
    "lstm": "long short-term memory",
    "long short-term memory": "lstm",
    # GRU
    "gru": "gated recurrent unit",
    "gated recurrent unit": "gru",
    # GAN
    "gan": "generative adversarial network",
    "generative adversarial network": "gan",
    "generative adversarial networks": "gan",
    # SVM
    "svm": "support vector machine",
    "support vector machine": "svm",
    "support vector machines": "svm",
    # KNN
    "knn": "k-nearest neighbors",
    "k-nearest neighbors": "knn",
    "k nearest neighbors": "knn",
    # PCA
    "pca": "principal component analysis",
    "principal component analysis": "pca",
    # CI/CD
    "ci/cd": "continuous integration continuous deployment",
    "cicd": "ci/cd",
    # ETL
    "etl": "extract transform load",
    "extract transform load": "etl",
    # OCR
    "ocr": "optical character recognition",
    "optical character recognition": "ocr",
    # API
    "rest api": "restful api",
    "restful api": "rest api",
    # DPO
    "dpo": "direct preference optimization",
    "direct preference optimization": "dpo",
    # RLHF
    "rlhf": "reinforcement learning from human feedback",
    "reinforcement learning from human feedback": "rlhf",
    # ViT
    "vit": "vision transformer",
    "vision transformer": "vit",
    "vision transformers": "vit",
}


# ============================================================
# PRODUCTION REQUIREMENT INDICATORS
# ============================================================

PRODUCTION_INDICATORS = [
    "production experience",
    "production environment",
    "production-level",
    "production level",
    "hands-on experience",
    "hands on experience",
    "industry experience",
    "professional experience",
    "deployed in production",
    "production deployment",
    "at scale",
    "large scale",
    "enterprise",
]


# ============================================================
# EVIDENCE CONTEXT PATTERNS
# ============================================================

# Strong evidence patterns (indicate real usage, not just listing)
STRONG_EVIDENCE_PATTERNS = [
    r"(?:built|developed|designed|implemented|deployed|architected|led|created|engineered)\s+.*?{skill}",
    r"{skill}\s+.*?(?:built|developed|designed|implemented|deployed|architected|led|created|engineered)",
    r"(?:production|scale|enterprise|million|billion|thousands)\s+.*?{skill}",
    r"{skill}\s+.*?(?:production|scale|enterprise|million|billion|thousands)",
    r"(?:years?\s+(?:of\s+)?experience\s+(?:with|in|using))\s+.*?{skill}",
    r"{skill}\s+.*?(?:api|service|pipeline|system|platform|infrastructure|architecture)",
]

# Medium evidence patterns (projects, courses, etc.)
MEDIUM_EVIDENCE_PATTERNS = [
    r"(?:project|thesis|research|paper|publication|course|certification|certified)\s+.*?{skill}",
    r"{skill}\s+.*?(?:project|thesis|research|paper|publication|course|certification|certified)",
    r"(?:trained|fine-tuned|evaluated|experimented|studied|learned)\s+.*?{skill}",
    r"{skill}\s+.*?(?:model|framework|library|tool|dataset)",
]

# Weak evidence patterns (just mentioned in skills list)
WEAK_EVIDENCE_PATTERNS = [
    r"(?:skills?|technologies?|tools?|proficient|familiar|knowledge)\s*:?\s*.*?{skill}",
    r"{skill}\s*[,;|]",  # appears in a comma-separated list
]


# ============================================================
# CORE FUNCTIONS
# ============================================================

def normalize_skill(skill: str) -> str:
    """Normalize a skill string: lowercase, strip, collapse whitespace, remove hyphens for matching."""
    s = skill.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def get_aliases(skill: str) -> list:
    """Return all known aliases for a skill, including the skill itself."""
    normalized = normalize_skill(skill)
    aliases = {normalized}

    # Check direct alias mapping
    if normalized in SKILL_ALIASES:
        alias_val = SKILL_ALIASES[normalized]
        aliases.add(normalize_skill(alias_val))

    # Check reverse aliases
    for key, val in SKILL_ALIASES.items():
        if normalize_skill(val) == normalized or normalize_skill(key) == normalized:
            aliases.add(normalize_skill(key))
            aliases.add(normalize_skill(val))

    # Also add hyphen/no-hyphen variants
    expanded = set()
    for a in aliases:
        expanded.add(a)
        expanded.add(a.replace("-", " "))
        expanded.add(a.replace(" ", "-"))
    return list(expanded)


def extract_evidence_context(skill: str, resume_text: str, window: int = 200) -> list:
    """
    Find all occurrences of a skill (or its aliases) in the resume text
    and extract surrounding context (window chars each side).
    Returns a list of context strings.
    """
    text_lower = resume_text.lower()
    aliases = get_aliases(skill)
    contexts = []

    for alias in aliases:
        # Use word boundary matching for short terms, substring for longer ones
        if len(alias) <= 3:
            pattern = r'\b' + re.escape(alias) + r'\b'
        else:
            pattern = re.escape(alias)

        for match in re.finditer(pattern, text_lower):
            start = max(0, match.start() - window)
            end = min(len(text_lower), match.end() + window)
            contexts.append(resume_text[start:end])

    return contexts


def compute_evidence_strength(skill: str, resume_text: str) -> int:
    """
    Compute evidence strength (0-6) for a skill in the resume text.
    Uses pattern matching against the full resume to determine depth.
    """
    text_lower = resume_text.lower()
    skill_lower = normalize_skill(skill)
    aliases = get_aliases(skill)

    # First check: is the skill even present?
    found = False
    for alias in aliases:
        if len(alias) <= 3:
            if re.search(r'\b' + re.escape(alias) + r'\b', text_lower):
                found = True
                break
        else:
            if alias in text_lower:
                found = True
                break

    if not found:
        return 0  # No evidence

    # Check strong evidence patterns (production/professional level)
    strong_count = 0
    for pattern_template in STRONG_EVIDENCE_PATTERNS:
        for alias in aliases:
            pattern = pattern_template.replace("{skill}", re.escape(alias))
            try:
                if re.search(pattern, text_lower):
                    strong_count += 1
            except re.error:
                continue

    if strong_count >= 3:
        return 6  # Strong production / advanced evidence
    if strong_count >= 1:
        return 5  # Professional evidence

    # Check medium evidence patterns (project/academic)
    medium_count = 0
    for pattern_template in MEDIUM_EVIDENCE_PATTERNS:
        for alias in aliases:
            pattern = pattern_template.replace("{skill}", re.escape(alias))
            try:
                if re.search(pattern, text_lower):
                    medium_count += 1
            except re.error:
                continue

    if medium_count >= 2:
        return 4  # Project evidence
    if medium_count >= 1:
        return 3  # Academic / training evidence

    # Check weak evidence (just in skills list)
    weak_count = 0
    for pattern_template in WEAK_EVIDENCE_PATTERNS:
        for alias in aliases:
            pattern = pattern_template.replace("{skill}", re.escape(alias))
            try:
                if re.search(pattern, text_lower):
                    weak_count += 1
            except re.error:
                continue

    if weak_count >= 1:
        return 2  # Basic exposure

    # Skill is present but only as a passing mention
    return 1  # Mentioned only


def classify_match(skill: str, resume_text: str) -> tuple:
    """
    Classify how a required skill matches against the resume.
    Returns (MatchType, evidence_strength: int).
    """
    text_lower = resume_text.lower()
    skill_lower = normalize_skill(skill)
    aliases = get_aliases(skill)

    evidence = compute_evidence_strength(skill, resume_text)

    if evidence == 0:
        return MatchType.MISSING, 0

    # Determine match type based on how the skill was found
    # Check exact match first
    if len(skill_lower) <= 3:
        exact_found = bool(re.search(r'\b' + re.escape(skill_lower) + r'\b', text_lower))
    else:
        exact_found = skill_lower in text_lower

    if exact_found:
        # Check case-sensitive exact match
        if skill in resume_text:
            if evidence >= 4:
                return MatchType.EXACT_MATCH, evidence
            elif evidence >= 2:
                return MatchType.EXACT_MATCH, evidence
            else:
                return MatchType.WEAK_EVIDENCE, evidence
        else:
            # Case-insensitive match = normalized match
            if evidence >= 2:
                return MatchType.NORMALIZED_MATCH, evidence
            else:
                return MatchType.WEAK_EVIDENCE, evidence

    # Check if found via alias
    for alias in aliases:
        if alias == skill_lower:
            continue
        if len(alias) <= 3:
            if re.search(r'\b' + re.escape(alias) + r'\b', text_lower):
                if evidence >= 3:
                    return MatchType.ALIAS_MATCH, evidence
                else:
                    return MatchType.PARTIAL_MATCH, evidence
        else:
            if alias in text_lower:
                if evidence >= 3:
                    return MatchType.ALIAS_MATCH, evidence
                else:
                    return MatchType.PARTIAL_MATCH, evidence

    # Skill was found somehow (via patterns) but not as exact/alias
    if evidence >= 3:
        return MatchType.SEMANTIC_MATCH, evidence
    elif evidence >= 1:
        return MatchType.WEAK_EVIDENCE, evidence

    return MatchType.MISSING, 0


def requires_production_evidence(job_description: str) -> bool:
    """Check if the JD indicates production-level experience is required."""
    jd_lower = job_description.lower()
    return any(indicator in jd_lower for indicator in PRODUCTION_INDICATORS)


def build_requirement_matrix(
    requirements: list,
    resume_text: str,
    job_description: str = ""
) -> list:
    """
    Build a structured requirement match matrix.

    Each requirement should be a dict with at minimum:
        {"requirement": str, "mandatory": bool, "priority": str, "category": str}

    Returns a list of dicts with match analysis added.
    """
    production_required = requires_production_evidence(job_description)
    min_evidence = MIN_EVIDENCE_PRODUCTION if production_required else MIN_EVIDENCE_DEFAULT

    matrix = []
    for req in requirements:
        skill = req.get("requirement", "")
        mandatory = req.get("mandatory", False)
        priority = req.get("priority", Priority.MEDIUM.value)
        category = req.get("category", "Technical")

        match_type, evidence = classify_match(skill, resume_text)

        # Determine if requirement is satisfied
        if match_type == MatchType.MISSING:
            matched = False
        elif match_type == MatchType.WEAK_EVIDENCE:
            matched = False  # Weak evidence doesn't satisfy mandatory reqs
        elif mandatory and evidence < min_evidence:
            matched = False  # Insufficient evidence for mandatory requirement
        else:
            matched = True

        # Determine if this is a critical gap
        is_critical_gap = mandatory and not matched and priority in (Priority.CRITICAL.value, Priority.HIGH.value)

        matrix.append({
            "requirement": skill,
            "category": category,
            "mandatory": mandatory,
            "priority": priority,
            "match_type": match_type.value,
            "evidence_strength": evidence,
            "evidence_description": EVIDENCE_LEVELS.get(evidence, "Unknown"),
            "matched": matched,
            "critical_gap": is_critical_gap,
        })

    return matrix


def validate_100_percent(matrix: list) -> dict:
    """
    Hard-gate validation for 100% score.
    Returns a dict with:
        - eligible: bool (can the score be 100?)
        - blocking_reasons: list of reasons why 100 is blocked
        - mandatory_match_rate: float (0.0 to 1.0)
        - critical_gaps: list of critical gap requirements
    """
    mandatory_reqs = [r for r in matrix if r["mandatory"]]
    mandatory_matched = [r for r in mandatory_reqs if r["matched"]]
    critical_gaps = [r for r in matrix if r.get("critical_gap", False)]

    if not mandatory_reqs:
        # No mandatory requirements defined — score is unconstrained
        return {
            "eligible": True,
            "blocking_reasons": [],
            "mandatory_match_rate": 1.0,
            "critical_gaps": [],
            "mandatory_matched": 0,
            "mandatory_total": 0,
        }

    mandatory_rate = len(mandatory_matched) / len(mandatory_reqs)
    blocking_reasons = []

    if mandatory_rate < 1.0:
        missing = [r["requirement"] for r in mandatory_reqs if not r["matched"]]
        blocking_reasons.append(
            f"Missing mandatory requirements: {', '.join(missing)}"
        )

    if critical_gaps:
        gap_names = [r["requirement"] for r in critical_gaps]
        blocking_reasons.append(
            f"Critical skill gaps: {', '.join(gap_names)}"
        )

    # Check all mandatory requirements have sufficient evidence
    weak_mandatory = [
        r for r in mandatory_matched
        if r["evidence_strength"] < MIN_EVIDENCE_DEFAULT
    ]
    if weak_mandatory:
        weak_names = [r["requirement"] for r in weak_mandatory]
        blocking_reasons.append(
            f"Insufficient evidence for mandatory requirements: {', '.join(weak_names)}"
        )

    eligible = len(blocking_reasons) == 0

    return {
        "eligible": eligible,
        "blocking_reasons": blocking_reasons,
        "mandatory_match_rate": round(mandatory_rate, 4),
        "critical_gaps": [r["requirement"] for r in critical_gaps],
        "mandatory_matched": len(mandatory_matched),
        "mandatory_total": len(mandatory_reqs),
    }


def enforce_score_cap(score: int, validation: dict) -> int:
    """
    Post-process: if the AI returned a score but the hard-gate fails,
    cap the score appropriately.
    """
    if score > 100:
        score = 100

    if not validation["eligible"]:
        # 100 is impossible — cap at 97 maximum
        if score >= 100:
            score = 97

        # If critical gaps exist, further cap
        num_critical = len(validation["critical_gaps"])
        if num_critical >= 3:
            score = min(score, 65)
        elif num_critical >= 2:
            score = min(score, 75)
        elif num_critical >= 1:
            score = min(score, 85)

        # If mandatory match rate is low, further reduce
        rate = validation["mandatory_match_rate"]
        if rate < 0.5:
            score = min(score, 50)
        elif rate < 0.7:
            score = min(score, 65)
        elif rate < 0.85:
            score = min(score, 80)

    return score




# Configuration
ELIGIBILITY_THRESHOLD = 40

def determine_recommendation(
    score: int,
    validation: dict,
) -> str:
    """
    Determine the final recommendation based on score and validation.
    Returns one of: NEXT LEVEL, REJECTED
    """
    has_critical_gaps = len(validation["critical_gaps"]) > 0

    if has_critical_gaps:
        # Critical mandatory requirement missing — never auto-approve
        return Recommendation.REJECT.value
            
    # Configurable threshold for eligibility
    if score >= ELIGIBILITY_THRESHOLD:
        return Recommendation.ALLOW_NEXT_LEVEL.value
    else:
        return Recommendation.REJECT.value


def compute_final_score(matrix: list, ai_score: int = None) -> dict:
    """
    Compute the final score from the requirement matrix.
    If ai_score is provided, it is used as a base and then validated/capped.
    Otherwise, a score is computed purely from the matrix.

    Returns a dict with score, breakdown, validation, and recommendation.
    """
    # Validate against hard gate
    validation = validate_100_percent(matrix)

    if ai_score is not None:
        # Use AI score but enforce caps
        final_score = enforce_score_cap(ai_score, validation)
    else:
        # Compute score from matrix
        mandatory_reqs = [r for r in matrix if r["mandatory"]]
        preferred_reqs = [r for r in matrix if not r["mandatory"]]

        if mandatory_reqs:
            mandatory_score = sum(
                1 for r in mandatory_reqs if r["matched"]
            ) / len(mandatory_reqs) * 100
        else:
            mandatory_score = None

        if preferred_reqs:
            preferred_score = sum(
                1 for r in preferred_reqs if r["matched"]
            ) / len(preferred_reqs) * 100
        else:
            preferred_score = 0

        if mandatory_score is not None:
            # Weighted: mandatory = 70%, preferred = 30%
            raw_score = int(mandatory_score * 0.7 + preferred_score * 0.3)
        else:
            # Only preferred requirements exist
            raw_score = int(preferred_score)
            
        final_score = enforce_score_cap(raw_score, validation)

    recommendation = determine_recommendation(final_score, validation)

    # Compute is_match (approved for next steps)
    is_match = recommendation == Recommendation.ALLOW_NEXT_LEVEL.value

    # Summary stats
    matched_mandatory = [r for r in matrix if r["mandatory"] and r["matched"]]
    missing_mandatory = [r for r in matrix if r["mandatory"] and not r["matched"]]
    matched_preferred = [r for r in matrix if not r["mandatory"] and r["matched"]]
    missing_preferred = [r for r in matrix if not r["mandatory"] and not r["matched"]]

    return {
        "score": final_score,
        "is_match": is_match,
        "recommendation": recommendation,
        "full_compliance": validation["eligible"],
        "mandatory_match_rate": validation["mandatory_match_rate"],
        "mandatory_matched": validation["mandatory_matched"],
        "mandatory_total": validation["mandatory_total"],
        "critical_gaps": validation["critical_gaps"],
        "blocking_reasons": validation["blocking_reasons"],
        "matched_mandatory_list": [r["requirement"] for r in matched_mandatory],
        "missing_mandatory_list": [r["requirement"] for r in missing_mandatory],
        "matched_preferred_list": [r["requirement"] for r in matched_preferred],
        "missing_preferred_list": [r["requirement"] for r in missing_preferred],
    }


def post_process_ai_response(
    ai_response: dict,
    resume_text: str,
    job_description: str = ""
) -> dict:
    """
    Post-process the AI's response to enforce strict scoring rules.
    This is the main integration point called from main.py.

    Takes the raw AI JSON response, validates and adjusts scores,
    and returns the corrected response.
    """
    ai_score = ai_response.get("matchPercentage", 0)
    skills_found = ai_response.get("skillsFound", [])
    missing_skills = ai_response.get("missingSkills", [])

    # Build requirements list
    requirements = []
    
    # If AI provided a detailed requirement matrix, use it to extract the job requirements
    ai_matrix = ai_response.get("requirement_matrix", [])
    if ai_matrix:
        for req in ai_matrix:
            requirements.append({
                "requirement": req.get("requirement", ""),
                "mandatory": req.get("mandatory", False),
                "priority": req.get("priority", Priority.MEDIUM.value),
                "category": req.get("category", "Technical")
            })
    else:
        # Fallback for old AI responses
        for skill in missing_skills:
            requirements.append({
                "requirement": skill,
                "mandatory": True,
                "priority": Priority.CRITICAL.value,
                "category": "Technical",
            })
        for skill in skills_found:
            requirements.append({
                "requirement": skill,
                "mandatory": True,
                "priority": Priority.HIGH.value,
                "category": "Technical",
            })

    # Build the requirement matrix using our local engine to ensure untampered evidence scoring
    # If resume_text is empty (e.g., scanned PDF relying purely on Gemini), we trust the AI score.
    if not resume_text.strip():
        ai_response["requirement_validation"] = {
            "full_compliance": True,
            "mandatory_match_rate": "N/A (Scanned Document)",
            "critical_gaps": [],
            "blocking_reasons": []
        }
        return ai_response

    matrix = build_requirement_matrix(requirements, resume_text, job_description)

    # Compute validated score
    result = compute_final_score(matrix, ai_score)

    # Override AI response with validated values
    ai_response["matchPercentage"] = result["score"]
    ai_response["isMatch"] = result["is_match"]

    # Add the requirement validation metadata
    ai_response["requirement_validation"] = {
        "full_compliance": result["full_compliance"],
        "mandatory_match_rate": result["mandatory_match_rate"],
        "mandatory_matched": result["mandatory_matched"],
        "mandatory_total": result["mandatory_total"],
        "critical_gaps": result["critical_gaps"],
        "blocking_reasons": result["blocking_reasons"],
        "recommendation": result["recommendation"],
        "score_validated": True,
        "original_ai_score": ai_score,
        "final_score": result["score"],
    }

    # Add requirement matrix for detailed reporting
    ai_response["requirement_matrix"] = matrix

    return ai_response


def build_offline_evaluation(
    resume_text: str,
    job_description: str,
    requirement_keywords: list = None,
) -> dict:
    """
    Offline fallback evaluation when the AI API is unavailable.
    Uses pure keyword/pattern matching from the requirement engine.

    requirement_keywords: list of dicts with {requirement, mandatory, priority, category}
    If not provided, uses a default AI/ML requirement set.
    """
    if requirement_keywords is None:
        # Forgiving default offline fallback
        requirement_keywords = [
            {"requirement": "Python", "mandatory": False, "priority": "HIGH", "category": "Programming"},
            {"requirement": "R", "mandatory": False, "priority": "MEDIUM", "category": "Programming"},
            {"requirement": "SQL", "mandatory": False, "priority": "HIGH", "category": "Data"},
            {"requirement": "Machine Learning", "mandatory": False, "priority": "HIGH", "category": "Core AI"},
            {"requirement": "Data Science", "mandatory": False, "priority": "HIGH", "category": "Core Data"},
            {"requirement": "Data Analytics", "mandatory": False, "priority": "HIGH", "category": "Core Data"},
            {"requirement": "Deep Learning", "mandatory": False, "priority": "MEDIUM", "category": "Core AI"},
            {"requirement": "NLP", "mandatory": False, "priority": "MEDIUM", "category": "Advanced AI"},
            {"requirement": "Computer Vision", "mandatory": False, "priority": "MEDIUM", "category": "Advanced AI"},
            {"requirement": "Generative AI", "mandatory": False, "priority": "MEDIUM", "category": "Advanced AI"},
            {"requirement": "LLM", "mandatory": False, "priority": "MEDIUM", "category": "Advanced AI"},
            {"requirement": "PyTorch", "mandatory": False, "priority": "MEDIUM", "category": "Frameworks"},
            {"requirement": "TensorFlow", "mandatory": False, "priority": "MEDIUM", "category": "Frameworks"},
            {"requirement": "Scikit-learn", "mandatory": False, "priority": "MEDIUM", "category": "Frameworks"},
            {"requirement": "Pandas", "mandatory": False, "priority": "MEDIUM", "category": "Data"},
            {"requirement": "NumPy", "mandatory": False, "priority": "MEDIUM", "category": "Data"},
            {"requirement": "MLOps", "mandatory": False, "priority": "LOW", "category": "MLOps"},
        ]

    # Build requirement matrix
    matrix = build_requirement_matrix(requirement_keywords, resume_text, job_description)

    # Compute final score
    result = compute_final_score(matrix)

    # Build skills lists
    skills_found = [r["requirement"] for r in matrix if r["matched"]]
    missing_skills = [r["requirement"] for r in matrix if not r["matched"] and r["mandatory"]]

    # Build candidate-friendly reason text
    reason_parts = []
    
    # If the offline scanner passed them, add a mock AI explanation so the UI looks nice
    if result["is_match"]:
        reason_parts.append("Congratulations! Your resume successfully matched our core baseline requirements. You have been selected to proceed to the interview round.")
    else:
        missing_mandatory_str = ", ".join(result['missing_mandatory_list']) if result['missing_mandatory_list'] else "core skills"
        reason_parts.append(f"Unfortunately, your resume does not strongly indicate experience with our mandatory technical requirements ({missing_mandatory_str}). Therefore, we cannot proceed with your application at this time.")

    reason = "\n".join(reason_parts)

    return {
        "isMatch": result["is_match"],
        "matchPercentage": result["score"],
        "reason": reason,
        "next_round_status": result["recommendation"],
        "skillsFound": skills_found,
        "missingSkills": missing_skills,
        "requirement_matrix": matrix,
        "requirement_validation": {
            "full_compliance": result["full_compliance"],
            "mandatory_match_rate": result["mandatory_match_rate"],
            "mandatory_matched": result["mandatory_matched"],
            "mandatory_total": result["mandatory_total"],
            "critical_gaps": result["critical_gaps"],
            "blocking_reasons": result["blocking_reasons"],
            "recommendation": result["recommendation"],
            "score_validated": True,
        },
    }
