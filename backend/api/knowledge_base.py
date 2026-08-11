# ATS-Optimized AI/ML Recruitment Knowledge Base

import json

# ============================================================
# COMPREHENSIVE ATS KEYWORD TAXONOMY
# ============================================================

TAXONOMY = {
    "PROGRAMMING": [
        "Python", "R", "SQL", "Java", "C", "C++", "JavaScript", "TypeScript",
        "Go", "Rust", "Scala", "Bash"
    ],
    "MACHINE_LEARNING": [
        "Machine Learning", "ML", "Supervised Learning", "Unsupervised Learning",
        "Semi-Supervised Learning", "Self-Supervised Learning",
        "Regression", "Classification", "Clustering", "Recommendation Systems",
        "Anomaly Detection", "Time Series", "Feature Engineering",
        "Feature Selection", "Feature Scaling", "Dimensionality Reduction",
        "Cross Validation", "Hyperparameter Tuning", "Model Selection",
        "Model Evaluation", "Model Optimization", "Model Calibration",
        "Imbalanced Learning", "Ensemble Learning", "Decision Tree",
        "Random Forest", "Extra Trees", "XGBoost", "LightGBM", "CatBoost",
        "SVM", "KNN", "Naive Bayes", "K-Means", "DBSCAN", "PCA",
        "Isolation Forest", "Gradient Boosting"
    ],
    "DEEP_LEARNING": [
        "Deep Learning", "Neural Networks", "ANN", "CNN", "RNN", "LSTM", "GRU",
        "Transformers", "Attention", "Transfer Learning", "Autoencoders",
        "GAN", "Diffusion Models", "PyTorch", "TensorFlow", "Keras", "JAX"
    ],
    "NLP": [
        "NLP", "Natural Language Processing", "Tokenization", "NER",
        "Named Entity Recognition", "Text Classification", "Sentiment Analysis",
        "Text Summarization", "Question Answering", "Semantic Search",
        "Information Retrieval", "Embeddings", "BERT", "RoBERTa", "T5",
        "Transformers"
    ],
    "GENERATIVE_AI": [
        "Generative AI", "GenAI", "LLM", "Large Language Models",
        "Foundation Models", "Prompt Engineering", "Fine-Tuning", "Fine Tuning",
        "LoRA", "QLoRA", "PEFT", "Embeddings", "RAG",
        "Retrieval Augmented Generation", "Vector Search", "Vector Database",
        "Semantic Retrieval", "Reranking", "Multimodal AI", "Function Calling",
        "Tool Calling"
    ],
    "AI_AGENTS": [
        "AI Agents", "Agentic AI", "Planning", "Reasoning", "Tool Calling",
        "Function Calling", "Memory", "Multi-Agent Systems",
        "Workflow Orchestration", "Human-in-the-Loop", "Agent Evaluation",
        "Agent Observability"
    ],
    "LLM_ECOSYSTEM": [
        "OpenAI", "GPT", "Claude", "Gemini", "Llama", "Mistral", "Hugging Face",
        "LangChain", "LangGraph", "LlamaIndex", "CrewAI", "AutoGen",
        "Semantic Kernel"
    ],
    "RAG_VECTOR": [
        "FAISS", "Pinecone", "Chroma", "Weaviate", "Milvus", "Qdrant",
        "Vector Database", "Vector Store", "Similarity Search", "Hybrid Search",
        "Chunking", "Retrieval", "Reranking"
    ],
    "COMPUTER_VISION": [
        "Computer Vision", "OpenCV", "Image Processing", "Image Classification", 
        "Object Detection", "Object Tracking", "Image Segmentation", 
        "Semantic Segmentation", "Instance Segmentation", "YOLO", "YOLOv5", 
        "YOLOv8", "YOLOv9", "YOLOv10", "OCR", "Face Detection", "Image Recognition", 
        "Pose Estimation", "Vision Transformers", "ViT"
    ],
    "SPEECH_AUDIO": [
        "Speech Recognition", "ASR", "Automatic Speech Recognition", 
        "Speech-to-Text", "TTS", "Text-to-Speech", "Whisper", 
        "Audio Classification", "Speaker Recognition"
    ],
    "MLOPS": [
        "MLOps", "Model Deployment", "Model Serving", "Model Monitoring",
        "Model Versioning", "MLflow", "Docker", "Kubernetes", "CI/CD",
        "FastAPI", "REST API"
    ],
    "CLOUD": [
        "AWS", "Azure", "GCP", "SageMaker", "Vertex AI", "Azure ML", "Bedrock"
    ],
    "API_BACKEND": [
        "FastAPI", "Flask", "REST API", "GraphQL", "Microservices",
        "Python Backend", "API Development", "Async Processing"
    ],
    "DATA_ENGINEERING": [
        "NumPy", "Pandas", "Polars", "Apache Spark", "PySpark", "Kafka",
        "Airflow", "Databricks", "Snowflake", "dbt", "ETL", "ELT",
        "Data Pipelines", "Data Warehouse", "Data Lake", "Data Lakehouse"
    ],
    "DATABASES": [
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch"
    ],
    "DATA_SCIENCE": [
        "Data Science", "Statistics", "Descriptive Statistics", "Inferential Statistics", 
        "Probability", "Statistical Modeling", "Predictive Modeling", "Exploratory Data Analysis", 
        "EDA", "Feature Engineering", "Data Cleaning", "Data Wrangling", "Data Preprocessing", 
        "Data Visualization", "Hypothesis Testing", "A/B Testing", "Regression Analysis", 
        "Forecasting", "Predictive Analytics", "Model Evaluation", "Business Analytics"
    ],
    "DATA_ANALYTICS": [
        "Data Analytics", "Data Analysis", "Business Analytics", "Descriptive Analytics", 
        "Diagnostic Analytics", "Predictive Analytics", "Prescriptive Analytics", 
        "KPI Analysis", "Dashboarding", "Reporting", "Data Visualization", "Trend Analysis", 
        "Cohort Analysis", "Funnel Analysis", "Segmentation", "SQL Analytics", "BI Analytics"
    ],
    "BI_ANALYTICS_TOOLS": [
        "Power BI", "Tableau", "Looker", "Excel", "Advanced Excel", "DAX", "Power Query"
    ]
}

# ============================================================
# ATS ALIAS DICTIONARY
# ============================================================

SKILL_ALIASES = {
    "NLP": "Natural Language Processing",
    "ML": "Machine Learning",
    "DL": "Deep Learning",
    "GenAI": "Generative AI",
    "LLM": "Large Language Model",
    "RAG": "Retrieval Augmented Generation",
    "NER": "Named Entity Recognition",
    "CV": "Computer Vision",
    "K8s": "Kubernetes",
    "SKLearn": "Scikit-learn",
    "TF": "TensorFlow",
    "PyTorch": "Torch",
    "ANN": "Artificial Neural Network",
    "CNN": "Convolutional Neural Network",
    "RNN": "Recurrent Neural Network",
    "LSTM": "Long Short-Term Memory",
    "GRU": "Gated Recurrent Unit",
    "GAN": "Generative Adversarial Network",
    "SVM": "Support Vector Machine",
    "KNN": "K-Nearest Neighbors",
    "PCA": "Principal Component Analysis",
    "ViT": "Vision Transformer",
    "PEFT": "Parameter Efficient Fine Tuning",
    "LoRA": "Low Rank Adaptation",
    "QLoRA": "Quantized Low Rank Adaptation",
    "OCR": "Optical Character Recognition",
}

# ============================================================
# SKILL RELATIONSHIPS (for semantic matching)
# ============================================================

SKILL_RELATIONSHIPS = {
    "RAG": ["Embeddings", "Vector Database", "Retrieval", "Chunking", "Reranking", "LLM", "FAISS", "Pinecone"],
    "LLM": ["Transformer", "Attention", "Tokenization", "Embeddings", "Prompt Engineering"],
    "Computer Vision": ["OpenCV", "Image Processing", "Object Detection", "YOLO"],
    "MLOps": ["Docker", "Kubernetes", "CI/CD", "Model Deployment", "Model Monitoring", "MLflow"],
    "AI Agent": ["LLM", "Tool Calling", "Planning", "Memory", "Workflow Orchestration", "Evaluation"],
    "Deep Learning": ["Neural Networks", "PyTorch", "TensorFlow", "Keras"],
    "NLP": ["Tokenization", "BERT", "Transformers", "Text Classification", "Sentiment Analysis", "Embeddings"],
    "Generative AI": ["LLM", "Prompt Engineering", "Fine-Tuning", "Diffusion Models", "RAG"],
    "Data Science": ["Statistics", "Predictive Modeling", "EDA", "Machine Learning", "Data Visualization"],
    "Data Analytics": ["Data Analysis", "KPI Analysis", "Dashboarding", "SQL Analytics", "Power BI", "Tableau"]
}

# ============================================================
# SENIORITY MATRIX & EVIDENCE
# ============================================================

SENIORITY_MATRIX = {
    "Intern": {"years": "0-1", "scope": "Learning, assisted implementation"},
    "Junior": {"years": "1-3", "scope": "Task execution, guided implementation"},
    "Mid-Level": {"years": "3-5", "scope": "Independent execution, system component design"},
    "Senior": {"years": "5-8+", "scope": "System design, mentoring, production ownership"},
    "Lead": {"years": "7-10+", "scope": "Team leadership, architecture, technical strategy"},
    "Principal/Architect": {"years": "10+", "scope": "Cross-organization architecture, business impact"},
}

EVIDENCE_STRENGTH = {
    0: "No evidence",
    1: "Mentioned only (Weak)",
    2: "Basic/Academic exposure",
    3: "Course/Certification/Personal Project exposure",
    4: "Project evidence",
    5: "Professional evidence",
    6: "Production/advanced evidence (Strongest)",
}

# ============================================================
# ROLE-SPECIFIC ATS PROFILES
# ============================================================

ROLE_PROFILES = {
    "LLM_Engineer": [
        "LLM", "RAG", "Embeddings", "Vector Database", "Prompt Engineering", 
        "Fine-Tuning", "Evaluation", "Agents"
    ],
    "Computer_Vision_Engineer": [
        "OpenCV", "CNN", "YOLO", "Object Detection", "Image Processing", "Segmentation"
    ],
    "MLOps_Engineer": [
        "Docker", "Kubernetes", "MLflow", "Cloud", "CI/CD", "Monitoring", "Deployment"
    ],
    "AI_Engineer": [
        "Python", "Machine Learning", "Deep Learning", "NLP", "LLM", "Cloud", "Deployment"
    ],
    "Data_Scientist": [
        "Python", "Machine Learning", "Statistics", "Pandas", "NumPy", "SQL", "Visualization"
    ]
}

# ============================================================
# REQUIREMENT CATEGORIES & MATCH TYPES
# ============================================================

REQUIREMENT_CATEGORIES = [
    "MANDATORY_TECHNICAL",
    "PREFERRED_TECHNICAL",
    "EXPERIENCE",
    "EDUCATION",
    "CERTIFICATION",
    "DOMAIN",
    "RESPONSIBILITIES",
    "SOFT_SKILLS",
]

MATCH_TYPES = {
    "EXACT_MATCH": "Required term directly present in resume with identical spelling",
    "NORMALIZED_MATCH": "Same term with different case/spacing/punctuation",
    "ALIAS_MATCH": "Recognized equivalent term (e.g. NLP = Natural Language Processing)",
    "SEMANTIC_MATCH": "Genuinely equivalent concept confirmed by evidence context",
    "PARTIAL_MATCH": "Related but not equivalent skill, or partial coverage",
    "WEAK_EVIDENCE": "Skill mentioned but no supporting usage evidence",
    "MISSING": "No evidence of the skill in the resume",
    "NOT_APPLICABLE": "Requirement not relevant to this evaluation",
}

# ============================================================
# PROFICIENCY ESTIMATION MATRIX
# ============================================================

PROFICIENCY_ESTIMATION = {
    "BEGINNER": {
        "definition": "Basic/Academic exposure. Mentioned in skills, simple projects, or theoretical knowledge.",
        "expected_concepts": ["Artificial Intelligence", "Machine Learning", "Deep Learning", "NLP", "Generative AI", "Data Science", "Statistics", "Computer Vision", "RAG", "Data Analytics"],
        "evidence_markers": ["academic", "coursework", "certification", "basic understanding", "internship"]
    },
    "INTERMEDIATE": {
        "definition": "Independent execution, mid-level complexity, professional exposure.",
        "expected_concepts": ["Feature Engineering", "Hyperparameter Tuning", "Ensemble Learning", "CNN", "RNN", "Semantic Search", "Prompt Engineering", "CI/CD", "MLflow", "A/B Testing", "Dashboarding", "SQL Joins"],
        "evidence_markers": ["used in production", "developed", "deployed", "implemented", "optimized"]
    },
    "ADVANCED": {
        "definition": "System design, advanced optimization, complex architectures, production ownership.",
        "expected_concepts": ["AutoML", "Transformers", "ViT", "LoRA", "PEFT", "RLHF", "Model Alignment", "Query Expansion", "Graph RAG", "Instance Segmentation", "Feature Store", "Bayesian Inference", "Predictive Analytics"],
        "evidence_markers": ["architected", "led", "designed", "scaled", "production deployment"]
    },
    "EXPERT": {
        "definition": "Cross-organization architecture, team leadership, deep theoretical and applied knowledge.",
        "expected_concepts": ["AI Architecture", "Distributed Machine Learning", "Enterprise LLM Architecture", "Advanced Causal Inference", "Production ML Platforms", "Advanced Decision Science"],
        "evidence_markers": ["chief", "principal", "architect", "strategy", "enterprise-scale"]
    }
}

# ============================================================
# PUBLIC API
# ============================================================

def get_knowledge_context():
    """Return the full knowledge base as a JSON string for AI ATS prompt injection."""
    return json.dumps({
        "taxonomy": TAXONOMY,
        "aliases": SKILL_ALIASES,
        "relationships": SKILL_RELATIONSHIPS,
        "evidence_strength": EVIDENCE_STRENGTH,
        "seniority_matrix": SENIORITY_MATRIX,
        "proficiency_estimation": PROFICIENCY_ESTIMATION,
        "role_profiles": ROLE_PROFILES,
        "requirement_categories": REQUIREMENT_CATEGORIES,
        "match_types": MATCH_TYPES,
    }, indent=2)
