from flask import Flask, request, render_template, redirect, url_for, flash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string
import pdfplumber
import docx
import os
import spacy
import re


nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to store uploaded files
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    print(f"Warning: No text found on page {page.page_number}")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return '\n'.join([para.text for para in doc.paragraphs])

# Similarity function
def compute_similarity(resume_text, job_text):
    documents = [resume_text, job_text]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1] * 100  # Convert to percentage

# Function for keyword analysis
def keyword_analysis(resume_text, job_text):
    resume_words = set(preprocess(resume_text).split())
    job_words = set(preprocess(job_text).split())
    missing_keywords = job_words - resume_words
    return list(missing_keywords)

# Function to check length of resume
def check_resume_length(resume_text):
    words = resume_text.split()
    return len(words)

# Function to count the number of specific sections in the resume
def count_sections(resume_text):
    sections = ['education', 'experience', 'skills', 'projects', 'certifications']
    section_count = {}
    for section in sections:
        section_count[section.capitalize()] = resume_text.lower().count(section)
    return section_count

def extract_keywords(job_description):
    doc = nlp(job_description)
    # Extract nouns and proper nouns as potential keywords
    keywords = {token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']}
    return keywords

# Function to detect email, phone number, and name
def detect_contact_info(resume_text):
    # Regex pattern for email
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    email = re.findall(email_pattern, resume_text)
    
    # Regex pattern for phone number (allowing various formats)
    phone_pattern = r'(\+?\d{1,4}?[-.\s]?(\d{1,3}?[-.\s]?)?\(?\d{2,4}?\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5})'
    phone_number = re.findall(phone_pattern, resume_text)
    
    # Assume the name is mentioned in the first 5 lines of the resume and does not contain numbers
    name = detect_name_in_resume(resume_text)
    
    return email, phone_number, name

# Function to detect name from the top of the resume (first few lines)
def detect_name_in_resume(resume_text):
    # Get first 5 lines of the resume
    lines = resume_text.splitlines()[:5]
    
    # Join the lines and apply named entity recognition (NER)
    doc = nlp(' '.join(lines))
    
    # Extract potential names using spaCy NER (PERSON)
    possible_names = [ent.text for ent in doc.ents if ent.label_ == 'PERSON' and not any(char.isdigit() for char in ent.text)]
    
    # Fallback: Use regex to find capitalized words (proper nouns) as a potential name
    if not possible_names:
        capitalized_words = re.findall(r'\b[A-Z][a-z]*\b', ' '.join(lines))
        if capitalized_words:
            # Assuming the first capitalized word(s) in the top 5 lines is a name
            possible_names = [' '.join(capitalized_words[:2])]  # Taking first two as first and last name
    
    # Return detected name or 'Name not found' if nothing is detected
    return possible_names if possible_names else ['Name not found']

def extract_hard_skills(job_text):
    # Define a list of potential hard skills to match against the job description
    hard_skills_list = [
        'python', 'java', 'sql', 'javascript', 'c++', 'html', 'css', 'data analysis', 
        'machine learning', 'deep learning', 'aws', 'cloud computing', 'docker', 
        'kubernetes', 'big data', 'hadoop', 'spark', 'tensorflow', 'pytorch', 'react', 
        'node.js', 'angular', 'linux', 'git', 'flask', 'django'
    ]
    
    # Preprocess the job description and extract hard skills
    job_text = preprocess(job_text).lower()
    extracted_hard_skills = [skill for skill in hard_skills_list if skill in job_text]
    
    return extracted_hard_skills

def compute_similarity(resume_text, job_text):
    documents = [resume_text, job_text]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1] * 100  # Convert to percentage


def analyze_resume_for_improvement(resume_text, job_text):
    processed_resume = preprocess(resume_text)
    processed_job = preprocess(job_text)

    # Extract hard skills dynamically from job description
    hard_skills_keywords = extract_hard_skills(job_text)
    
    # Hard Skills Detection
    detected_hard_skills = [skill for skill in hard_skills_keywords if skill.lower() in processed_resume]
    missing_hard_skills = [skill for skill in hard_skills_keywords if skill.lower() not in processed_resume]
    
    # Soft Skills Detection
    soft_skills_keywords = {'communication', 'teamwork', 'problem-solving', 'leadership', 'adaptability'}
    detected_soft_skills = [skill for skill in soft_skills_keywords if skill.lower() in processed_resume]
    missing_soft_skills = [skill for skill in soft_skills_keywords if skill.lower() not in processed_resume]
    
    # Check for missing sections
    sections = ['experience', 'education', 'skills', 'projects']
    section_count = {section: processed_resume.count(section) for section in sections}
    missing_sections = [section for section in sections if section_count[section] == 0]
    
    return detected_hard_skills, missing_hard_skills, detected_soft_skills, missing_soft_skills, section_count, missing_sections

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score_resume():
    if 'resume_file' not in request.files:
        return "No resume file part", 400

    resume_file = request.files['resume_file']
    job_text = request.form.get('job_description', '')

    if not resume_file or resume_file.filename == '':
        return "No selected resume file", 400

    if not job_text:
        return "Job description is missing", 400

    # Save the uploaded resume file
    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(resume_path)

    # Extract text from the resume file
    try:
        if resume_file.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_path)
        elif resume_file.filename.endswith('.docx'):
            resume_text = extract_text_from_docx(resume_path)
        else:
            return "Unsupported resume file type", 400
    except Exception as e:
        return f"Error processing resume file: {e}", 500

    # Analyze resume for improvement
    detected_hard_skills, missing_hard_skills, detected_soft_skills, missing_soft_skills, section_count, missing_sections = analyze_resume_for_improvement(resume_text, job_text)

    # Combine hard and soft skills
    skills_summary = {
        "Hard Skills": {
            "Detected": detected_hard_skills,
            "Missing": missing_hard_skills
        },
        "Soft Skills": {
            "Detected": detected_soft_skills,
            "Missing": missing_soft_skills
        }
    }

    # Detect contact info
    email, phone_number, name = detect_contact_info(resume_text)
    
    # Calculate similarity score
    similarity_score = compute_similarity(resume_text, job_text)
    match_status = 'Match score is below 85%' if similarity_score < 85 else 'Congratulations! You have achieved a good match score.'

    rounded_score = round(similarity_score)

    # Render the result template with contact info and skill analysis
    return render_template('score.html', 
                            resume_length=len(resume_text.split()),
                            skills_summary=skills_summary,
                            score=rounded_score,
                            match_status=match_status,
                            contact_info={'email': email, 'phone': phone_number, 'name': name})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
