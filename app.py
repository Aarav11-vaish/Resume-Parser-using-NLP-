# app.py

import re
import json
from flask import Flask, request, jsonify, render_template
from pdfminer.high_level import extract_text as extract_pdf_text
import nltk
from summa import summarizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

app = Flask(__name__)

# Regular expressions for phone, email, URL, and hyperlinks
PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
URL_REG = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
HYPERLINK_REG = re.compile(r'\bhttps?://\S+\b')  # Regex for hyperlinks embedded in text

# Expanded skills database
SKILLS_DB = [
    'python', 'java', 'c++', 'c#', 'javascript', 'ruby', 'swift', 'go', 'rust', 'php', 'typescript', 'kotlin', 'r',
    'machine learning', 'data science', 'deep learning', 'neural networks', 'nlp', 'natural language processing',
    'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'data analysis',
    'data visualization', 'big data', 'hadoop', 'spark', 'sql', 'nosql', 'mongodb', 'data mining',
    'html', 'css', 'bootstrap', 'tailwind', 'sass', 'less', 'react', 'angular', 'vue', 'django', 'flask', 'express',
    'nodejs', 'nextjs', 'gatsby', 'rest api', 'graphql', 'web development', 'front end', 'back end', 'full stack',
    'aws', 'azure', 'google cloud', 'cloud computing', 'docker', 'kubernetes', 'devops', 'ci/cd', 'jenkins', 'terraform',
    'ansible', 'puppet', 'chef', 'serverless', 'cloudformation', 'helm',
    'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'agile', 'scrum', 'kanban', 'test driven development',
    'unit testing', 'integration testing', 'continuous integration', 'continuous deployment', 'version control',
    'automation', 'selenium', 'cucumber', 'jenkins',
    'mysql', 'postgresql', 'sqlite', 'oracle', 'sql server', 'mariadb', 'redis', 'cassandra', 'elasticsearch',
    'firebase', 'dynamodb', 'redshift', 'snowflake',
    'android', 'ios', 'flutter', 'react native', 'swift', 'kotlin', 'xamarin', 'mobile development', 'ionic',
    'microsoft word', 'word', 'excel', 'powerpoint', 'office 365', 'outlook', 'visio', 'microsoft project', 'trello',
    'slack', 'zoom', 'teams', 'communication', 'presentation', 'project management', 'time management', 'teamwork',
    'leadership', 'problem solving', 'critical thinking', 'analytical skills', 'creativity', 'adaptability',
    'attention to detail', 'english', 'spanish', 'french', 'german', 'chinese', 'japanese', 'italian', 'portuguese',
    'russian', 'korean'
]

# Reserved words for educational institutions
RESERVED_WORDS = [
    'school', 'college', 'univers', 'academy', 'faculty', 'institute',
    'faculdades', 'schola', 'schule', 'lise', 'lyceum', 'lycee', 'polytechnic',
    'kolej', 'ünivers', 'okul',
]

def extract_text_from_pdf(pdf_path):
    return extract_pdf_text(pdf_path)

def extract_phone_number(text):
    phone = re.findall(PHONE_REG, text)
    if phone:
        number = ''.join(phone[0])
        if text.find(number) >= 0 and len(number) < 16:
            return number
    return None

def extract_emails(text):
    return re.findall(EMAIL_REG, text)

def extract_urls(text):
    return re.findall(URL_REG, text)

def extract_hyperlinks(text):
    return re.findall(HYPERLINK_REG, text)

def extract_skills(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(text)
    filtered_tokens = [w for w in word_tokens if w not in stop_words and w.isalpha()]
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
    found_skills = set()
    for token in filtered_tokens:
        if token.lower() in SKILLS_DB:
            found_skills.add(token)
    for ngram in bigrams_trigrams:
        if ngram.lower() in SKILLS_DB:
            found_skills.add(ngram)
    return found_skills

def extract_education(text):
    organizations = []
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION':
                organizations.append(' '.join(c[0] for c in chunk.leaves()))
    education = set()
    for org in organizations:
        for word in RESERVED_WORDS:
            if word in org.lower():
                education.add(org)
    return education

def summarize_text(text):
    return summarizer.summarize(text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    file_path = 'Aarav Vaish_Resume.pdf'
    file.save(file_path)

    pdf_text = extract_text_from_pdf(file_path)

    phone_number = extract_phone_number(pdf_text)
    emails = extract_emails(pdf_text)
    urls_pdf = extract_urls(pdf_text)
    hyperlinks_pdf = extract_hyperlinks(pdf_text)
    skills = extract_skills(pdf_text)
    education_information = extract_education(pdf_text)
    summary_pdf = summarize_text(pdf_text)

    return jsonify({
        'success': True,
        'phone_number': phone_number,
        'emails': emails,
        'urls': urls_pdf,
        'hyperlinks': hyperlinks_pdf,
        'skills': list(skills),
        'education': list(education_information),
        'summary': summary_pdf
    })

if __name__ == '__main__':
    app.run(debug=True)
