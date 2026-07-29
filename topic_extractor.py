import os
import json
from pypdf import PdfReader
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_pdf_snippet(filepath, max_chars=1000):
    try:
        reader = PdfReader(filepath)
        if len(reader.pages) > 0:
            text = reader.pages[0].extract_text()
            if text:
                return text[:max_chars]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return ""

def extract_topics_from_lessons(pdf_directory="./lessons", cache_file="topics_cache.json"):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    return data
        except Exception:
            pass

    if not os.path.exists(pdf_directory):
        return {}

    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return {}

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    categories_dict = {}
    
    file_data = []
    for f in pdf_files:
        path = os.path.join(pdf_directory, f)
        snippet = get_pdf_snippet(path)
        file_data.append({"filename": f, "snippet": snippet})
    
    prompt = f"""You are a curriculum analyzer. I will provide you with a list of {len(file_data)} PDF filenames and a small snippet of their content.
Your task is to assign each file to a broad Category (e.g., "Web Services", "Data Science", "AI & ML", "Networking") and extract a specific, concise Topic Name (2-5 words) representing that file.
You MUST process EVERY single file. There should be exactly {len(file_data)} topics distributed across your categories.

Data:
{json.dumps(file_data)}

Return ONLY a valid JSON object matching this schema:
{{
  "Category1": ["Topic1", "Topic2"],
  "Category2": ["Topic3"]
}}
Do NOT include markdown formatting (like ```json), just the raw JSON text."""
    
    try:
        response = llm.invoke(prompt)
        content = response.content
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        categories_dict = json.loads(json_str)
        
        if categories_dict:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(categories_dict, f, indent=2)
                
    except Exception as e:
        print(f"Extraction error: {e}")
        return {}

    return categories_dict
