import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import plotly
import plotly.graph_objects as go
from werkzeug.utils import secure_filename
import re

app = Flask(__name__)
load_dotenv()

try:
    # Configure the Gemini API with the key from the .env file
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring Google AI: {e}. Make sure your .env file is set up correctly.")


def enforce_quality_with_ai(text_content: str) -> dict:
    model = genai.GenerativeModel('gemini-2.5-pro-preview-06-05')
    style_guide = "1. Use active voice. 2. Be clear and direct. 3. Maintain a professional, formal tone."
    quality_prompt = f"""
    You are an expert technical editor. Rewrite the following text to strictly adhere to these rules: {style_guide}.
    Do not add any new information. Return ONLY a single, valid JSON object with two keys: "title" (a string) and "paragraphs" (an array of strings).
    ORIGINAL TEXT: {text_content}
    """
    response = model.generate_content(quality_prompt)
    cleaned_response = response.text.strip().lstrip('```json').rstrip('```')
    return json.loads(cleaned_response)

def generate_ai_metadata(text_content: str) -> dict:
    model = genai.GenerativeModel('gemini-1.5-flash')
    summary_prompt = f"Write a single, concise sentence that summarizes this text: {text_content}"
    keywords_prompt = f'Extract the 4 to 6 most important keywords from this text. Return a single, valid JSON array of strings, like ["keyword1", "keyword2"]. TEXT: {text_content}'
    
    summary = model.generate_content(summary_prompt).text.strip()
    keywords_text = model.generate_content(keywords_prompt).text.strip().lstrip('```json').rstrip('```')
    keywords = json.loads(keywords_text)
    
    return {"summary": summary, "keywords": keywords}

def create_dita_xml(data: dict, metadata: dict) -> str:
    """Generates a DITA XML string. (Simplified for backend generation)"""
    title = data.get('title', 'Untitled')
    paragraphs = data.get('paragraphs', [])
    topic_id = title.lower().replace(' ', '_').replace('?', '')
    
    xml = f'<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += f'<!DOCTYPE topic PUBLIC "-//OASIS//DTD DITA Topic//EN" "topic.dtd">\n'
    xml += f'<topic id="{topic_id}">\n'
    xml += f'  <title>{title}</title>\n'
    xml += '  <prolog>\n'
    xml += f'    <shortdesc>{metadata.get("summary", "")}</shortdesc>\n'
    xml += '    <metadata>\n'
    for kw in metadata.get("keywords", []):
        xml += f'      <indexterm>{kw}</indexterm>\n'
    xml += '    </metadata>\n'
    xml += '  </prolog>\n'
    xml += '  <body>\n'
    for p in paragraphs:
        xml += f'    <p>{p}</p>\n'
    xml += '  </body>\n'
    xml += '</topic>'
    return xml


@app.route('/')
def index():
    """Renders the main dashboard page from the templates folder."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_text():
    """API endpoint to process text and return DITA, metadata, and chart."""
    try:
        data = request.get_json()
        raw_text = data.get('text')
        if not raw_text:
            return jsonify({"error": "Text input is missing."}), 400

        improved_data = enforce_quality_with_ai(raw_text)
        improved_text_for_meta = f"{improved_data['title']}\n\n" + "\n".join(improved_data['paragraphs'])
        
        metadata = generate_ai_metadata(improved_text_for_meta)
        
        final_dita = create_dita_xml(improved_data, metadata)

        improved_full_text = ' '.join(improved_data['paragraphs'])
        keyword_freq = {kw: improved_full_text.lower().split().count(kw.lower()) for kw in metadata['keywords']}
        fig = go.Figure([go.Bar(x=list(keyword_freq.keys()), y=list(keyword_freq.values()), marker_color='#3b82f6')])
        fig.update_layout(
            title_text='AI-Generated Keyword Frequency in Improved Content',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb'),
            xaxis=dict(gridcolor='#4b5563', title_text='Keywords'),
            yaxis=dict(gridcolor='#4b5563', title_text='Frequency')
        )
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({
            "dita_xml": final_dita,
            "summary": metadata["summary"],
            "keywords": metadata["keywords"],
            "chart_json": chart_json,
            "keyword_freq": keyword_freq
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

ALLOWED_EXTENSIONS = {'.txt', '.md'}
def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process_file', methods=['POST'])
def process_file():
    """API endpoint to process uploaded .txt or .md file and return DITA, metadata, and chart."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported file type. Please upload a .txt or .md file."}), 400
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        file_content = file.read().decode('utf-8')
        if ext == '.txt':
            lines = [line for line in file_content.splitlines() if line.strip()]
            if not lines:
                return jsonify({"error": "The .txt file is empty or contains only whitespace."}), 400
            title = lines[0]
            paragraphs = lines[1:]
        elif ext == '.md':
            import markdown_it
            md = markdown_it.MarkdownIt()
            tokens = md.parse(file_content)
            title = None
            paragraphs = []
            for i, token in enumerate(tokens):
                if token.type == 'heading_open' and token.tag == 'h1':
                    title_token = tokens[i+1]
                    if title_token.type == 'inline':
                        title = title_token.content
                        break
            if not title:
                return jsonify({"error": "Markdown file must contain a level 1 heading (e.g., # My Title)."}), 400
            for i, token in enumerate(tokens):
                if token.type == 'paragraph_open':
                    p_content_token = tokens[i + 1]
                    if p_content_token.type == 'inline':
                        paragraphs.append(p_content_token.content)
        else:
            return jsonify({"error": "Unsupported file type."}), 400
        raw_text = f"{title}\n\n" + "\n".join(paragraphs)
        improved_data = enforce_quality_with_ai(raw_text)
        improved_text_for_meta = f"{improved_data['title']}\n\n" + "\n".join(improved_data['paragraphs'])
        metadata = generate_ai_metadata(improved_text_for_meta)
        final_dita = create_dita_xml(improved_data, metadata)
        improved_full_text = ' '.join(improved_data['paragraphs'])
        keyword_freq = {kw: improved_full_text.lower().split().count(kw.lower()) for kw in metadata['keywords']}
        fig = go.Figure([go.Bar(x=list(keyword_freq.keys()), y=list(keyword_freq.values()), marker_color='#3b82f6')])
        fig.update_layout(
            title_text='AI-Generated Keyword Frequency in Improved Content',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb'),
            xaxis=dict(gridcolor='#4b5563', title_text='Keywords'),
            yaxis=dict(gridcolor='#4b5563', title_text='Frequency')
        )
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({
            "dita_xml": final_dita,
            "summary": metadata["summary"],
            "keywords": metadata["keywords"],
            "chart_json": chart_json,
            "keyword_freq": keyword_freq
        })
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
