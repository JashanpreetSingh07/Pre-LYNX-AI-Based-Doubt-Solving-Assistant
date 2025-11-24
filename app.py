import os
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import requests
from bs4 import BeautifulSoup
import math

load_dotenv()

app = Flask(__name__)

try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini API Configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

def extract_youtube_video_id(url):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
            except NoTranscriptFound:
                 return None, "No English transcript found for this video."

        full_transcript_data = transcript.fetch()
        return full_transcript_data, None
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None, f"Could not fetch transcript: {str(e)}"

def get_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        main_content = soup.find('article') or soup.find('main') or soup.find(role='main')
        if not main_content:
            potential_containers = soup.find_all('div')
            if potential_containers:
                main_content = max(potential_containers, key=lambda tag: len(tag.find_all('p', recursive=False)))
            else:
                main_content = soup.body

        if not main_content:
             return None, "Could not identify main content area."

        paragraphs = main_content.find_all('p')
        if not paragraphs:
             text = main_content.get_text(separator=' ', strip=True)
             text = re.sub(r'\s+', ' ', text)
             return text, None
        else:
            article_text = "\n".join([p.get_text(strip=True) for p in paragraphs])
            return article_text, None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching article URL {url}: {e}")
        return None, f"Could not fetch URL: {e}"
    except Exception as e:
        print(f"Error parsing article {url}: {e}")
        return None, f"Could not parse article content: {e}"

def parse_timestamp(ts_str):
    parts = list(map(int, ts_str.split(':')))
    seconds = 0
    if len(parts) == 3:
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        seconds = parts[0] * 60 + parts[1]
    elif len(parts) == 1:
        seconds = parts[0]
    else:
        raise ValueError("Invalid timestamp format")
    return seconds

def find_transcript_context(transcript_data, target_seconds, window=30):
    context_parts = []
    start_time = max(0, target_seconds - window)
    end_time = target_seconds + window

    relevant_entries = []
    if transcript_data:
        full_text_for_prompt = " ".join([entry.text for entry in transcript_data])

        for entry in transcript_data:
            entry_start = entry.start
            entry_end = entry.start + entry.duration

            if entry_start < end_time and entry_end > start_time:
                relevant_entries.append(entry.text)

        if not relevant_entries:
            closest_entry = min(transcript_data, key=lambda x: abs(x.start - target_seconds))
            return closest_entry.text, full_text_for_prompt
        else:
            return " ".join(relevant_entries), full_text_for_prompt
    else:
        return "", ""

def get_gemini_response(context, query, content_type, source_identifier):
    if not model:
        return "Error: AI Model not configured."

    prompt = ""
    if content_type == 'youtube':
        context_snippet, full_transcript_text = context
        prompt = f"""You are analyzing a YouTube video transcript.
        Here is a relevant snippet from the transcript around a specific timestamp:
        "{context_snippet}"

        The user has the following query related to this video and the provided snippet:
        "{query}"

        Based on the context, please answer the user's query concisely and accurately.
        If the query requires information beyond the provided snippet, you can refer to the broader context of the video (though the full transcript is not directly available).
        """
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating AI response: {e}"

    elif content_type == 'article':
        prompt = f"""You are analyzing the following article text:
        "{context}"

        The user has the following query related to this article:
        "{query}"

        Based on the article text, please answer the user's query concisely and accurately.
        """
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating AI response: {e}"

    return "Error: Content type not supported or prompt construction failed."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_query():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    content_type = data.get('contentType')
    url = data.get('url')
    query = data.get('query')
    identifier = data.get('identifier')

    if not all([content_type, url, query, identifier]):
         return jsonify({"error": "Missing required fields (contentType, url, query, identifier)"}), 400

    ai_response = "Sorry, an error occurred before generating the AI response."

    try:
        if content_type == 'youtube':
            video_id = extract_youtube_video_id(url)
            if not video_id:
                return jsonify({"error": "Invalid YouTube URL or could not extract video ID."}), 400

            transcript_data, error = get_youtube_transcript(video_id)
            if error:
                return jsonify({"error": error}), 400
            if not transcript_data:
                 return jsonify({"error": "Failed to retrieve transcript data."}), 500

            try:
                target_seconds = parse_timestamp(identifier)
            except ValueError:
                 return jsonify({"error": "Invalid timestamp format. Use HH:MM:SS or MM:SS."}), 400

            context_snippet, full_transcript_text = find_transcript_context(transcript_data, target_seconds)
            if not context_snippet:
                 return jsonify({"error": "Could not find transcript content near the specified timestamp."}), 404

            ai_response = get_gemini_response((context_snippet, full_transcript_text), query, 'youtube', identifier)

        elif content_type == 'article':
            article_text, error = get_article_text(url)
            if error:
                return jsonify({"error": error}), 400
            if not article_text:
                return jsonify({"error": "Failed to extract text from the article."}), 500

            ai_response = get_gemini_response(article_text, query, 'article', identifier)

        else:
            return jsonify({"error": "Invalid content type specified."}), 400

        return jsonify({"answer": ai_response})

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
