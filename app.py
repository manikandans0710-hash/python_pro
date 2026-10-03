import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai
from google.genai import types
from chatbot_config import CHATBOT_TITLE, SYSTEM_PROMPT, MODEL_NAME

load_dotenv()

app = Flask(__name__)
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please enter a study question."}), 400

    if client is None:
        return jsonify({"error": "Gemini API key is not configured."}), 500

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.35,
                max_output_tokens=1200,
            ),
        )
        answer = (response.text or "").strip()
        return jsonify({
            "answer": answer or "I could not generate a response. Please try again.",
            "chatbot": CHATBOT_TITLE,
        })
    except Exception as exc:
        return jsonify({"error": f"Gemini API error: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
