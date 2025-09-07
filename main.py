from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    query_text = data.get("queryResult", {}).get("queryText", "")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a polite, professional AI assistant. Always respond helpfully and never use foul language."},
                {"role": "user", "content": query_text}
            ],
            max_tokens=200
        )
        ai_response = response.choices[0].message.content.strip()
    except Exception as e:
        ai_response = f"Error: {str(e)}"

    return jsonify({"fulfillmentText": ai_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
