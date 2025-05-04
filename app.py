from flask import Flask, render_template, request, jsonify
from bot import ChatBot
import sys
import markdown

app = Flask(__name__)

# Initialize the chatbot with the path to the PDF
chatbot = ChatBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getanswer', methods=['POST'])
def get_answer():
    user_input = request.json.get('msg', '')
    answer = chatbot.askQuestion(user_input)
    html = markdown.markdown(answer, extensions=["md_in_html"])
    return jsonify({"answer": html})

if __name__ == '__main__':
    # Set your custom port number here
    app.run(debug=True, host="0.0.0.0", port=8000)