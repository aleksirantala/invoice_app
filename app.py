from flask import Flask, render_template, request, jsonify
import os
import base64
import requests
import re

app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload.jpg'))
        return jsonify({"message": "File uploaded successfully"})

@app.route('/analyze', methods=['POST'])
def analyze_image():
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'upload.jpg')
    if not os.path.exists(image_path):
        return jsonify({"error": "No image found to analyze"}), 400

    # Encode the image as base64
    base64_image = encode_image(image_path)

    # Extract individual pieces of information with multiple API calls
    sender_name = extract_info(base64_image, "sender's name")
    invoice_number = extract_info(base64_image, "invoice number")
    due_date = extract_info(base64_image, "the due date for payment")
    total_sum = extract_info(base64_image, "the total sum on the invoice")
    category = extract_info(base64_image, "the category of the products")

    # Clean each extracted result using another LLM call
    sender_name_cleaned = clean_with_llm(sender_name, "sender's name")
    invoice_number_cleaned = clean_with_llm(invoice_number, "invoice number")
    due_date_cleaned = clean_with_llm(due_date, "due date for payment")
    total_sum_cleaned = clean_with_llm(total_sum, "total sum on the invoice")
    category_cleaned = clean_with_llm(category, "category of products")

    result = {
        "sender_name": sender_name_cleaned if sender_name_cleaned else "Not found",
        "invoice_number": invoice_number_cleaned if invoice_number_cleaned else "Not found",
        "due_date": due_date_cleaned if due_date_cleaned else "Not found",
        "total_sum": total_sum_cleaned if total_sum_cleaned else "Not found",
        "category": category_cleaned if category_cleaned else "Not found"
    }

    return jsonify(result)

def extract_info(base64_image, target_info):
    """Makes an API call to OpenAI and returns the extracted information."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    prompt = f"Extract {target_info} from the following image."

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 150
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        try:
            content = response.json()['choices'][0]['message']['content']
            return content.strip()
        except KeyError:
            return None
    else:
        return None

def clean_with_llm(text, target_info):
    """Uses another LLM to clean up the extracted information, removing unwanted phrases."""
    if text is None:
        return None
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    if target_info == "category of products":
        # For categories, ask the LLM to select only one main or upper-level category
        prompt = f"Return only the single, most relevant or upper-level category of products from this text. Do not return multiple categories, only one: '{text}'"
    else:
        # For other fields, clean as before
        prompt = f"Return only the {target_info} from this text and remove unnecessary details. For example, remove periods and all texts other than the value we are looking for: '{text}'"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 50
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        try:
            content = response.json()['choices'][0]['message']['content'].strip()

            if target_info == "category of products":
                # Post-process to ensure only the first category is returned
                # Split the content by common delimiters like commas, colons, or semicolons
                categories = re.split(r'[,:;]', content)
                if categories:
                    return categories[0].strip()  # Return only the first category
                else:
                    return content.strip()  # If splitting fails, return the content as-is
            else:
                return content  # For other fields, return the cleaned result as-is

        except KeyError:
            return text  # If something goes wrong, return the original text
    return text

def encode_image(image_path):
    """Converts an image file into a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)
