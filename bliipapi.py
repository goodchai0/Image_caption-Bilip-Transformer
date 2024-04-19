from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
from flask_cors import CORS  # Import CORS from flask_cors module
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

device = "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
translator = Translator()

@app.route('/caption_image', methods=['POST'])
def caption_image():
    # Get image data and reference sentence from request
    img_base64 = request.json.get('img_base64')
    ref_sentence = request.json.get('ref_sentence')

    # Decode base64 image data
    img_data = base64.b64decode(img_base64)

    # Open and process image
    raw_image = Image.open(BytesIO(img_data)).convert('RGB')

    # Conditional image captioning
    # text = "a photography of"
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Translate caption to Bengali using Google Translate
    translated_caption = translator.translate(caption, dest='bn').text

    # BLEU score calculation
    reference = [ref_sentence.split()]
    candidate = translated_caption.split()

    # Simple BLEU score calculation (without smoothing)
    matching_ngrams = sum(min(reference[0].count(word), candidate.count(word)) for word in set(candidate))
    bleu_score = matching_ngrams / len(candidate)

    return jsonify({
        'translated_caption': translated_caption,
        'bleu_score': bleu_score
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
