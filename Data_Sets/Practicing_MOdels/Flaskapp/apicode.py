from flask import Flask, request, jsonify
import os
from happytransformer import HappyGeneration
from transformers import GPTNeoForCausalLM, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
model = AutoModelForCausalLM.from_pretrained("./uo")
tokenizer =AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json(force=True)
    text = data['text']
    # Adding start
    text = "<|startoftext|>"+text
    encoded_input = tokenizer(text, return_tensors="pt").input_ids
    generated_text = model.generate(encoded_input, 
                    do_sample=True, 
                    top_k=50,
                    max_length=300, 
                    top_p=0.95, 
                    temperature=0.7,
                    num_return_sequences=1)
    return jsonify(format(tokenizer.decode(generated_text[0],skip_special_tokens=True)))

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8080)
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
