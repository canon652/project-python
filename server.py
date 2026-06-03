import cv2
import re
from flask import Flask, request, send_file
from numpy import uint8, frombuffer
from flask_cors import CORS
from base64 import b64encode
from json import loads
from pprint import pprint
import pytesseract
import torch
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


app = Flask(__name__)
CORS(app)


def names(text):
    messages = [
        {"role": "system", "content": "Ты ассистент. выведи построчно все полные имена. Ответ должен содержать только ФИО, разделенные пробелом. Не выводи должности, предлоги, цифры и пояснения."},
        {"role": "user", "content": f"Вот текст:\n{text}"}
    ]
    
    # Применяем chat template и преобразуем в тензоры
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Токенизируем вручную
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    
    # Генерируем ответ
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=500,
            do_sample=False,
            temperature=None
        )
    
    # Декодируем результат
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлекаем только часть после assistant
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return response.replace('\n', ' ')


def gen(bytecode, settings):
    img = cv2.imdecode(frombuffer(bytecode, uint8), cv2.IMREAD_ANYCOLOR)
    data = pytesseract.image_to_data(img, lang='rus+eng', output_type=pytesseract.Output.DICT)
    parsed_data = {}
    # СНИЛС (формат: XXX-XXX-XXX XX или 11 цифр подряд)
    snils_pattern = r'^\d{3}[- ]?\d{3}[- ]?\d{3}[- ]?\d{2}$'

    # ИНН (10 или 12 цифр)
    inn_pattern = r'^\d{10}$|^\d{12}$'

    # Серия и номер паспорта РФ (10 цифр, возможны разделители)
    passport_seria_pattern = r'\d{4}'
    passport_number_pattern = r'\d{6}'

    # Номер телефона (Россия, мобильные форматы)
    phone_pattern = r'^(\+7|8)[ -]?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}$'

    # Полис ОМС (16 цифр, 6+8 цифр, 4 группы по 4 цифры)
    oms_policy_pattern = r'^(\d{16}|\d{6}[ -]\d{8}|(\d{4}[ -]?){3}\d{4})$'

    for i in range(len(data['text'])):
        sim = data['text'][i]
        if not sim:
            continue
        if sim in parsed_data.keys():
            parsed_data[sim].append(((data['top'][i], data['left'][i]), (data['width'][i], data['height'][i])))
        else:
            parsed_data[sim] = [((data['top'][i], data['left'][i]), (data['width'][i], data['height'][i]))]

    fios = ''
    if 'fio' in settings:
        fios = names(' '.join([i for i in parsed_data.keys()])).split()

    pprint(parsed_data)
    print(fios)

    for i in parsed_data.keys():
        if (('phone' in settings and re.search(phone_pattern, i)) or
                ('email' in settings and i.count('@')) or
                (i in fios) or
                ('inn' in settings and re.search(inn_pattern, i)) or
                ('snils' in settings and re.search(snils_pattern, i)) or
                ('oms' in settings and re.search(oms_policy_pattern, i)) or
                ('passport' in settings and re.search(passport_seria_pattern, i)) or
                ('passport' in settings and re.search(passport_number_pattern, i))
        ):
            print(i in fios)
            for (y, x), (w, h) in parsed_data[i]:
                poi = img[y:y+h, x:x+w]
                blur = cv2.GaussianBlur(poi, (51, 51), 0)
                img[y:y + h, x:x + w] = blur
    return cv2.imencode('.png', img)[1]


@app.route('/', methods=['POST'])
def main():
    settings = loads(request.form['documentJson'])
    settings = [i for i in settings.keys() if settings[i]]
    print(settings)
    data = gen(request.files['file'].read(), settings)
    return b64encode(data), 200


@app.route('/sogl')
def lol():
    return send_file('Пользовательское соглашение.pdf')


if __name__ == "__main__":
    print('start init model')
    from transformers import Gemma3ForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('./gemma/')
    model = Gemma3ForCausalLM.from_pretrained('./gemma/').eval()

    # import intel_extension_for_pytorch as ipex

    # model = ipex.llm.optimize(model)

    print('model has been initialized')

    app.run(port=8080)
