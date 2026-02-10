import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

my_key = os.getenv("GOOGLE_API_KEY")

if not my_key:
    print("Error: There's no key in this folder")
else:
    print("Linking to google key ... ")
    genai.configure(api_key=my_key)

    # for m in genai.list_models():
    # if 'generateContent' in m.supported_generation_methods:
    #     print(m.name)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("请用鲁迅的语气夸一下我的代码写得好")
    print(response.text)
