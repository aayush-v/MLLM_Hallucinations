import torch
import json
from PIL import Image
import requests
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import sys

IMAGES_PATH = "/scratch/averma90/CLEVR_v1.0/images/val/"
QUES_PATH = "/home/averma90/CSE576/github/CLEVR_v1.0/questions/CLEVR_val_questions.json"

i = sys.argv[1]
json_file = open(QUES_PATH, 'r')
questions  = json.load(json_file)['questions']

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(device)



img_file = open(f"/home/averma90/CSE576/github/MLLM_Hallucinations/CLEVR_v1/datasetSplits/val_images_objectsNum{i}.txt", "r") 
img_names = img_file.readlines()

ansFile = open(f"/home/averma90/CSE576/github/MLLM_Hallucinations/CLEVR_v1/answers/val/num_objects_based_difficulty_vanilla/num_{i}.json", "a")
object = []

for img_name in tqdm(img_names[:1000]):
    image_ques = []
    image_ans = []
    count = 0
    img = Image.open(IMAGES_PATH+img_name[:-1]).convert("RGB")
    for q in questions:
        if q["image_filename"] == img_name[:-1]:
            image_ques.append(q["question"])
            image_ans.append(q["answer"])
    for iqs in image_ques:
        prompt=iqs
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=1.0,
            repetition_penalty=1.0,
            length_penalty=1,
            top_k=50,
            temperature=1e-05
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        # print(generated_text)
        object.append({'Image Id': img_name,
                      'Question': prompt,
                      'Ground truth': image_ans[count],
                      'Model generated answer': generated_text
        })
        # ansFile.write(f"{prompt} Answer: {image_ans[count]}. Output: {generated_text}\n")
        
        # ansFile.write(prompt +" " + image_ans[count] + ": " + generated_text+"\n")
        count+=1

json.dump(object, ansFile)
ansFile.close()