from open_flamingo import create_model_and_transforms
import json
from tqdm import tqdm
from PIL import Image
import requests
from huggingface_hub import hf_hub_download
import torch
import sys
import os 

print(sys.argv[1])

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-7b",
    tokenizer_path="anas-awadalla/mpt-7b",
    cross_attn_every_n_layers=4
)


checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(device)

imagesPath = "/scratch/nmachav1/CLEVR_v1.0/images/val/"
quesDirectory = f"/scratch/nmachav1/MLLM_Hallucinations/CLEVR_v1/datasetSplits/num{sys.argv[1]}/"

# Open the JSON file and process each line
outputFileDirectory = "/scratch/nmachav1/MLLM_Hallucinations/CLEVR_v1/answers/val/language_augmentation/OpenFlam"

for filename in os.listdir(quesDirectory):
    if filename.endswith('.json'):
        file_path = os.path.join(quesDirectory, filename)
        print(filename)
        with open(file_path, 'r') as jsonFile, open(f'{outputFileDirectory}{filename[:-5]}_results.json', 'a') as resultFile:
            res = []
            for line in jsonFile.readlines()[:100]:
                entry = json.loads(line)
                img_name = entry["image_name"]
                image_ques = entry["questions"]
        
                ansList = []  # List to store results for each question
        
                for q in image_ques:
                    img = Image.open(imagesPath+img_name).convert("RGB")
                    vision_x = [image_processor(img).unsqueeze(0)]
                    vision_x = torch.cat(vision_x, dim=0).to(device)
                    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
                    
                    ques_for_model = "Question: " + q["question"] + " Answer: "
                    tokenizer.padding_side = "left"  # For generation, padding tokens should be on the left
                    lang_x = tokenizer(
                        ["<image>" + ques_for_model],
                        return_tensors="pt",
                    ).to(device)
            
                    generated_text = model.generate(
                        vision_x=vision_x,
                        lang_x=lang_x["input_ids"],
                        attention_mask=lang_x["attention_mask"],
                        max_new_tokens=50,  # You can adjust this to control the length of the generated text
                        num_beams=6,
                    )
            
                    model_output = tokenizer.decode(generated_text[0])
                   
                    answer_index = model_output.split("Answer:")
            
                    if "<|" in answer_index[-1]:
                        answer = answer_index[-1].split("<|")[0].strip()
                    else:
                        answer = answer_index[-1].strip()
                    print(answer)
        
                    # Append results for each question to the file
                    res.append({
                        "Image ID":img_name,
                        "Question": q["question"],
                        "Ground truth": q["answer"],
                        "Model generated answer": generated_text
                    })
                    # json.dump(res, resultFile)
        
                # Append results for each image to the file
            json.dump(res, resultFile)
                #resultFile.write('\n')  # Add a newline for better readability
