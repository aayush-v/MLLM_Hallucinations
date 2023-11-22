from open_flamingo import create_model_and_transforms
import json
from tqdm import tqdm
from PIL import Image
import requests
from huggingface_hub import hf_hub_download
import torch
import sys



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

print(f"Using device: {device}")
imagesPath = "/scratch/nmachav1/CLEVR_v1.0/images/val/"
quesPath = "/scratch/nmachav1/CLEVR_v1.0/questions/CLEVR_val_questions.json"

jsonFile = open(quesPath, 'r')
questions  = json.load(jsonFile)

i = sys.argv[1]
img_file = open(f"/scratch/nmachav1/CLEVR_v1.0/datasetSplits/val_images_objectsNum{i}.txt", "r") 
img_names = img_file.readlines()

ansFile = open(f"/scratch/nmachav1/MLLM_Hallucinations/OpenFlamingo/all_images_val/num_{i}.json", "a")
object = []
for img_name in tqdm(img_names[:1000]):
    image_ques = []
    image_ans = []
    count = 0
    img = Image.open(imagesPath+img_name[:-1]).convert("RGB")
    vision_x = [image_processor(img).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0).to(device)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    for q in questions["questions"]:
        if q["image_filename"] == img_name[:-1]:
            image_ques.append(q["question"])
            image_ans.append(q["answer"])
    for iqs in image_ques:
        ques_for_model = "Question: " + iqs + " Answer: "
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
        object.append({'image_id': img_name,
                      'Question': iqs,
                      'Ground truth': image_ans[count],
                      'Model generated answer': answer
        })

        count+=1
json.dump(object, ansFile)
ansFile.close()