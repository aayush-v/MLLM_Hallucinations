import json
import os
import glob

def writeFile(path):
    data_object = []
    with open(path, 'r') as file:
        for idx, line in enumerate(file):
            qa = line.strip()
            print(qa)
            [question, rest] = qa.split('Answer') 
            [gt, output] = rest.split('Output')

            new_item = {
                "Question": question[:-1],
                "Ground truth": gt[2:-1],
                "Model generated answer": output[2:] + '.' if output[-1] != '.' else output[2:]
            }

            data_object.append(new_item)
            
            # if idx>3:
            #     break

    print(path)
    # print(path[:-4])
    with open(f"{path[:-4]}.json", 'w') as jsonFile:
        json.dump(data_object, jsonFile)



if __name__ == "__main__":
    path = '/home/averma90/CSE576/github/MLLM_Hallucinations/CLEVR_v1/answers/val'
    files = glob.glob(os.path.join(path, '*.txt'), recursive=True)

    for file in files:
        writeFile(file)
