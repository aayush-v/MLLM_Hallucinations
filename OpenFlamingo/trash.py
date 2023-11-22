import json



with open('all_images_val/num_10.json', 'r') as file:

    data = json.load(file)

with open('num_10_1000.json', 'w') as file:

    json.dump(data, file, indent=2)

#with open('output.json', 'w', indent=2) as file:

 #   json.dump(data, file)
