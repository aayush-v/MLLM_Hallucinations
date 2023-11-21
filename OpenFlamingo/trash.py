import json



with open('num_10.txt', 'r') as file:

    data = json.load(file)

with open('num_10.json', 'w') as file:

    json.dump(data, file, indent=2)

#with open('output.json', 'w', indent=2) as file:

 #   json.dump(data, file)
