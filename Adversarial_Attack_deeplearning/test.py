import json
output_path = "test.json"
a = [1,2,3,4]
with open(output_path, "w+") as f:
    json.dump(a, f)
    
with open(output_path, "r") as f:
    b = json.load(f)
    print(type(b))