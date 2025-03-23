import yaml
with open("example.yml") as f:
    result = yaml.load(f, Loader=yaml.BaseLoader)
    print(result)
    type(result)