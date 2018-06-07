import yaml
import os


if __name__ == "__main__":
    path = "MedleyDB/Audio/"
    yamlfiles = []
    for (root, dirs, files) in os.walk(path):
        yamlfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".yaml")])

    for y in yamlfiles:
        with open(y, 'r') as yf:
            whole = yaml.load(yf)
            stems = whole['stems']
            for st in stems.values():
                stem_file = st["filename"]
                stem_instrument = st["instrument"]

                print(stem_file, stem_instrument)
