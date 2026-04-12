import json

with open('metadata/contestant_manifest.json') as f:
    data = json.load(f)

print('Train sequences:', len(data['train']))
print('Test sequences:', len(data['public_lb']))
print()
print('First 3 train sequences:')
for i, (k, v) in enumerate(data['train'].items()):
    if i >= 3:
        break
    print(f'  {k} -> {v["n_frames"]} frames')