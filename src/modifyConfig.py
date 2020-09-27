import sys
import json

if __name__=='__main__':
    if len(sys.argv) != 3:
        print('usage: python3 modifyConfig.py data.json mode')
        exit(0)

    with open('datasets/config.json', 'r') as f:
        config = json.load(f)

    mode = sys.argv[2]
    ### mode 0 for training 
    ### mode 1 for testing
    if mode == '0':
        trainPath = sys.argv[1]
        config['train'] = trainPath
    elif mode == '1':
        validPath = sys.argv[1]
        config['dev'] = validPath

    print(config)

    with open('datasets/config.json', 'w') as f:
        config = json.dump(config, f)