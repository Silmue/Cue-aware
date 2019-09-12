from dircadrn import DIRCADRN
import yaml
import argparse

parser = argparse.ArgumentParser('description: DICADRN')
parser.add_argument('--config', type=str, default='../config/default.yaml')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.load(f)

learner = DIRCADRN(config)
learner.train(debug=True)
