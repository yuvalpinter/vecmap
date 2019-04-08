import pickle
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Check sense mapping against list of known pairs')
    parser.add_argument('gold', help='Gold pairings')  # 'data/synsets/gold_it_pairings_v3b_20k.pkl'
    parser.add_argument('mapping_files', nargs='+', help='Predicted files to score')
    parser.add_argument('--threshold', type=float, default=0, help='Minimum score to be considered as a mapping')
    ### TODO add synset names file and word list file for human-readable results
    args = parser.parse_args()
    
    # pickle file string pairs
    gold_it_rec = pickle.load(open(args.gold,'rb'))
    print(f'number of pairs loaded: {len(gold_it_rec)}')

    for map_f_name in args.mapping_files:
        print(f'now evaluating mapping file: {map_f_name}')
        with open(map_f_name, 'rb') as map_f:
            pred = pickle.load(map_f) #.get()
            corr = []
            for p in tqdm(gold_it_rec):
                if pred[p] > args.threshold:
                    corr.append(p)
            #corr = [p for p in gold_it_rec if pred[p] > args.threshold]
            print(f'{len(corr)} correct predictions:')
            print(corr)
            ### TODO there's a core dump here (or sooner if '.get()' is called). No idea why.

if __name__ == '__main__':
    main()
