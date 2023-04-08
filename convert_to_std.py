import os
import os.path as osp
import argparse

import pandas as pd
from options.options import set_inference_options, Database
from options.denoisors import Denoisor
from tqdm import tqdm

from evtool.dtype import Event, Frame, Size

from evtool.dvs import DvsFile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment of EMLB benchmark')
    parser.add_argument('-i', '--input_path', type=str, default='/media/kuga/瓜果山/results/evzoom/', help='path to load dataset')
    parser.add_argument('-o', '--output_path', type=str,
                        default='/media/kuga/瓜果山/results/final/', help='path to output dataset')
    parser.add_argument('-d', '--denoisor', type=str, default='evzoom', help='choose denoisors')

    parser.add_argument('-w', '--save_file', action='store_false', help="save denoising result")
    # parser.add_argument('-s', '--calc_esr_score', action='store_false', help="ecaluate esr performance")
    args = set_inference_options(parser)
    
    for dataset in Database(args):
        table = dataset.table()
        pbar = tqdm(dataset.seqs(), leave=False)
        
        model = Denoisor(args.denoisor)
        for seq in pbar:
            # print info
            info = (args.denoisor.center(10, " "), seq.name.rjust(15, " "), dataset.name.ljust(15, " "))
            pbar.set_description("Implementing %s to inference on %s / %s" % info)

            # load noisy event data and perform inference
            data = pd.read_pickle(seq.path)
            data['events'] = Event(data['events'])

            if 'frames' in data.keys():
                data.pop('frames')

            if dataset.name == 'RGB DAVIS':
                data['size'] = Size((180, 240))
            else:
                data['size'] = Size((260, 346))

            # save inference result
            if args.save_file:
                output_file = f"{args.output_path}/{args.denoisor}/{dataset.name}/{seq.category}/{seq.name}.{args.output_file_type}"
                output_dir, _ = osp.split(output_file)
                if not osp.exists(output_dir): os.makedirs(output_dir)

                DvsFile.save(data, output_file)
