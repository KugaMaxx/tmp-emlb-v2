import os
import os.path as osp
import argparse

from options.options import set_inference_options, Database
from options.denoisors import Denoisor
from tqdm import tqdm

from evtool.dvs import DvsFile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment of EMLB benchmark')
    parser.add_argument('-i', '--input_path', type=str, default='/media/kuga/瓜果山/Datasets/A/', help='path to load dataset')
    parser.add_argument('-o', '--output_path', type=str, default='./results', help='path to output dataset')
    parser.add_argument('-d', '--denoisor', type=str, default='multiLayer_perceptron_filter', help='choose denoisors')

    parser.add_argument('-w', '--save_file', action='store_false', help="save denoising result")
    # parser.add_argument('-s', '--calc_esr_score', action='store_false', help="ecaluate esr performance")
    args = set_inference_options(parser)
    
    for dataset in Database(args):
        table = dataset.table()
        pbar = tqdm(dataset.seqs(), leave=False)
        
        model = Denoisor(args)
        for seq in pbar:
            # print info
            info = (model.name.center(10, " "), seq.name.rjust(15, " "), dataset.name.ljust(15, " "))
            pbar.set_description("Implementing %s to inference on %s / %s" % info)

            # load noisy event data and perform inference
            data = DvsFile.load(seq.path)

            # filter all hot-pixels
            if args.excl_hotpixel:
                idx = data['events'].hotpixel(data['size'], thres=1000)
                data['events'] = data['events'][idx]

            data = model.run(data)

            # save inference result
            if args.save_file:
                output_file = f"{args.output_path}/{dataset.name}/{seq.category}/{seq.name}.{args.output_file_type}"
                output_dir, _ = osp.split(output_file)
                if not osp.exists(output_dir): os.makedirs(output_dir)
                
                DvsFile.save(data, output_file)

        #     # calculate ESR
        #     score = calc_event_structural_ratio(ev, size)            
        #     table.update(seq, model, score)

        # table.show(mode="summary")