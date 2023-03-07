# coding=utf-8

from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse,os


class Flickr30KDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_path_list = list(tqdm(image_dir.iterdir()))
        self.n_images = len(self.image_path_list)

        # self.transform = image_transform

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
#        print(image_path)
        image_id = image_path.stem

        img = cv2.imread(str(image_path))

        return {
            'img_id': image_id,
            'img': img
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--flickrroot', type=str,
                        default='/path/to/input_dir/BBC/xlsum/XLSum_input/individual_img/')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'test', 'val'])
    parser.add_argument('--lang', type=str, default='english')

    args = parser.parse_args()

    SPLIT2DIR = {
        'trainval': 'flickr30k_images',
        'test2017': 'test_2017_flickr_images',
        'test2018': 'test_2018_flickr_images',
    }
    SPLIT2DIR = {
        'train': 'train',
        'test': 'test',
        'val': 'val',
    }

    flickr_dir = Path(args.flickrroot).resolve()
    flickr_img_dir = flickr_dir.joinpath(args.lang).joinpath(SPLIT2DIR[args.split])

    dataset_name = args.lang

    out_dir = flickr_dir.joinpath(args.lang)
    if not out_dir.exists():
        out_dir.mkdir()

    print('Load images from', flickr_img_dir)
    print('# Images:', len(list(flickr_img_dir.iterdir())))

    dataset = Flickr30KDataset(flickr_img_dir)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)
    out_dir = "/path/to/output_dir/BBC/individual_img"
    out_dir = Path(out_dir).resolve()
    fea_path = os.path.join(out_dir, args.lang)
    os.makedirs(fea_path, exist_ok=True)
    output_fname = out_dir.joinpath(args.lang).joinpath(f'{args.split}_boxes{NUM_OBJECTS}.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{args.split}_{(NUM_OBJECTS, DIM)}'

    extract(output_fname, dataloader, desc)
