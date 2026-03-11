import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps


unseen_classes = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]

class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig

        self.all_categories = os.listdir(os.path.join(self.opts.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')
            
        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen_classes))
            else:
                self.all_categories = unseen_classes
        self.all_categories = sorted(self.all_categories)

        self.all_sketches_path = []
        self.photo_path_by_category = {}
        self.photo_ids_by_category = {}

        for category in self.all_categories:
            photo_paths = []
            for pattern in ('*.jpg', '*.jpeg', '*.png'):
                photo_paths.extend(glob.glob(os.path.join(self.opts.data_dir, 'photo', category, pattern)))

            photo_path_by_id = {
                self._instance_id_from_photo(path): path for path in sorted(photo_paths)
            }
            self.photo_path_by_category[category] = photo_path_by_id
            self.photo_ids_by_category[category] = sorted(photo_path_by_id.keys())

            for sketch_path in sorted(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png'))):
                instance_id = self._instance_id_from_sketch(os.path.basename(sketch_path))
                if instance_id in photo_path_by_id:
                    self.all_sketches_path.append(sketch_path)

    def __len__(self):
        return len(self.all_sketches_path)

    @staticmethod
    def _instance_id_from_sketch(filename):
        return os.path.splitext(filename)[0].rsplit('-', 1)[0]

    @staticmethod
    def _instance_id_from_photo(filepath):
        return os.path.splitext(os.path.basename(filepath))[0]

    def _sample_negative_path(self, category, positive_instance_id):
        negative_ids = [
            photo_id for photo_id in self.photo_ids_by_category[category]
            if photo_id != positive_instance_id
        ]
        if not negative_ids:
            return self.photo_path_by_category[category][positive_instance_id]
        negative_id = np.random.choice(negative_ids)
        return self.photo_path_by_category[category][negative_id]
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)
        instance_id = self._instance_id_from_sketch(filename)

        sk_path  = filepath
        img_path = self.photo_path_by_category[category][instance_id]
        neg_path = self._sample_negative_path(category, instance_id)

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, instance_id,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, instance_id)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode='train', return_orig=True)
    dataset_val = Sketchy(opts, dataset_transforms, mode='val', used_cat=dataset_train.all_categories, return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, category, instance_id,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1
