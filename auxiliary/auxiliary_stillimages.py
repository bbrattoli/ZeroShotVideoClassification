import os, numpy as np, cv2, torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.io import imread
from skimage.transform import resize

"""========================================================="""

excluded = ['barndoor', 'thriftshop']
def get_sun():
    directory = '/workplace/SUN397/'
    with open(directory + 'ClassName.txt') as f:
        classes_path = [l[:-1] for l in f.readlines()]

    fnames, labels = [], []
    for cp in classes_path:
        label = cp[3:]
        if label in excluded: continue
        fold = directory+cp
        for fname in os.listdir(fold):
            fnames.append(os.path.join(str(fold), fname))
            labels.append(label)

    classes = np.unique(labels)
    return fnames, labels, classes


"""========================================================="""


class ImageDataset(Dataset):

    def __init__(self, fnames, labels, class_embed, classes, name,
                 clip_len=8, n_clips=1, crop_size=112, is_validation=False):
        self.data = fnames
        self.labels = labels
        self.class_embed = class_embed
        self.class_name = classes
        self.name = name

        self.clip_len = clip_len
        self.n_clips = n_clips

        self.crop_size = crop_size  # 112
        self.is_validation = is_validation

        self.transform = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.RandomResizedCrop((self.crop_size, self.crop_size)),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                               std=[0.22803, 0.22145, 0.216989]),
        ])

        self.crop_transform = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Resize((self.crop_size, self.crop_size)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                               std=[0.22803, 0.22145, 0.216989]),
        ])

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

    def __getitem__(self, idx):
        fname = self.data[idx]
        label = self.label_array[idx]
        try:
            img = imread(fname)
        except KeyboardInterrupt:
            import sys; sys.exit()
        except:
            buffer = np.random.rand(self.n_clips, 3, self.clip_len, 112, 112).astype('float32')
            buffer = torch.from_numpy(buffer)
            return buffer, -1, self.class_embed[0], -1

        if img is None or len(img) == 0:
            buffer = np.random.rand(self.n_clips, 3, self.clip_len, 112, 112).astype('float32')
            buffer = torch.from_numpy(buffer)
            return buffer, -1, self.class_embed[0], -1
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        buffer = self.extract_camera_motion(img)
        return buffer, label, self.class_embed[label], idx

    def __len__(self):
        return len(self.data)

    def extract_camera_motion(self, img):
        m = min(img.shape[:2])
        if m < 172:
            scale = int(172.0 / m)
            img = resize(img, (scale*img.shape[0], scale*img.shape[1]),
                         mode='constant', anti_aliasing=True)
            img = (255*img).astype('uint8')
        elif m > 512:
            scale = 512.0 / m
            new_shape = (int(scale*img.shape[0]), int(scale*img.shape[1]))
            img = resize(img, new_shape, mode='constant', anti_aliasing=True)
            img = (255 * img).astype('uint8')

        if len(img.shape) == 2 or img.shape[2] == 1:
            img = np.repeat(img.reshape([img.shape[0], img.shape[1], 1]), 3, 2)

        if img.shape[2] == 2:
            img = np.concatenate([img[:, :, 0], img[:, :, 1], img[:, :, 1]], 2)

        if img.shape[2] == 4:
            img = img[:, :, :3]

        s = img.shape
        crop = self.crop_size
        N = self.n_clips*self.clip_len

        start = [np.random.randint(0, max(s[i] - crop, 1)) for i in range(2)]
        start_side = np.random.randint(crop, max(min(s[0] - start[0], s[1] - start[1]), crop+1))

        end   = [np.random.randint(0, max(s[i] - crop, 1)) for i in range(2)]
        end_side = np.random.randint(crop, max(min(s[0] - end[0], s[1] - end[1]), crop+1))

        trajectory = [np.linspace(start[0], end[0], N).astype(int),
                      np.linspace(start[1], end[1], N).astype(int),
                      np.linspace(start_side, end_side, N).astype(int)]
        trajectory = np.stack(trajectory).T


        clip = []
        for tj in trajectory:
            im = img[tj[0]:tj[0]+tj[2], tj[1]:tj[1]+tj[2]]
            assert len(img.shape) == 3 and img.shape[2] == 3, str(img.shape)
            im = self.crop_transform(im)
            clip.append(im)
        clip = torch.stack(clip)
        clip = clip.reshape(self.n_clips, self.clip_len, 3, self.crop_size, self.crop_size).transpose(1, 2)
        return clip

    def extract_video(self, img):
        buffer = self.transform(img)
        buffer = torch.repeat_interleave(torch.unsqueeze(buffer, 1), self.clip_len, 1)
        buffer = torch.repeat_interleave(torch.unsqueeze(buffer, 0), self.n_clips, 0)
        return buffer








