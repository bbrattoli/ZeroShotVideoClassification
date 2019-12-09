import os, numpy as np
from time import time

import cv2, torch
from torch.utils.data import Dataset
from auxiliary.transforms import get_transform
from scipy.spatial.distance import cdist


def get_ucf101():
    folder = '/workplace/UCF101/videos/'
    fnames, labels = [], []
    for label in sorted(os.listdir(str(folder))):
        for fname in os.listdir(os.path.join(str(folder), label)):
            fnames.append(os.path.join(str(folder), label, fname))
            labels.append(label)

    classes = np.unique(labels)
    return fnames, labels, classes


def get_hmdb():
    folder = '/workplace/HMDB51/videos/'
    fnames, labels = [], []
    for label in sorted(os.listdir(str(folder))):
        dir = os.path.join(str(folder), label)
        if not os.path.isdir(dir): continue
        for fname in sorted(os.listdir(dir)):
            if fname[-4:] != '.avi':
                continue
            fnames.append(os.path.join(str(folder), label, fname))
            labels.append(label.replace('_', ' '))

    fnames, labels = np.array(fnames), np.array(labels)
    classes = np.unique(labels)
    return fnames, labels, classes


'''
This function is ad-hoc to my personal format of kinetics.
You need to adjust it to your data format.
'''
def get_kinetics(dataset=''):
    sourcepath = '/workplace/kinetics/'
    n_classes = '700' if '700' in dataset else '400'
    with open('./assets/kinetics%s.txt' % n_classes, 'r') as f:
        data = [r[:-1].split(',') for r in f.readlines()]

    fnames, labels = [], []
    for x in data:
        if len(x) < 2: continue
        fnames.append(sourcepath + x[0])
        labels.append(x[1][1:])

    classes = sorted(np.unique(labels).tolist())
    return fnames, labels, classes


"""========================================================="""


def filter_samples(opt, fnames, labels, classes):
    """
    Select a subset of classes. Mostly for faster debugging.
    """
    fnames, labels = np.array(fnames), np.array(labels)
    if opt.train_samples != -1:
        sel = np.linspace(0, len(fnames)-1, min(opt.train_samples, len(fnames))).astype(int)
        fnames, labels = fnames[sel], labels[sel]
    return np.array(fnames), np.array(labels), np.array(classes)


def filter_classes(opt, fnames, labels, classes, class_embedding):
    """
    Select a subset of classes. Mostly for faster debugging.
    """
    sel = np.ones(len(classes)) == 1
    if opt.class_total > 0:
        sel = np.linspace(0, len(classes)-1, opt.class_total).astype(int)

    classes = np.array(classes)[sel].tolist()
    class_embedding = class_embedding[sel]
    fnames = [f for i, f in enumerate(fnames) if labels[i] in classes]
    labels = [l for l in labels if l in classes]
    return np.array(fnames), np.array(labels), np.array(classes), class_embedding


def filter_overlapping_classes(fnames, labels, classes, class_embedding, ucf_class_embedding, class_overlap):
    class_distances = cdist(class_embedding, ucf_class_embedding, 'cosine').min(1)
    sel = class_distances >= class_overlap

    classes = np.array(classes)[sel].tolist()
    class_embedding = class_embedding[sel]

    fnames = [f for i, f in enumerate(fnames) if labels[i] in classes]
    labels = [l for l in labels if l in classes]

    return fnames, labels, classes, class_embedding


"""========================================================="""


def load_clips_tsn(fname, clip_len=16, n_clips=1, is_validation=False):
    if not os.path.exists(fname):
        print('Missing: '+fname)
        return []
    # initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count == 0 or frame_width == 0 or frame_height == 0:
        print('loading error, switching video ...')
        print(fname)
        return []

    total_frames = frame_count #min(frame_count, 300)
    sampling_period = max(total_frames // n_clips, 1)
    n_snipets = min(n_clips, total_frames // sampling_period)
    if not is_validation:
        starts = np.random.randint(0, max(1, sampling_period - clip_len), n_snipets)
    else:
        starts = np.zeros(n_snipets)
    offsets = np.arange(0, total_frames, sampling_period)
    selection = np.concatenate([np.arange(of+s, of+s+clip_len) for of, s in zip(offsets, starts)])

    frames = []
    count = ret_count = 0
    while count < selection[-1]+clip_len:
        retained, frame = capture.read()
        if count not in selection:
            count += 1
            continue
        if not retained:
            if len(frames) > 0:
                frame = np.copy(frames[-1])
            else:
                frame = (255*np.random.rand(frame_height, frame_width, 3)).astype('uint8')
            frames.append(frame)
            ret_count += 1
            count += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    capture.release()
    frames = np.stack(frames)
    total = n_clips * clip_len
    while frames.shape[0] < total:
        frames = np.concatenate([frames, frames[:(total - frames.shape[0])]])
    frames = frames.reshape([n_clips, clip_len, frame_height, frame_width, 3])
    return frames


class VideoDataset(Dataset):

    def __init__(self, fnames, labels, class_embed, classes, name, load_clips=load_clips_tsn,
                 clip_len=8, n_clips=1, crop_size=112, is_validation=False, evaluation_only=False):
        if 'kinetics' in name:
            fnames, labels = self.clean_data(fnames, labels)
        self.data = fnames
        self.labels = labels
        self.class_embed = class_embed
        self.class_name = classes
        self.name = name

        self.clip_len = clip_len
        self.n_clips = n_clips

        self.crop_size = crop_size  # 112
        self.is_validation = is_validation

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        self.transform = get_transform(self.is_validation, crop_size)
        self.loadvideo = load_clips

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label_array[idx]
        buffer = self.loadvideo(sample, self.clip_len, self.n_clips, self.is_validation)
        if len(buffer) == 0:
            buffer = np.random.rand(self.n_clips, 3, self.clip_len, 112, 112).astype('float32')
            buffer = torch.from_numpy(buffer)
            return buffer, -1, self.class_embed[0], -1
        s = buffer.shape
        buffer = buffer.reshape(s[0] * s[1], s[2], s[3], s[4])
        buffer = torch.stack([torch.from_numpy(im) for im in buffer], 0)
        buffer = self.transform(buffer)
        buffer = buffer.reshape(3, s[0], s[1], self.crop_size, self.crop_size).transpose(0, 1)
        return buffer, label, self.class_embed[label], idx

    def __len__(self):
        return len(self.data)

    @staticmethod
    def clean_data(fnames, labels):
        if not isinstance(fnames[0], str):
            print('Cannot check for broken videos')
            return fnames, labels
        broken_videos_file = 'assets/kinetics_broken_videos.txt'
        if not os.path.exists(broken_videos_file):
            print('Broken video list does not exists')
            return fnames, labels

        t = time()
        with open(broken_videos_file, 'r') as f:
            broken_samples = [r[:-1] for r in f.readlines()]
        data = [x[75:] for x in fnames]
        keep_sample = np.in1d(data, broken_samples) == False
        fnames = np.array(fnames)[keep_sample]
        labels = np.array(labels)[keep_sample]
        print('Broken videos %.2f%% - removing took %.2f' % (100 * (1.0 - keep_sample.mean()), time() - t))
        return fnames, labels

