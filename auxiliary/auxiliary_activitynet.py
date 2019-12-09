import os, numpy as np
import simplejson as json
import cv2

'''
This function is ad-hoc to my personal format of activity-net.
It probably does not work with the defult version. 
So, you need to adjust it to your data format.
'''
def get_activitynet():
    sourcepath = '/workplace/activitynet/'
    sourcepath += 'clips_split/'
    annotationfile = sourcepath + 'annotations_all.csv'
    with open(annotationfile, 'r') as f:
        lines = [l[:-1].split(',') for l in f.readlines()]
        fnames = [sourcepath + 'numpy/' + l[0] + '.npy' for l in lines]
        labels = [l[1] for l in lines]

    classes = np.unique(labels)
    return fnames, labels, classes


def load_clips_npy(fname, clip_len=16, n_clips=1, is_validation=False):
    if not os.path.exists(fname):
        return []

    try:
        clip = np.load(fname, mmap_mode='r')
    except ValueError:
        print('MMAP ERROR!!')
        return []

    frame_count, H, W, ch = clip.shape

    total_frames = min(frame_count, 300)
    sampling_period = max(total_frames // n_clips, 1)
    n_snipets = min(n_clips, total_frames // sampling_period)
    if not is_validation:
        starts = np.random.randint(0, max(1, sampling_period - clip_len), n_snipets)
    else:
        starts = np.zeros(n_snipets)
    offsets = np.arange(0, total_frames, sampling_period)
    selection = np.concatenate([np.arange(of+s, of+s+clip_len) for of, s in zip(offsets, starts)])
    selection = selection[selection < frame_count]

    clip = clip[selection.astype(int)]
    total = n_clips * clip_len
    while clip.shape[0] < total:
        clip = np.concatenate([clip, clip[:(total - clip.shape[0])]])
    clip = clip.reshape([n_clips, clip_len, H, W, 3])
    return clip


"""===================================================================="""
"""================== PREPARE ACTIVITYNET DATASET ====================="""
"""================ EXTRACTS CLIPS FROM FULL VIDEOS ==================="""
"""===================================================================="""


def save_clips2npy(sourcepath, sample):
    fname = sample['filename']
    loc = [anno['temporal_location'] for anno in sample['annotations']]

    if len(loc) == 0 or os.path.exists(savepath + 'videos/' + fname[len('dataset') + 1:-4] + '_%d.npy' % (len(loc)-1)):
        return

    frames = []
    capture = cv2.VideoCapture(sourcepath + 'videos/' + fname)
    count, loc_idx = 0, 0
    while count < loc[-1][1]:
        retained, frame = capture.read()
        if not retained or count < loc[loc_idx][0]:
            count += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        r = 256.0 / h
        h, w = int(r * h), int(r * w)
        frame = resize(frame, (h, w), mode='constant', anti_aliasing=True)
        frame = (255 * frame).astype('uint8')
        frames.append(frame)
        count += 1
        if count == loc[loc_idx][1]:
            frames = np.stack(frames).astype('uint8')
            np.save(savepath + 'videos/' + fname[len('dataset') + 1:-4] + '_%d' % loc_idx, frames)
            loc_idx += 1
            frames = []
    return


def resize_video(video):
    t, h, w, _ = video.shape
    r = 256.0 / min(h, w)
    h, w = int(r * h), int(r * w)
    ret = np.zeros([t, h, w, 3], 'uint8')
    for i, frame in enumerate(video):
        frame = resize(frame, (h, w), mode='constant', anti_aliasing=True)
        ret[i] = (255 * frame).astype('uint8')
    return ret


if __name__ == "__main__":
    from tqdm import tqdm
    from skimage.transform import resize
    from joblib import Parallel, delayed
    import multiprocessing

    sourcepath = '/workplace/activitynet/'
    annotationpath = sourcepath + 'annotation/'
    with open(annotationpath+'annotation-v1.json') as f:
        data = json.load(f)['videos']

    savepath = sourcepath + 'clips_split/'
    os.makedirs(savepath)
    datalist = list(data.values())

    with open(savepath+'annotations_all.csv', 'w') as f:
        for sample in tqdm(datalist):
            fname = sample['filename']
            annotations = sample['annotations']
            loc = [anno['temporal_location'] for anno in annotations]
            labels = [anno['label'] for anno in annotations]
            if len(loc) == 0: continue
            # f.write('{}_0, {}\n'.format(fname[len('dataset') + 1:-4], labels[0]))
            for loc_idx in range(len(loc)):
                f.write('{}_{}, {}\n'.format(fname[len('dataset')+1:-4], loc_idx, labels[loc_idx]))

    # [save_clips(sourcepath, sample) for sample in tqdm(datalist)]
    with Parallel(n_jobs=multiprocessing.cpu_count()) as par:
        par(delayed(save_clips2npy)(sourcepath, sample) for sample in tqdm(datalist))
