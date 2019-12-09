import numpy as np, cv2, torch
from auxiliary.auxiliary_word2vec import classes2embedding, load_word2vec
from auxiliary.auxiliary_dataset import VideoDataset, filter_samples, filter_classes, filter_overlapping_classes, \
                                        get_ucf101, get_hmdb, get_kinetics

from auxiliary.auxiliary_activitynet import get_activitynet, load_clips_npy
from auxiliary.auxiliary_stillimages import ImageDataset, get_sun


def get_datasets(opt):
    if 'other' in opt.dataset:
        get_datasets = get_all_datasets(opt)
    elif 'both' in opt.dataset:
        get_datasets = get_both_datasets(opt)
    elif 'image' in opt.dataset:
        get_datasets = get_image_datasets(opt)

    datasets = get_datasets(opt)

    # Move datasets to dataloaders.
    dataloaders = {}
    for key, datasets in datasets.items():
        dataloader = []
        for dataset in datasets:
            dl = torch.utils.data.DataLoader(dataset,
                      batch_size=opt.bs // 2 if (not dataset.is_validation and 'image' in opt.dataset and opt.class_total != 0) else opt.bs,
                      num_workers=opt.kernels // 2, shuffle=not dataset.is_validation, drop_last=False)
            dataloader.append(dl)
        dataloaders[key] = dataloader
    return dataloaders


def get_all_datasets(opt):
    wv_model = load_word2vec()

    # TESTING ON UCF101
    test_fnames, test_labels, test_classes = get_ucf101()
    test_class_embedding = classes2embedding('ucf101', test_classes, wv_model)
    print('UCF101: total number of videos {}, classes {}'.format(len(test_fnames), len(test_classes)))

    # TESTING ON HMDB51
    test_fnames2, test_labels2, test_classes2 = get_hmdb()
    test_class_embedding2 = classes2embedding('hmdb51', test_classes2, wv_model)
    print('HMDB51: total number of videos {}, classes {}'.format(len(test_fnames2), len(test_classes2)))

    # TRAINING ON ActivityNet
    test_fnames3, test_labels3, test_classes3 = get_activitynet()
    test_class_embedding3 = classes2embedding('activitynet', test_classes3, wv_model)
    print('ACTIVITYNET: total number of videos {}, classes {}'.format(len(test_fnames3), len(test_classes3)))

    if not opt.evaluate:
        # TRAINING ON KINETICS
        train_fnames, train_labels, train_classes = get_kinetics(opt.dataset)
        train_fnames, train_labels, train_classes = filter_samples(opt, train_fnames, train_labels, train_classes)
        train_class_embedding = classes2embedding('kinetics', train_classes, wv_model)
        print('KINETICS: total number of videos {}, classes {}'.format(len(train_fnames), len(train_classes)))

        # Filter overlapping classes
        train_fnames, train_labels, train_classes, train_class_embedding = filter_overlapping_classes(
            train_fnames, train_labels, train_classes, train_class_embedding,
            np.concatenate([test_class_embedding, test_class_embedding2, test_class_embedding3]),
            opt.class_overlap)
        print('After filtering) KINETICS: total number of videos {}, classes {}'.format(
            len(train_fnames), len(train_classes)))

        train_fnames, train_labels, train_classes, train_class_embedding = filter_classes(opt,
                                    train_fnames, train_labels, train_classes, train_class_embedding)

        # Initialize datasets
        train_dataset = VideoDataset(train_fnames, train_labels, train_class_embedding, train_classes,
                                     'kinetics%d'%len(train_classes), clip_len=opt.clip_len, n_clips=opt.n_clips,
                                     crop_size=opt.size, is_validation=False)

    n_clips = opt.n_clips if not opt.evaluate else max(5*5, opt.n_clips)
    val_dataset   = VideoDataset(test_fnames, test_labels, test_class_embedding, test_classes, 'ucf101',
                                 clip_len=opt.clip_len, n_clips=n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)
    val_dataset2  = VideoDataset(test_fnames2, test_labels2, test_class_embedding2, test_classes2, 'hmdb51',
                                 clip_len=opt.clip_len, n_clips=n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)
    val_dataset3  = VideoDataset(test_fnames3, test_labels3, test_class_embedding3, test_classes3, 'ActivityNet',
                 load_clips_npy, clip_len=opt.clip_len, n_clips=n_clips,
                                 crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)

    if opt.evaluate:
        return {'training': [], 'testing': [val_dataset3, val_dataset, val_dataset2]}
    else:
        return {'training': [train_dataset], 'testing': [val_dataset, val_dataset2]}


def get_both_datasets(opt):
    wv_model = load_word2vec()

    # TESTING ON UCF101
    test_fnames, test_labels, test_classes = get_ucf101()
    test_class_embedding = classes2embedding('ucf101', test_classes, wv_model)
    print('UCF101: total number of videos {}, classes {}'.format(len(test_fnames), len(test_classes)))

    # TESTING ON HMDB51
    test_fnames2, test_labels2, test_classes2 = get_hmdb()
    test_class_embedding2 = classes2embedding('hmdb51', test_classes2, wv_model)
    print('HMDB51: total number of videos {}, classes {}'.format(len(test_fnames2), len(test_classes2)))

    if not opt.evaluate:
        # TRAINING ON KINETICS
        train_fnames, train_labels, train_classes = get_kinetics(opt.dataset)
        train_fnames, train_labels, train_classes = filter_samples(opt, train_fnames, train_labels, train_classes)
        train_class_embedding = classes2embedding('kinetics', train_classes, wv_model)
        print('KINETICS: total number of videos {}, classes {}'.format(len(train_fnames), len(train_classes)))

        # Filter overlapping classes
        train_fnames, train_labels, train_classes, train_class_embedding = filter_overlapping_classes(
            train_fnames, train_labels, train_classes, train_class_embedding,
            np.concatenate([test_class_embedding, test_class_embedding2]),
            opt.class_overlap)
        print('After filtering) KINETICS: total number of videos {}, classes {}'.format(
            len(train_fnames), len(train_classes)))

        train_fnames, train_labels, train_classes, train_class_embedding = filter_classes(opt,
                                    train_fnames, train_labels, train_classes, train_class_embedding)

        # Initialize datasets
        train_dataset = VideoDataset(train_fnames, train_labels, train_class_embedding, train_classes,
                                     'kinetics%d' % len(train_classes), clip_len=opt.clip_len, n_clips=opt.n_clips,
                                     crop_size=opt.size, is_validation=False)

    n_clips = opt.n_clips if not opt.evaluate else max(5*5, opt.n_clips)
    val_dataset   = VideoDataset(test_fnames, test_labels, test_class_embedding, test_classes, 'ucf101',
                                 clip_len=opt.clip_len, n_clips=n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)
    val_dataset2  = VideoDataset(test_fnames2, test_labels2, test_class_embedding2, test_classes2, 'hmdb51',
                                 clip_len=opt.clip_len, n_clips=n_clips, crop_size=opt.size, is_validation=True,
                                 evaluation_only=opt.evaluate)
    if opt.evaluate:
        return {'training': [], 'testing': [val_dataset, val_dataset2]}
    else:
        return {'training': [train_dataset], 'testing': [val_dataset, val_dataset2]}


def get_image_datasets(opt):
    wv_model = load_word2vec()

    # TRAINING ON SUN
    train_fnames2, train_labels2, train_classes2 = get_sun()
    train_class_embedding2 = classes2embedding('sun', train_classes2, wv_model)
    train_dataset = ImageDataset(train_fnames2, train_labels2, train_class_embedding2, train_classes2, 'sun',
                                 clip_len=opt.clip_len, n_clips=opt.n_clips, crop_size=opt.size,
                                 is_validation=False)
    print('SUN: total number of videos {}, classes {}'.format(len(train_fnames2), len(train_classes2)))

    # TESTING ON UCF101
    test_fnames, test_labels, test_classes = get_ucf101()
    test_class_embedding = classes2embedding('ucf101', test_classes, wv_model)
    print('UCF101: total number of videos {}, classes {}'.format(len(test_fnames), len(test_classes)))

    # TESTING ON HMDB51
    test_fnames2, test_labels2, test_classes2 = get_hmdb()
    test_class_embedding2 = classes2embedding('hmdb51', test_classes2, wv_model)
    print('HMDB51: total number of videos {}, classes {}'.format(len(test_fnames2), len(test_classes2)))

    n_clips = opt.n_clips if not opt.evaluate else max(5 * 5, opt.n_clips)
    val_dataset = VideoDataset(test_fnames, test_labels, test_class_embedding, test_classes, 'ucf101',
                               clip_len=opt.clip_len, n_clips=n_clips, crop_size=opt.size, is_validation=True,
                               evaluation_only=opt.evaluate)
    val_dataset2 = VideoDataset(test_fnames2, test_labels2, test_class_embedding2, test_classes2, 'hmdb51',
                                clip_len=opt.clip_len, n_clips=n_clips, crop_size=opt.size, is_validation=True,
                                evaluation_only=opt.evaluate)
    return {'training': [train_dataset], 'testing': [val_dataset, val_dataset2]}

