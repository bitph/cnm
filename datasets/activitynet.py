import os

import h5py
import numpy as np

from datasets.base import BaseDataset, build_collate_data


class ActivityNet(BaseDataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        super().__init__(data_path, vocab, args, **kwargs)
        self.collate_fn = build_collate_data(args['max_num_frames'], args['max_num_words'],
                                             args['frame_dim'], args['word_dim'])
        # with h5py.File(os.path.join(self.args['feature_path'], 'sub_activitynet_v1-3.c3d.hdf5'), 'r') as f:
            # self.frame_features = {key: np.asarray(f[key]['c3d_features']).astype(np.float32) for key in f.keys()}
        
    def _load_frame_features(self, vid):
        #with h5py.File(os.path.join(self.args['feature_path'], '%s.h5' % vid), 'r') as fr:
        #    return np.asarray(fr['feature']).astype(np.float32)
        # return self.frame_features[vid]
        with h5py.File(self.args['feature_path'], 'r') as f:
            if 'clip_vit_32_features' in self.args['feature_path']:
                return np.asarray(f[vid]).astype(np.float32)
            return np.asarray(f[vid]['c3d_features']).astype(np.float32)

    def collate_data(self, samples):
        return self.collate_fn(samples)
