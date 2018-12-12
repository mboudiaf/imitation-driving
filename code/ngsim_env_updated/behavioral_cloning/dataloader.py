import collections
import _pickle as cPickle
import h5py
import math
import numpy as np
import os
import random
import utils
# Class to load and preprocess data


class DataLoader():
    def __init__(self, batch_size, val_frac, seq_length, extract_temporal):
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seq_length = seq_length

        print ('validation fraction: ', self.val_frac)

        print ("loading data...")
        self._load_data_test()

        print ('creating splits...')
        self._create_split()



    def _trim_data(self, full_s, full_a, intervals):
        # Python indexing; find bounds on data given seq_length
        intervals -= 1
        lengths = np.floor(np.diff(np.append(intervals, len(
            full_s) - 1)) / self.seq_length) * self.seq_length
        intervals = np.vstack((intervals, intervals + lengths)).T.astype(int)
        ret_bounds = np.insert(np.cumsum(lengths), 0, 0.).astype(int)

        # Remove states that don't fit due to value of seq_length
        s = np.zeros((int(sum(lengths)), full_s.shape[1]))
        for i in range(len(ret_bounds) - 1):
            s[ret_bounds[i]:ret_bounds[i + 1]
              ] = full_s[intervals[i, 0]:intervals[i, 1]]
        s = np.reshape(s, (-1, self.seq_length, full_s.shape[1]))

        # Remove actions that don't fit due to value of seq_length
        a = np.zeros((int(sum(lengths)), full_a.shape[1]))
        for i in range(len(ret_bounds) - 1):
            a[ret_bounds[i]:ret_bounds[i + 1]
              ] = full_a[intervals[i, 0]:intervals[i, 1]]
        a = np.reshape(a, (-1, self.seq_length, full_a.shape[1]))

        return s, a
        
    def _load_data_test(self,exp_dir='./data/experiments',expert_filepath='./data/trajectories/ngsim_all.h5',env_H=200,env_primesteps=50,normalize_clip_std_multiple=10.,ngsim_filename='trajdata_lankershim_trajectories.txt'):
        args={'ngsim_filename':ngsim_filename,'env_H':env_H,'env_primesteps':env_primesteps,'env_action_repeat':1,'n_envs':50,'remove_ngsim_veh':False,'env_reward':0,'env_multiagent':False}
        _,act_low,act_high=utils.build_ngsim_env(args,exp_dir,vectorize=False)
        data=utils.load_data(
                expert_filepath,
                act_low=act_low,
                act_high=act_high,
                min_length=env_H + env_primesteps,
                clip_std_multiple = normalize_clip_std_multiple,
                ngsim_filename = ngsim_filename
                )
        self.s = data['observations']
        self.a = data['actions']
        valid_feat=[]
        for feat in range(self.s.shape[1]):
            if not np.isnan(self.s[:,feat]).any():
                if np.abs(self.s[:,feat]).sum()>1e-05:
                    valid_feat.append(feat)
                else:
                    print("Feature ",feat," is invalid")
        print(len(valid_feat), " valid features")
        self.s = self.s[:,valid_feat]

        #self.s_mean = data['obs_mean']
        #self.s_std = data['obs_std']
        #data_dir = '../julia/2d_drive_data/data_'
        #data_stuff = '_clb20_rlb0_rll0_clmr100_rlmr50.jld'
        #filename = data_dir + ngsim_filename.split('.')[0] + data_stuff
        #data = h5py.File(filename,'r')
        #intervals =data['intervals'][:]
        #print(data['intervals'].shape[0],s.shape[0])
        #cat_s, cat_a = self._trim_data(s,a,intervals)
        #self.s = cat_s[:int(np.floor(len(cat_s) / self.batch_size) * self.batch_size)]
        #self.s = np.reshape(self.s, (-1, self.batch_size,self.seq_length, cat_s.shape[2]))
        #self.a = cat_a[:int(np.floor(len(cat_a) / self.batch_size) * self.batch_size)]
        #self.a = np.reshape(self.a, (-1, self.batch_size,self.seq_length, cat_a.shape[2]))
        print(self.s.shape,self.a.shape)
        self.batch_dict = {}
        #self.batch_dict["states"] = np.zeros((self.batch_size, self.seq_length, cat_s.shape[2]))
        #self.batch_dict["actions"] = np.zeros((self.batch_size, self.seq_length, cat_a.shape[2]))

        p = np.random.permutation(len(self.s))
        self.s = self.s[p]
        self.a = self.a[p]

    def _load_data(self, extract_temporal):
        data_dir = '/home/malik_boudiaf/gail-driver/julia/2d_drive_data/'
        filenames = ['data_trajdata_i101_trajectories-0750am-0805am',
                    #'data_trajdata_i101_trajectories-0805am-0820am',
                    # 'data_trajdata_i101_trajectories-0820am-0835am',
                    # 'data_trajdata_i80_trajectories-0400-0415',
                    # 'data_trajdata_i80_trajectories-0500-0515',
                    # 'data_trajdata_i80_trajectories-0515-0530',
                     'data_trajdata_lankershim_trajectories']
        data_suff = '_clb20_rlb0_rll0_clmr100_rlmr50.jld'

        filename = data_dir + filenames[0] + data_suff
        data = h5py.File(filename, 'r')
        s = data['features'][:]
        if extract_temporal:
            #rge = np.r_[0:17, 45:85]
            rge = np.r_[0:17]
            s = s[:, rge]
        else:
            #rge = np.r_[0:8, 14:17, 45:85]
            rge = np.r_[0:8, 14:17]
            s = s[:, rge]
        a = data['targets'][:]
        intervals = data['intervals'][:]
        data.close()

        cat_s, cat_a = self._trim_data(s, a, intervals)

        for i in range(1, len(filenames)):
            filename = data_dir + filenames[i] + data_suff
            data = h5py.File(filename, 'r')
            s = data['features'][:]
            if extract_temporal:
                #rge = np.r_[0:17, 45:85]
                rge = np.r_[0:17]
                s = s[:, rge]
            else:
                #rge = np.r_[0:8, 14:17, 45:85]
                rge = np.r_[0:8 , 14:17]
                s = s[:, rge]
            a = data['targets'][:]
            intervals = data['intervals'][:]
            data.close()

            ret_s, ret_a = self._trim_data(s, a, intervals)

            cat_s = np.concatenate((ret_s, cat_s), axis=0)
            cat_a = np.concatenate((ret_a, cat_a), axis=0)

        # Make sure batch_size divides into num of examples
        self.s = cat_s[:int(
            np.floor(len(cat_s) / self.batch_size) * self.batch_size)]
        self.s = np.reshape(self.s, (-1, self.batch_size,
                                     self.seq_length, cat_s.shape[2]))
        self.a = cat_a[:int(
            np.floor(len(cat_a) / self.batch_size) * self.batch_size)]
        self.a = np.reshape(self.a, (-1, self.batch_size,
                                     self.seq_length, cat_a.shape[2]))

        # Print tensor shapes
        print ('states: ', self.s.shape)
        print ('actions: ', self.a.shape)

        # Create batch_dict
        self.batch_dict = {}
        self.batch_dict["states"] = np.zeros(
            (self.batch_size, self.seq_length, cat_s.shape[2]))
        self.batch_dict["actions"] = np.zeros(
            (self.batch_size, self.seq_length, cat_a.shape[2]))

        # Shuffle data
        print ('shuffling...')
        p = np.random.permutation(len(self.s))
        self.s = self.s[p]
        self.a = self.a[p]

    # Separate data into train/validation sets
    def _create_split(self):

        # compute number of batches
        #self.n_batches =len(self.s)
        self.n_batches = math.floor(len(self.s)/self.batch_size)
        self.n_batches_val = int(math.floor(self.val_frac * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val

        print ('num training batches: ', self.n_batches_train)
        print ('num validation batches: ', self.n_batches_val)

        self.reset_batchptr_train()
        self.reset_batchptr_val()

    # Shift and scale data to be zero-mean, unit variance
    def _shift_scale(self, extract_temporal):
        #data = h5py.File("../julia/validation/models/gail_gru.h5", "r")
        #self.shift = data["initial_obs_mean"][:]
        #self.scale = data["initial_obs_std"][:]
        self.shift = self.s.mean(axis=0)
        self.scale = self.s.std(axis=0)
        # Transform data
        self.s = (self.s - self.shift) / self.scale

    # Sample a new batch of data
    def next_batch_train(self):
        # Extract next batch
        batch_index = self.batch_permuation_train[self.batchptr_train]
        cor_indexes = range(batch_index*self.batch_size,(batch_index + 1 )*self.batch_size)
        self.batch_dict["states"] = self.s[cor_indexes,:]
        self.batch_dict["actions"] = self.a[cor_indexes,:]

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    # Return to first batch in train set
    def reset_batchptr_train(self):
        self.batch_permuation_train = np.random.permutation(
            self.n_batches_train)
        self.batchptr_train = 0

    # Return next batch of data in validation set
    def next_batch_val(self):
        # Extract next validation batch
        batch_index = self.batchptr_val + self.n_batches_train - 1
        cor_indexes = range(batch_index*self.batch_size,(batch_index+1)*self.batch_size)
        self.batch_dict["states"] = self.s[cor_indexes,:]
        self.batch_dict["actions"] = self.a[cor_indexes,:]
        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    # Return to first batch in validation set
    def reset_batchptr_val(self):
        self.batchptr_val = 0
