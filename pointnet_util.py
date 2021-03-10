""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
import tensorflow as tf
import numpy as np

from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point


def sample_and_group(new_xyz, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0]
    nsample = xyz.get_shape()[1]
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz
def kernel_density_estimation_ball(pts, radius, sigma, N_points = 128, is_norm = False):
    with tf.variable_scope("ComputeDensity") as sc:
        idx, pts_cnt = query_ball_point(radius, N_points, pts, pts)
        g_pts = group_point(pts, idx)
        g_pts -= tf.tile(tf.expand_dims(pts, 2), [1, 1, N_points, 1])

        R = tf.sqrt(sigma)
        xRinv = tf.div(g_pts, R)
        quadform = tf.reduce_sum(tf.square(xRinv), axis = -1)
        logsqrtdetSigma = tf.log(R) * 3
        mvnpdf = tf.exp(-0.5 * quadform - logsqrtdetSigma - 3 * tf.log(2 * 3.1415926) / 2)

        first_val, _ = tf.split(mvnpdf, [1, N_points - 1], axis = 2)

        mvnpdf = tf.reduce_sum(mvnpdf, axis = 2, keepdims = True)

        num_val_to_sub = tf.expand_dims(tf.cast(tf.subtract(N_points, pts_cnt), dtype = tf.float32), axis = -1)

        val_to_sub = tf.multiply(first_val, num_val_to_sub)

        mvnpdf = tf.subtract(mvnpdf, val_to_sub)

        scale = tf.div(1.0, tf.expand_dims(tf.cast(pts_cnt, dtype = tf.float32), axis = -1))
        density = tf.multiply(mvnpdf, scale)

        if is_norm:
            #grouped_xyz_sum = tf.reduce_sum(grouped_xyz, axis = 1, keepdims = True)
            density_max = tf.reduce_max(density, axis = 1, keepdims = True)
            density = tf.div(density, density_max)

        return density
