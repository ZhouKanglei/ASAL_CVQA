#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/09/06 15:33:57
import os
import torch
import torch.nn as nn

import numpy as np
from torchvision.models import resnet18
from sklearn.cluster import KMeans

import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer


class OursCap(ContinualModel):
    NAME = 'ours_cap'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(OursCap, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, 'cpu')

        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

    def select_key_frames(self, inputs, num_key_frames):
        """
        Select key frames from the representative samples
        """
        # Load a pre-trained model for feature extraction
        model = resnet18(pretrained=True)
        # Remove the classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()

        # Extract features for each frame
        features = []
        # input dimension: frames x 3 x height x width
        x_size = inputs.shape
        with torch.no_grad():
            features = model(inputs).squeeze().numpy()  # Extract features
        features = features.reshape(x_size[0], -1)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_key_frames)
        kmeans.fit(features)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Select the frame closest to each cluster center
        key_frames = []
        for i in range(num_key_frames):
            cluster_indices = np.where(labels == i)[0]
            cluster_features = features[cluster_indices]
            center = cluster_centers[i]
            distances = np.linalg.norm(cluster_features - center, axis=1)
            closest_index = cluster_indices[np.argmin(distances)]
            key_frames.append(inputs[closest_index])

        key_frames = torch.stack(key_frames, dim=0)

        return key_frames

    def select_samples_by_scores(self, all_names, all_scores, num_per_task):
        # sort the scores from low to high
        scores = np.array(all_scores)
        names = np.array(all_names)
        sorted_indices = np.argsort(scores)
        scores = scores[sorted_indices]
        names = names[sorted_indices]

        # evenly select num_per_task samples from the scores
        selected_samples = []
        if len(scores) < num_per_task:
            selected_indices = np.arange(len(scores))
        else:
            indices = np.linspace(0, len(scores) - 1, num_per_task)
            selected_indices = np.array(indices, dtype=int)

        selected_samples = names[selected_indices]

        return selected_samples

    def add_samples_to_buffer(self, dataset):
        num_per_task = self.args.buffer_size // dataset.N_TASKS

        # assemble the scores and the names of all samples
        train_loader = dataset.train_loader
        all_scores, all_names = [], []
        for _, labels, names in train_loader:
            all_names.extend(names)
            all_scores.extend(labels)

        # select representative samples
        selected_sample_names = self.select_samples_by_scores(
            all_names, all_scores, num_per_task)
        selected_inputs = []
        selected_scores = []

        for inputs, labels, names in train_loader:
            for i, name in enumerate(names):
                if name in selected_sample_names:
                    key_frames = self.select_key_frames(
                        inputs[i], self.args.num_key_frames)
                    selected_inputs.append(key_frames)
                    selected_scores.append(labels[i])

        selected_inputs = torch.stack(selected_inputs, dim=0)
        selected_scores = torch.stack(selected_scores, dim=0)

        # add the selected samples to the buffer
        self.buffer.add_data(
            examples=selected_inputs,
            labels=selected_scores,
            task_labels=torch.ones(num_per_task) * (dataset.i + 1)
        )

        self.args.logging.info(f"Add {num_per_task} samples to the buffer.")

    def save_buffer_feats(self, dataset):
        self.end_task(dataset)

        # get all data in the buffer
        examples, labels, task_ids = self.buffer.get_all_data()

        feats, refined_feats, label_scores, predict_scores, label_task_ids = [], [], [], [], []
        for example, label, task_id in zip(examples, labels, task_ids):

            if task_id:
                with torch.no_grad():
                    example = example.unsqueeze(0).to(self.device)
                    predict_score, feat, refined_feat = self.net.forward_with_adapter(
                        example, returnt='all')
                    feats.append(feat.squeeze().cpu().numpy())
                    refined_feats.append(refined_feat.squeeze().cpu().numpy())
                    label_scores.append(label)
                    predict_scores.append(
                        predict_score.squeeze().cpu().numpy())
                    label_task_ids.append(task_id)

        # save the feature
        output_feats_dir = os.path.join(self.args.output_dir, 'buffer_feats')
        os.makedirs(output_feats_dir, exist_ok=True)
        output_feats_file = os.path.join(
            output_feats_dir, f'buffer_feats_{dataset.i + 1}.npz')

        feats = np.stack(feats, axis=0)
        refined_feats = np.stack(refined_feats, axis=0)
        label_scores = np.stack(label_scores, axis=0)
        predict_scores = np.stack(predict_scores, axis=0)
        label_task_ids = np.stack(label_task_ids, axis=0)

        np.savez(output_feats_file, feats=feats,
                 refined_feats=refined_feats,
                 label_scores=label_scores,
                 predict_scores=predict_scores,
                 task_ids=label_task_ids)
        self.args.logging.info(
            f"Save the buffer features to {output_feats_file}.")

    def save_test_feats(self, dataset):
        test_loader = dataset.test_loader
        feats, label_scores, predict_scores, label_names = [], [], [], []
        for batch_data in test_loader:
            inputs, labels, name = batch_data
            with torch.no_grad():
                inputs = inputs.to(self.device)
                predict_score, feat = self.net(
                    inputs, returnt='all')
                feats.append(feat.cpu().numpy())
                label_scores.append(labels)
                predict_scores.extend(predict_score.cpu().numpy())
                label_names.extend(name)

        feats = np.concatenate(feats, axis=0)

        label_scores = np.concatenate(label_scores, axis=0)
        # save the feature
        output_feats_dir = os.path.join(self.args.output_dir, 'test_feats')
        os.makedirs(output_feats_dir, exist_ok=True)
        output_feats_file = os.path.join(
            output_feats_dir, f'test_feats_{dataset.i + 1}.npz')
        np.savez(output_feats_file, feats=feats,
                 label_scores=label_scores,
                 predict_scores=predict_scores,
                 names=label_names)
        self.args.logging.info(
            f'Save the test features to {output_feats_file}.')

    def end_task(self, dataset):
        self.add_samples_to_buffer(dataset)

        # statistics
        _, _, buf_tl = self.buffer.get_all_data()
        for ttl in buf_tl.unique():
            if ttl == 0:
                continue
            idx = (buf_tl == ttl)

            self.args.logging.info(
                f"Task {int(ttl)} has {sum(idx)} samples in the buffer.")

    def reconstruct_full_video(self, inputs, total_frames):
        reconstructed_inputs = []
        for i in range(len(inputs)):
            key_frames = inputs[i]

            # Calculate the interval between key frames
            interval = (total_frames - len(key_frames)
                        ) // (len(key_frames) - 1)

            # Initialize the list of reconstructed frames
            reconstructed_frames = []

            for i in range(len(key_frames) - 1):
                start_frame = key_frames[i]
                end_frame = key_frames[i + 1]
                reconstructed_frames.append(start_frame)
                for j in range(interval):
                    alpha = j / interval
                    interpolated_frame = (1 - alpha) * \
                        start_frame + alpha * end_frame
                    reconstructed_frames.append(interpolated_frame)

            # Add the last key frame
            reconstructed_frames.append(key_frames[-1])

            reconstructed_frames = torch.stack(reconstructed_frames, dim=0)

            reconstructed_inputs.append(reconstructed_frames)

        reconstructed_inputs = torch.stack(reconstructed_inputs, dim=0)

        return reconstructed_inputs

    def random_select_frames(self, feats, num_frames):
        """
        Randomly select num_frames frames from the input
        """
        indices = np.random.choice(feats.shape[1], num_frames, replace=False)
        indices = np.sort(indices)

        selected_frames = feats[:, indices]

        return selected_frames

    def observe(self, inputs, labels, not_aug_inputs, epoch=True, task=True):
        self.opt.zero_grad()

        outputs, feats = self.net(inputs, returnt='all')
        loss = self.loss(outputs, labels)

        # loss.backward()
        # self.opt.step()

        # self.opt.zero_grad()

        features = self.random_select_frames(feats, self.args.num_key_frames)
        refined_feats = self.net.adapter(features)

        loss = loss + self.args.beta * F.mse_loss(refined_feats, feats)
        
        loss.backward()
        self.opt.step()

        if not self.buffer.is_empty() and task:  # task > 0
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            buf_inputs = buf_inputs.to(self.device)
            buf_labels = buf_labels.to(self.device)

            self.opt.zero_grad()

            buf_outputs = self.net.forward_with_adapter(buf_inputs)
            buf_loss = self.loss(buf_outputs, buf_labels)

            buf_loss.backward()
            self.opt.step()

        return loss.item()
