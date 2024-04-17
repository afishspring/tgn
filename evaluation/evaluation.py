import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import csv


def eval_edge_prediction_by_timestamps(model, negative_edge_sampler, data, n_neighbors, batch_size=200, output_file="results/evaluation_t.csv"):
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []

    with torch.no_grad():
        model = model.eval()
        num_test_instance = len(data.sources)
        unique_timestamps = sorted(set(data.timestamps))

        for timestamp in unique_timestamps:
            sources_batch = []
            destinations_batch = []
            timestamps_batch = []
            edge_idxs_batch = []

            for i in range(num_test_instance):
                if data.timestamps[i] == timestamp:
                    sources_batch.append(data.sources[i])
                    destinations_batch.append(data.destinations[i])
                    timestamps_batch.append(data.timestamps[i])
                    edge_idxs_batch.append(data.edge_idxs[i])

            size = len(sources_batch)
            if size == 0:
                continue

            _, negative_samples = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_edge_probabilities(np.array(sources_batch),
                                                                  np.array(destinations_batch),
                                                                  negative_samples,
                                                                  np.array(timestamps_batch),
                                                                  np.array(edge_idxs_batch),
                                                                  n_neighbors)

            pred_score = np.concatenate(
                [(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            # Area Under Precision-Recall Curve
            val_ap.append(average_precision_score(true_label, pred_score))
            # AUC area
            val_auc.append(roc_auc_score(true_label, pred_score))

            print("==time=="+str(timestamp))

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Val_AP', 'Val_AUC'])
        for i in range(len(unique_timestamps)):
            writer.writerow([unique_timestamps[i], val_ap[i], val_auc[i]])

    return np.mean(val_ap), np.mean(val_auc)

def eval_edge_prediction_by_rank(model, negative_edge_sampler, data, n_neighbors, batch_size=200, top_k=10):
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    with torch.no_grad():
        model = model.eval()

        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        ranks_pred = []
        
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)

            ranks = np.zeros(size, dtype=bool)

            for index in range(size):
                timestamps_batch.fill(timestamps_batch[index])

                pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                    negative_samples, timestamps_batch,
                                                                    edge_idxs_batch, n_neighbors)
                # 获取从大到小排序的索引
                sorted_indices = np.argsort(-pos_prob.squeeze().cpu().numpy(), kind='stable')  # 注意负号，因为argsort默认是升序
                # 找到第index个pos_prob在排序后的数组中的位置
                rank = np.where(sorted_indices == index)[0][0] + 1
                ranks[index] = rank <= top_k
            
            ranks_pred.append(ranks)

        ranks_pred = np.concatenate(ranks_pred)
        ranks_label = np.ones_like(ranks_pred, dtype=bool)

        val_ap = average_precision_score(ranks_pred, ranks_label)
        val_auc = roc_auc_score(ranks_pred, ranks_label)

        return val_ap, val_auc

def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                  negative_samples, timestamps_batch,
                                                                  edge_idxs_batch, n_neighbors)

            pred_score = np.concatenate(
                [(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)

def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
    pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx: e_idx]
            destinations_batch = data.destinations[s_idx: e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx: e_idx]

            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                         destinations_batch,
                                                                                         destinations_batch,
                                                                                         timestamps_batch,
                                                                                         edge_idxs_batch,
                                                                                         n_neighbors)
            pred_prob_batch = decoder(source_embedding).sigmoid()
            pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc
