import torch
from GCL.eval import get_split, LREvaluator, from_predefined_split
import numpy as np
from utils import seed_everything


def test(model, data, dataset_name, repeat_times=3, test_epochs=2000, lr2=0.01, test_interval=5, wd2=0.0):
    z = model.get_embedding(data.x, data.edge_index)
    z = z.detach()

    acc_list = []
    for i in range(repeat_times):
        seed_everything(i)
        if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            split = from_predefined_split(data)
        else:
            split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)

        evaluator = LREvaluator(num_epochs=test_epochs, learning_rate=lr2, test_interval=test_interval,
                                weight_decay=wd2)
        result = evaluator(z, data.y, split)
        acc_list.append(result['acc'] * 100)

    print(f'Average accuracy: {np.mean(acc_list):.2f} + {np.std(acc_list):.2f}')
    return np.mean(acc_list), np.std(acc_list)


from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score


def evaluate_clustering(emb, num_class, true_y, repetition_cluster):
    true_y = true_y.detach().cpu().numpy()
    embeddings = torch.nn.functional.normalize(emb, dim=-1, p=2).detach().cpu().numpy()

    estimator = KMeans(n_clusters=num_class, n_init='auto')

    NMI_list = []
    ARI_list = []
    FMI_list = []

    for _ in range(repetition_cluster):
        estimator.fit(embeddings)
        y_pred = estimator.predict(embeddings)

        nmi_score = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(true_y, y_pred)
        NMI_list.append(nmi_score)
        ARI_list.append(ari_score)

        fmi_score = fowlkes_mallows_score(true_y, y_pred)
        FMI_list.append(fmi_score)

    return np.mean(NMI_list), np.std(NMI_list), np.mean(ARI_list), np.std(ARI_list), np.mean(FMI_list), np.std(FMI_list)
