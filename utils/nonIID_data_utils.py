import random
import time
import collections
import math
import numpy as np
import pdb

def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def iid_split(dataset, n_clients, frac, seed=1234):
    """
    split classification dataset among `n_clients` in an IID fashion. The dataset is split as follow:

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_clients: number of clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)
    rng.shuffle(selected_indices)

    return iid_divide(selected_indices, n_clients)


def by_labels_non_iid_split(dataset, n_classes, n_clients, n_clusters, alpha, frac, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_clusters == -1:
        n_clusters = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx
    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = np.random.randint(0, len(dataset), n_samples)

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}
    for idx in selected_indices:
        label = dataset[idx]
        group_id = label2cluster[label]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_indices = [[] for _ in range(n_clients)]    
    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster
    
    for cluster_id in range(n_clusters):
            weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
            clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    
    clients_counts = np.cumsum(clients_counts, axis=1)

    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            clients_indices[client_id] += indices

                    
    return clients_indices

def pathological_non_iid_split(dataset, n_classes, n_clients, n_classes_per_client, frac=1, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) sort the data by label
        2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
        3) assign each of the `n_clients` with `n_classes_per_client` shards

    Inspired by the split in
     "Communication-Efficient Learning of Deep Networks from Decentralized Data"__(https://arxiv.org/abs/1602.05629)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: umber of classes present in `dataset`
    :param n_clients: number of clients
    :param n_classes_per_client:
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    if n_classes == 10:
        n_classes_per_client = 2
    else:
        n_classes_per_client = 10
    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = np.arange(len(dataset))

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        label = dataset[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_clients * n_classes_per_client
    shards = iid_divide(sorted_indices, n_shards)
    random.shuffle(shards)
    tasks_shards = iid_divide(shards, n_clients)

    clients_indices = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            clients_indices[client_id] += shard

    return clients_indices

def split_data_by_group(label, args, N, K, least_samples, num_group):
    min_size = 0
    
    print('[+] allocating indices for group')
    idx_map = [[] for _ in range(num_group)]  # map from client to data index in original dataset
    while min_size < least_samples:
        idx_map = [[] for _ in range(num_group)]  # map from client to data index in original dataset
        for k in range(K):
            # get all label of class k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            # get proportion of class k that each client have
            proportions = np.random.dirichlet(np.repeat(args.alpha, num_group))
            # filter only client that have not had more than N/num_node
            proportions = np.array([p * (len(idx_j) < N / num_group) for p, idx_j in zip(proportions, idx_map)])
            # normalize to probability distribution
            proportions = proportions / proportions.sum()
            # ? TODO
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_map = [idx_j + idx.tolist() for idx_j, idx in zip(idx_map, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_map])

    for i in range(num_group):
        idx_map[i] = np.array(idx_map[i])
        np.random.shuffle(idx_map[i])

    print('   - shuffle a portion of test set to include some global data!')
    # extract p_hybrid * p_test from idxmap of each client
    indices = []
    num_shuffle = []
    for i in range(num_group):
        n = int(len(idx_map[i]) * args.p_test * args.p_hybrid)
        indices.append(idx_map[i][-n:])
        num_shuffle.append(n)
    # join the extracted indices and randomly permutate
    indices = np.concatenate(indices)
    indices = np.random.permutation(indices)
    # assign the shuffled indices back to clients
    ptr = 0
    for i, n in enumerate(num_shuffle):
        idx_map[i][-n:] = indices[ptr:ptr + n]
        ptr += n
    print('   - done!')
    return idx_map

def split_data_iid(image, args):
    d_idxs = np.random.permutation(len(image))
    idx_map = np.array_split(d_idxs, args.num_node)
    idx_map = [data_idx.tolist() for data_idx in idx_map]

    return idx_map
    
def split_data_non_iid_label_skew(image, label, args):
    MIN_ALPHA = 0.01
    skewness = args.skewness
    alpha = (-4*np.log(skewness + 10e-8))**4
    alpha = max(alpha, MIN_ALPHA)
    
    print('[+] allocating indices for group')
    total_label, count_label = np.unique(label, return_counts=True)
    p = np.array([1.0*v/len(image) for v in count_label])
    lb_dict = {}
    for lb in label:
        lb_dict[lb] = np.where(label == lb)[0]
    proportions = [np.random.dirichlet(alpha*p) for _ in range(args.num_node)]
    while np.any(np.isnan(proportions)):
        proportions = [np.random.dirichlet(alpha * p) for _ in range(args.num_node)]
    while True:
        mean_prop = np.mean(proportions, axis=0)
        error_norm = ((mean_prop-p)**2).sum()
        print("Error: {:.8f}".format(error_norm))
        if error_norm<=1e-2/ len(total_label):
            break
        exclude_norms = []
        for cid in range(args.num_node):
            mean_excid = (mean_prop * args.num_node - proportions[cid]) / (args.num_node - 1)
            error_excid = ((mean_excid-p) ** 2).sum()
            exclude_norms.append(error_excid)
        excid = np.argmin(exclude_norms)
        sup_prop = [np.random.dirichlet(alpha*p) for _ in range(args.num_node)]
        alter_norms = []
        for cid in range(args.num_node):
            if np.any(np.isnan(sup_prop[cid])):
                continue
            mean_alter_cid = mean_prop - proportions[excid]/args.num_node + sup_prop[cid]/args.num_node
            error_alter = ((mean_alter_cid-p)**2).sum()
            alter_norms.append(error_alter)
        if len(alter_norms)>0:
            alcid = np.argmin(alter_norms)
            proportions[excid] = sup_prop[alcid]
    
    idx_map = [[] for _ in range(args.num_node)]
    dirichlet_dist = [] # for efficiently visualizing
    for lb in total_label:
        lb_idxs = lb_dict[lb]
        lb_proportion = np.array([pi[lb] for pi in proportions])
        lb_proportion = lb_proportion/lb_proportion.sum()
        lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
        lb_datas = np.split(lb_idxs, lb_proportion)
        dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
        idx_map = [local_data+lb_data.tolist() for local_data,lb_data in zip(idx_map, lb_datas)]
    dirichlet_dist = np.array(dirichlet_dist).T
    for i in range(args.num_node):
        np.random.shuffle(idx_map[i])
    print('   - done!')
    return idx_map
    
def split_data_non_iid_imbalance_data(image, label, args, K):
    MIN_ALPHA = 0.01
    alpha = (-4 * np.log(args.skewness + 10e-8)) ** 4
    alpha = max(alpha, MIN_ALPHA)   
    total_label, count_label = np.unique(label, return_counts=True)
    p = np.array([1.0*v/len(image) for v in count_label])
    
    print('[+] allocating indices for group')
    total_data_size = len(image)
    mean_datasize = total_data_size/args.num_node
    mu = np.log(mean_datasize) - 0.5
    sigma = 1
    samples_per_client = np.random.lognormal(mu, sigma, (args.num_node)).astype(int)
    thresold = int(0.1*total_data_size)
    delta = int(0.1 * thresold)
    crt_data_size = sum(samples_per_client)
    
    while crt_data_size != total_data_size:
        if crt_data_size - total_data_size >= thresold:
            maxid = np.argmax(samples_per_client)
            samples = np.random.lognormal(mu, sigma, (args.num_node))
            new_size_id = np.argmin([np.abs(crt_data_size-samples_per_client[maxid]+s) for s in samples])
            samples_per_client[maxid] = samples[new_size_id]
        elif crt_data_size - total_data_size >= delta:
            maxid = np.argmax(samples_per_client)
            samples_per_client[maxid] -= delta
        elif crt_data_size - total_data_size >0:
            maxid = np.argmax(samples_per_client)
            samples_per_client[maxid] -= (crt_data_size - total_data_size)
        elif total_data_size - crt_data_size >= delta:
            minid = np.argmin(samples_per_client)
            samples_per_client[minid] += delta
        elif total_data_size - crt_data_size >= delta:
            minid = np.argmin(samples_per_client)
            samples = np.random.lognormal(mu, sigma, (args.num_node))
            new_size_id = np.argmin([np.abs(crt_data_size-samples_per_client[minid]+s) for s in samples])
            samples_per_client[minid] = samples[new_size_id]
        else:
            minid = np.argmin(samples_per_client)
            samples_per_client[minid] += (total_data_size - crt_data_size)
        crt_data_size = sum(samples_per_client)
    
    lb_dict = {}
    for lb in range(K):
        lb_dict[lb] = np.where(label == lb)[0]

    proportions = [np.random.dirichlet(alpha * p) for _ in range(args.num_node)]
    while np.any(np.isnan(proportions)):
        proportions = [np.random.dirichlet(alpha * p) for _ in range(args.num_node)]
    sorted_cid_map = {k:i for k,i in zip(np.argsort(samples_per_client), [_ for _ in range(args.num_node)])}
    error_increase_interval = 5000
    max_error = 1e-2 / K
    loop_count = 0
    crt_id = 0
    while True:
        if loop_count >= error_increase_interval:
            loop_count = 0
            max_error = max_error * 10
        # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
        mean_prop = np.sum([pi*di for pi,di in zip(proportions, samples_per_client)], axis=0)
        mean_prop = mean_prop/mean_prop.sum()
        error_norm = ((mean_prop - p) ** 2).sum()
        print("Error: {:.8f}".format(error_norm))
        if error_norm <= max_error:
            break
        excid = sorted_cid_map[crt_id]
        crt_id = (crt_id+1)%args.num_node
        sup_prop = [np.random.dirichlet(alpha * p) for _ in range(args.num_node)]
        del_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
        del_prop -= samples_per_client[excid]*proportions[excid]
        alter_norms = []
        for cid in range(args.num_node):
            if np.any(np.isnan(sup_prop[cid])):
                continue
            alter_prop = del_prop + samples_per_client[excid]*sup_prop[cid]
            alter_prop = alter_prop/alter_prop.sum()
            error_alter = ((alter_prop - p) ** 2).sum()
            alter_norms.append(error_alter)
        if len(alter_norms) > 0:
            alcid = np.argmin(alter_norms)
            proportions[excid] = sup_prop[alcid]
        loop_count += 1
    
    idx_map = [[] for _ in range(args.num_node)]

    for lb in range(K):
        lb_idxs = lb_dict[lb]
        lb_proportion = np.array([pi[lb]*si for pi,si in zip(proportions, samples_per_client)])
        lb_proportion = lb_proportion / lb_proportion.sum()
        lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
        lb_datas = np.split(lb_idxs, lb_proportion)
        idx_map = [local_data + lb_data.tolist() for local_data, lb_data in zip(idx_map, lb_datas)]
        
    for i in range(args.num_node):
        np.random.shuffle(idx_map[i]) 
    print('   - done!')
    return idx_map
    
def split_data_non_iid_label_skew_quantity(image, label, args):
    dpairs = [[did, label[did]] for did in range(image.shape[0])]
    num = max(int((1 - args.skewness) * args.num_class), 1)
    K = args.num_class
    idx_map = [[] for _ in range(args.num_node)]
    if num == K:
        for k in range(K):
            idx_k = [p[0] for p in dpairs if p[1]==k]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, args.num_node)
            for cid in range(args.num_node):
                idx_map[cid].extend(split[cid].tolist())
    else:
        times = [0 for _ in range(args.num_class)]
        contain = []
        for i in range(args.num_node):
            current = []
            j =0
            while (j < num):
                mintime = np.min(times)
                ind = np.random.choice(np.where(times == mintime)[0])
                if (ind not in current):
                    j = j + 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        for k in range(K):
            idx_k = [p[0] for p in dpairs if p[1]==k]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[k])
            ids = 0
            for cid in range(args.num_node):
                if k in contain[cid]:
                    idx_map[cid].extend(split[ids].tolist())
                    ids += 1
    return idx_map

def split_data_non_iid(image, label, args):
    # calculate alpha = (-4log(skewness + epsilon))**4
    MIN_ALPHA = 0.01
    alpha = (-4 * np.log(args.skewness + 10e-8)) ** 4
    alpha = max(alpha, MIN_ALPHA)
    # ensure imbalance data sizes
    total_data_size = image.shape[0]
    mean_datasize = total_data_size / args.num_node
    mu = np.log(mean_datasize) - 0.5
    sigma = 1
    samples_per_client = np.random.lognormal(mu, sigma, (args.num_node)).astype(int)
    thresold = int(0.1*total_data_size)
    delta = int(0.1 * thresold)
    crt_data_size = sum(samples_per_client)
    # force current data size to match the total data size
    while crt_data_size != total_data_size:
        if crt_data_size - total_data_size >= thresold:
            maxid = np.argmax(samples_per_client)
            samples = np.random.lognormal(mu, sigma, (args.num_node))
            new_size_id = np.argmin([np.abs(crt_data_size-samples_per_client[maxid]+s) for s in samples])
            samples_per_client[maxid] = samples[new_size_id]
        elif crt_data_size - total_data_size >= delta:
            maxid = np.argmax(samples_per_client)
            samples_per_client[maxid] -= delta
        elif crt_data_size - total_data_size >0:
            maxid = np.argmax(samples_per_client)
            samples_per_client[maxid] -= (crt_data_size - total_data_size)
        elif total_data_size - crt_data_size >= delta:
            minid = np.argmin(samples_per_client)
            samples_per_client[minid] += delta
        elif total_data_size - crt_data_size >= delta:
            minid = np.argmin(samples_per_client)
            samples = np.random.lognormal(mu, sigma, (args.num_node))
            new_size_id = np.argmin([np.abs(crt_data_size-samples_per_client[minid]+s) for s in samples])
            samples_per_client[minid] = samples[new_size_id]
        else:
            minid = np.argmin(samples_per_client)
            samples_per_client[minid] += (total_data_size - crt_data_size)
        crt_data_size = sum(samples_per_client)
    # count the label distribution
    labels = [label[did] for did in range(image.shape[0])]
    lb_counter = collections.Counter(labels)
    p = np.array([1.0 * v / image.shape[0] for v in lb_counter.values()])
    lb_dict = {}
    labels = np.array(labels)
    for lb in range(len(lb_counter.keys())):
        lb_dict[lb] = np.where(labels == lb)[0]
    proportions = [np.random.dirichlet(alpha * p) for _ in range(args.num_node)]
    while np.any(np.isnan(proportions)):
        proportions = [np.random.dirichlet(alpha * p) for _ in range(args.num_node)]
    sorted_cid_map = {k:i for k,i in zip(np.argsort(samples_per_client), [_ for _ in range(args.num_node)])}
    error_increase_interval = 5000
    max_error = 1e-2 / args.num_class
    loop_count = 0
    crt_id = 0
    while True:
        if loop_count >= error_increase_interval:
            loop_count = 0
            max_error = max_error * 10
        # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*args.num_class
        mean_prop = np.sum([pi*di for pi,di in zip(proportions, samples_per_client)], axis=0)
        mean_prop = mean_prop/mean_prop.sum()
        error_norm = ((mean_prop - p) ** 2).sum()
        print("Error: {:.8f}".format(error_norm))
        if error_norm <= max_error:
            break
        excid = sorted_cid_map[crt_id]
        crt_id = (crt_id+1)%args.num_node
        sup_prop = [np.random.dirichlet(alpha * p) for _ in range(args.num_node)]
        del_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
        del_prop -= samples_per_client[excid]*proportions[excid]
        alter_norms = []
        for cid in range(args.num_node):
            if np.any(np.isnan(sup_prop[cid])):
                continue
            alter_prop = del_prop + samples_per_client[excid]*sup_prop[cid]
            alter_prop = alter_prop/alter_prop.sum()
            error_alter = ((alter_prop - p) ** 2).sum()
            alter_norms.append(error_alter)
        if len(alter_norms) > 0:
            alcid = np.argmin(alter_norms)
            proportions[excid] = sup_prop[alcid]
        loop_count += 1
        
    idx_map = [[] for _ in range(args.num_node)]
    for lb in lb_counter.keys():
        lb_idxs = lb_dict[lb]
        lb_proportion = np.array([pi[lb]*si for pi,si in zip(proportions, samples_per_client)])
        lb_proportion = lb_proportion / lb_proportion.sum()
        lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
        lb_datas = np.split(lb_idxs, lb_proportion)
        idx_map = [local_data + lb_data.tolist() for local_data, lb_data in zip(idx_map, lb_datas)]
    for i in range(args.num_node):
        np.random.shuffle(idx_map[i])
    
    return idx_map

def split_data_non_iid_pareto(image, label, args):
    labels = list(label)
    total_image = image.shape[0]
    idxs = range(total_image)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]
    tmp = 0
    list_tmp = []

    for i in range(total_image):
        if(label[idxs[i]] == tmp):
            tmp +=1
            list_tmp.append(i)
    list_tmp.append(total_image)
    
    if args.dataset == 'cifar10':
        limit = 100
        client_class = 3
    elif args.dataset == 'cifar100' or 'fmnist':
        limit = 200
        client_class = 10
      
    list_label = {}
    a = set()
    for i in range(args.num_node):
        list_label[i] = np.random.randint(0,args.num_class, client_class) 
        a.update(list_label[i])
        
    key = True
    count = 0
        
    while(key):
        count += 1
        if count > limit:
            print("Infinite loop")
            exit(0)
            
        try:
            list_dict = [0] * args.num_class
            for i in range(args.num_class):
                list_dict[i] = idxs[list_tmp[i]:list_tmp[i+1]]

            
            dis = np.random.pareto(args.alpha, args.num_node)
            dis = dis/np.sum(dis)
            percent = [0] * args.num_class
            for i in range(args.num_node):
                for j in list_label[i]:
                    percent[j] += dis[i]

            maxx = max(percent)
            if args.dataset == 'cifar10':
                total = np.around(10000/maxx)
                addition = 1
            elif args.dataset == 'cifar100':
                total = np.around(1000/maxx)
                addition = 1.5
            sample_client = [math.ceil(total * dis[i] * addition) for i in range(args.num_node)]
            for i in  range(len(sample_client)):
                if sample_client[i] < 5:
                    sample_client[i] = 5
                if args.dataset == 'cifar100':
                    if sample_client[i] > 600:
                        sample_client[i] = 600
            dict_client = {}
            idx_map = []
            for i in range(args.num_node):
                dict_client[i] = []
            for i in range(args.num_node):
                x = math.ceil(sample_client[i])
                for j in list_label[i]:
                    a = np.random.choice(list_dict[j],x,replace=False)
                    # list_dict[list_label[i][0]] = list(set(list_dict[list_label[i][0]]) - set(a))
                    dict_client[i] = dict_client[i]  + list(a)

                dict_client[i] = [int(j) for j in dict_client[i]]
                idx_map.append([int(j) for j in dict_client[i]])
            key = False
        except:
            key = True
    return idx_map