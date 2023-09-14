import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import networkx as nx
import copy
from ges.scores.gauss_obs_l0_pen import GaussObsL0Pen
import ges.utils as utils
from itertools import chain
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RMSE1=[]
MAE1=[]
RMSE2=[]
MAE2=[]

def add_subject(x_train, y_train, sort, n):
    for i in range(n):
        path1 = "".join(np.hstack((np.hstack(("p", i + 1)), ".csv")))
        parkinson_x = pd.read_csv(path1)
        columns = ['Unnamed: 0']
        parkinson_x = parkinson_x.drop(columns, axis=1)
        columns = ['MDS']
        parkinson_y = pd.DataFrame(parkinson_x, columns=columns)  # ['motor_UPDRS']
        # columns1 = ['motor_UPDRS', 'total_UPDRS']
        parkinson_x = parkinson_x.drop(columns, axis=1)

        dataset = parkinson_x.values
        parkinson_x = dataset.astype('float32')
        dataset = parkinson_y.values
        parkinson_y = dataset.astype('float32')

        x_train=np.vstack((x_train,parkinson_x))
        y_train = np.vstack((y_train, parkinson_y))
    return x_train,y_train

n=5
n_estimators1 = n_estimators2 = max_depth1 = max_depth2 = 30

for t in range(42):
    path1 = "".join(np.hstack((np.hstack(("p", t + 1)), ".csv")))
    parkinson_x = pd.read_csv(path1)
    columns = ['Unnamed: 0']
    parkinson_x = parkinson_x.drop(columns, axis=1)
    columns = ['motor_UPDRS']
    parkinson_y = pd.DataFrame(parkinson_x, columns=columns)  # ['motor_UPDRS']
    parkinson_x = parkinson_x.drop(columns, axis=1)
    print(parkinson_x.shape)

    dataset = parkinson_x.values
    parkinson_x = dataset.astype('float32')
    dataset = parkinson_y.values
    parkinson_y = dataset.astype('float32')

    xx_train, x_test, yy_train, y_test = train_test_split(parkinson_x, parkinson_y, test_size=0.2, shuffle=True, random_state=86)
    xx_train, x_val,  yy_train,  y_val = train_test_split(xx_train, yy_train, test_size=0.25, shuffle=True, random_state=83)

    csv = "".join(np.hstack((np.hstack(("motor-mae", t + 1)), ".csv")))
    parkinson_x = pd.read_csv(csv)
    columns = ['Unnamed: 0']
    sort = parkinson_x.drop(columns, axis=1)
    dataset = sort.values
    sort = dataset.astype('float32')
    sort = sort.reshape((1, -1))
    sort = [i for item in list(sort) for i in item]
    print(sort)
    print(xx_train.shape)
    x_train, y_train=add_subject(xx_train, yy_train,sort,n)
    print(x_train.shape)

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scale2 = MinMaxScaler()
    x_train = scale2.fit_transform(x_train)
    scale3 = MinMaxScaler()
    y_train = scale3.fit_transform(y_train)

    scale = MinMaxScaler()
    xx_train = scale.fit_transform(xx_train)
    x_test = scale.fit_transform(x_test)
    scale1 = MinMaxScaler()
    yy_train = scale1.fit_transform(yy_train)
    y_test = scale1.fit_transform(y_test)

    xy_train = np.hstack((x_train, y_train))

    from cdt.causality.graph import PC, LiNGAM, CAM, GES, SAM, CGNN, GIES, CCDr
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    dd = pd.DataFrame(xy_train)
    dd.to_csv("xytrain.csv")

    data = pd.read_csv("xytrain.csv")
    col = ['Unnamed: 0', ]
    data = data.drop(col, axis=1)
    print("data:", data)
    pc_out = GES().create_graph_from_data(data)
    estimate = np.array(nx.adjacency_matrix(pc_out).todense())
    print("data.shape:", data.shape)
    estimate = estimate + (estimate.T - estimate) * (estimate.T > estimate)
    print(estimate)

    def delete(x, y, H, A):
        """
        Applies the delete operator:
          1) deletes the edge x -> y or x - y
          2) for every node h in H
               * orients the edge y -> h
               * if the edge with x is undirected, orients it as x -> h

        Note that H must be a subset of the neighbors of y which are
        adjacent to x. A ValueError exception is thrown otherwise.

        Parameters
        ----------
        x : int
            the "origin" node (i.e. x -> y or x - y)
        y : int
            the "target" node
        H : iterable of ints
            a subset of the neighbors of y which are adjacent to x
        A : np.array
            the current adjacency matrix

        Returns
        -------
        new_A : np.array
            the adjacency matrix resulting from applying the operator

        """
        H = set(H)
        # Check inputs
        if A[x, y] == 0:
            raise ValueError("There is no (un)directed edge from x=%d to y=%d" % (x, y))
        # neighbors of y which are adjacent to x
        na_yx = utils.na(y, x, A)
        if not H <= na_yx:
            raise ValueError(
                "The given set H is not valid, H=%s is not a subset of NA_yx=%s" % (H, na_yx))
        # Apply operator
        new_A = A.copy()
        # delete the edge between x and y
        new_A[x, y], new_A[y, x] = 0, 0
        # orient the undirected edges between y and H towards H
        new_A[list(H), y] = 0
        # orient any undirected edges between x and H towards H
        n_x = utils.neighbors(x, A)
        new_A[list(H & n_x), x] = 0
        return new_A

    def score_valid_delete_operators(x, y, A, cache, debug=0):
        """Generate and score all valid delete(x,y,H) operators involving the edge
        x -> y or x - y, and all possible subsets H of neighbors of y which
        are adjacent to x.

        Parameters
        ----------
        x : int
            the "origin" node (i.e. x -> y or x - y)
        y : int
            the "target" node
        A : np.array
            the current adjacency matrix
        cache : instance of ges.scores.DecomposableScore
            the score cache to compute the score of the
            operators that are valid
        debug : int
            if larger than 0, debug are traces printed. Higher values
            correspond to increased verbosity

        Returns
        -------
        valid_operators : list of tuples
            a list of tubles, each containing a valid operator, its score
            and the resulting connectivity matrix

        """
        # Check inputs
        if A[x, y] == 0:
            raise ValueError("There is no (un)directed edge from x=%d to y=%d" % (x, y))
        # One-hot encode all subsets of H0, plus one column to mark if
        # they have already passed the validity condition
        na_yx = utils.na(y, x, A)
        H0 = sorted(na_yx)
        p = len(A)
        if len(H0) == 0:
            subsets = np.zeros((1, (p + 1)), dtype=bool)
        else:
            subsets = np.zeros((2 ** len(H0), (p + 1)), dtype=bool)
            subsets[:, H0] = utils.cartesian([np.array([False, True])] * len(H0), dtype=bool)
        valid_operators = []
        print("    delete(%d,%d) H0=" % (x, y), set(H0)) if debug > 1 else None
        while len(subsets) > 0:
            print("      len(subsets)=%d, len(valid_operators)=%d" %
                  (len(subsets), len(valid_operators))) if debug > 1 else None
            # Access the next subset
            H = np.where(subsets[0, :-1])[0]
            cond_1 = subsets[0, -1]
            subsets = subsets[1:]
            # Check if the validity condition holds for H, i.e. that
            # NA_yx \ H is a clique.
            # If it has not been tested previously for a subset of H,
            # check it now
            if not cond_1 and utils.is_clique(na_yx - set(H), A):
                cond_1 = True
                # For all supersets H' of H, the validity condition will also hold
                supersets = subsets[:, H].all(axis=1)
                subsets[supersets, -1] = True
            # If the validity condition holds, apply operator and compute its score
            print("      delete(%d,%d,%s)" % (x, y, H), "na_yx - H = ",
                  na_yx - set(H), "validity:", cond_1) if debug > 1 else None
            if cond_1:
                # Apply operator
                new_A = delete(x, y, H, A)
                # Compute the change in score
                aux = (na_yx - set(H)) | utils.pa(y, A) | {x}
                # print(x,y,H,"na_yx:",na_yx,"old:",aux,"new:", aux - {x})
                old_score = cache.local_score(y, aux)
                new_score = cache.local_score(y, aux - {x})
                print("        new: s(%d, %s) = %0.6f old: s(%d, %s) = %0.6f" %
                      (y, aux - {x}, new_score, y, aux, old_score)) if debug > 1 else None
                # Add to the list of valid operators
                valid_operators.append((new_score - old_score, new_A, x, y, H))
                print("    delete(%d,%d,%s) -> %0.16f" %
                      (x, y, H, new_score - old_score)) if debug else None
        # Return all the valid operators
        return valid_operators

    def backward_step(A, cache, debug=0):
        # Construct edge candidates:
        #   - directed edges
        #   - undirected edges, counted only once
        fro, to = np.where(utils.only_directed(A))
        directed_edges = zip(fro, to)
        fro, to = np.where(utils.only_undirected(A))
        undirected_edges = filter(lambda e: e[0] > e[1], zip(fro, to))  # zip(fro,to)
        edge_candidates = list(directed_edges) + list(undirected_edges)
        assert len(edge_candidates) == utils.skeleton(A).sum() / 2
        # For each edge, enumerate and score all valid operators
        valid_operators = []
        print("  %d candidate edges" % len(edge_candidates)) if debug > 1 else None
        for (x, y) in edge_candidates:
            valid_operators += score_valid_delete_operators(x, y, A, cache, debug=max(0, debug - 1))
        # Pick the edge/operator with the highest score
        if len(valid_operators) == 0:
            print("  No valid delete operators remain") if debug else None
            return 0, A
        else:
            scores = [op[0] for op in valid_operators]
            score, new_A, x, y, H = valid_operators[np.argmax(scores)]
            print("  Best operator: delete(%d, %d, %s) -> (%0.4f)" %
                  (x, y, H, score)) if debug else None
            return score, new_A

    def shap(money, zuhe):
        P = set(chain(*zuhe))

        def len1(a):
            int_count = 0
            for i in a:
                if i.isdigit():
                    int_count += 1
            return int_count

        def phi(channel_index):
            S_channel = [k for k in money.keys() if str(channel_index) in k]
            score = 0
            print(f"Computing phi for channel {channel_index}...")
            for S in tqdm(S_channel):
                score += money[S] / len1(S)
            return score

        value1 = []
        for j in P:
            value = phi(j)
            value1.append(value)
        return value1

    neighbor1 = []
    for i in range(estimate.shape[0]):
        if estimate[i, -1] == 1:
            neighbor1.append(i)
    print("neighbor1:", neighbor1)

    ratio = 0.6
    G = nx.random_graphs.erdos_renyi_graph(len(neighbor1), ratio)
    c = nx.find_cliques(G)

    c_list = []
    for clist in c:
        c_list.append(clist)
    print("c_list:", c_list)

    zuhe = []
    for ic in c_list:
        ll = []
        for jc in ic:
            ll.append(neighbor1[jc])
        zuhe.append(ll)
    print("zuhe:", zuhe)

    money = {}
    for l in zuhe:
        if len(l) != 1:  # [3, 4]
            # print("l:", l)
            new_list = []
            for aa in l:
                new_list.append(list(neighbor1).index(aa))
            # print("new_list:", new_list)
            estimate1 = copy.deepcopy(estimate)
            index_target = len(neighbor1) - 1
            estimate1[index_target, new_list] = 0
            estimate1[new_list, index_target] = 0
            # print("estimate1:",estimate1)
            data1 = data.values
            data1 = data1.astype('float32')
            cache = GaussObsL0Pen(data1)
            score, _ = backward_step(estimate1, cache)
            # print("score:", score)
            money[str(l)] = score
        if len(l) == 1:

            for ll in l:  # [3]
                # print("estimate:", estimate)

                estimate1 = copy.deepcopy(estimate)

                index_nb = list(neighbor1).index(ll)
                index_target = len(neighbor1) - 1
                estimate1[index_target, index_nb] = 0
                estimate1[index_nb, index_target] = 0
                # print("ll:",ll)
                # print("AAA:",estimate1)
                data1 = data.values
                data1 = data1.astype('float32')
                cache = GaussObsL0Pen(data1)
                score, _ = backward_step(estimate1, cache)
                print("score:", score)
                money[str(l)] = score
    print("money1:", money)

    weight = shap(money, zuhe)
    m_sorted = sorted(enumerate([-i for i in weight]), key=lambda x: x[1])
    sorted_inds = [m[0] for m in m_sorted]
    new_sorted_inds = []
    for i in sorted_inds:
        new_sorted_inds.append(neighbor1[i])

    sampling1 = np.array(new_sorted_inds).reshape((1, -1))
    print(sampling1.shape)
    if int(sampling1.shape[1] * 0.8) >= 1:
        sampling1 = sampling1[0, :int(sampling1.shape[1] * 0.8)]
    else:
        sampling1 = sampling1
        sampling1 = np.array(sampling1).reshape((1, -1))
        sampling1 = [i for item in list(sampling1) for i in item]
    print("sampling1:", sampling1)

    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators = n_estimators1, max_depth = max_depth1)
    rf.fit(xx_train[:, sampling1], yy_train)
    pre = rf.predict(x_test[:, sampling1])
    pre = pre.reshape((-1, 1))
    testPredict = scale1.inverse_transform(pre)
    testY = scale1.inverse_transform(y_test)

    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(testY, testPredict))
    mae = mean_absolute_error(testY, testPredict)
    RMSE1.append(rmse)
    MAE1.append(mae)

    shapley = {}
    for i in list(sampling1):
        neighbor2 = []
        for j in range(estimate.shape[0]):
            if estimate[i, j] == 1:
                neighbor2.append(j)
        for t in neighbor1:
            if t in neighbor2:
                neighbor2.remove(t)
        ratio = 0.8
        G = nx.random_graphs.erdos_renyi_graph(len(neighbor2), ratio)
        c = nx.find_cliques(G)

        c_list = []
        for clist in c:
            c_list.append(clist)
        zuhe = []
        for ic in c_list:
            ll = []
            for jc in ic:
                ll.append(neighbor2[jc])
            zuhe.append(ll)

        money = {}
        for l in zuhe:
            if len(l) != 1:  # [3, 4]
                # print("l:", l)
                new_list = []
                for aa in l:
                    new_list.append(list(neighbor2).index(aa))
                estimate1 = copy.deepcopy(estimate)
                index_target = len(neighbor2) - 1
                estimate1[i, new_list] = 0
                estimate1[new_list, i] = 0
                data1 = data.values
                data1 = data1.astype('float32')
                cache = GaussObsL0Pen(data1)
                score, _ = backward_step(estimate1, cache)
                # print("score:", score)
                money[str(l)] = score
            if len(l) == 1:
                for ll in l:  # [3]
                    estimate1 = copy.deepcopy(estimate)  # 深拷贝才不会改变原列表中的值
                    index_nb = list(neighbor2).index(ll)
                    index_target = len(neighbor2) - 1
                    estimate1[i, index_nb] = 0
                    estimate1[index_nb, i] = 0
                    data1 = data.values
                    data1 = data1.astype('float32')
                    cache = GaussObsL0Pen(data1)
                    score, _ = backward_step(estimate1, cache)
                    print("score:", score)
                    money[str(l)] = score
        print("money1:", money)
        weight = shap(money, zuhe)

        count = 0
        for k in neighbor2:
            shapley[k] = weight[count]
            count = count + 1
    print(shapley)
    z = zip(shapley.values(), shapley.keys())
    dd = sorted(z, reverse=True)

    sampling2 = []
    for i in range(len(dd)):
        sampling2.append(dd[i][1])

    w = []
    for i in sampling2:
        print("i:", i)
        if i != 43:
            if i not in sampling1:
                w.append(i)
    sampling2=w

    sampling2 = np.array(sampling2).reshape((1, -1))
    sampling2 = sampling2[0, :int(sampling2.shape[1] * 0.5)]

    feature = np.hstack((np.array(sampling1), np.array(sampling2))).reshape((1, -1))
    feature1 = [i for item in list(feature) for i in item]

    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators = n_estimators2, max_depth = max_depth2)
    rf.fit(xx_train[:, feature1], yy_train)
    pre = rf.predict(x_test[:, feature1])
    pre = pre.reshape((-1, 1))
    testPredict = scale1.inverse_transform(pre)
    testY = scale1.inverse_transform(y_test)

    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(testY, testPredict))
    mae = mean_absolute_error(testY, testPredict)
    RMSE2.append(rmse)
    MAE2.append(mae)

print(np.mean(RMSE2))
print(np.mean(MAE2))