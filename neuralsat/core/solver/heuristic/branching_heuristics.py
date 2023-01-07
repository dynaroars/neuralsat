'''
    adopt from branching_heuristics.py of alpha-beta-CROWN
'''

from numpy.lib.twodim_base import mask_indices
from torch.nn import functional as F
from torch import nn
import numpy as np
import torch

from core.abstraction.third_party import BoundRelu, BoundLinear, BoundConv, BoundBatchNormalization, BoundAdd
from util.misc.logger import logger


Icp_score_counter = 0


def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound.clamp(max=0)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio

    return slope_ratio, intercept


def get_branching_op(branching_reduceop):
    if branching_reduceop == 'min':
        reduce_op = torch.min
    elif branching_reduceop == 'max':
        reduce_op = torch.max
    elif branching_reduceop == 'mean':
        reduce_op = torch.mean
    else:
        reduce_op = None
    return reduce_op


@torch.no_grad()
def choose_node_parallel_crown(lower_bounds, upper_bounds, orig_mask, net, pre_relu_indices, lAs, sparsest_layer=0, decision_threshold=0.001, batch=5, branching_reduceop='min'):
    batch = min(batch, len(orig_mask[0]))
    # Mask is 1 for unstable neurons. Otherwise it's 0.
    mask = orig_mask
    reduce_op = get_branching_op(branching_reduceop)

    score = []
    intercept_tb = []
    relu_idx = -1

    for layer in reversed(net.net.relus):
        ratio = lAs[relu_idx]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                   upper_bounds[pre_relu_indices[relu_idx]])
        # Intercept
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1
        intercept_tb.insert(0, intercept_candidate.view(batch, -1) * mask[relu_idx])

        # Bias
        input_node = layer.inputs[0]
        assert isinstance(input_node, (BoundConv, BoundLinear, BoundBatchNormalization, BoundAdd))
        if type(input_node) == BoundConv:
            if len(input_node.inputs) > 2:
                b_temp = input_node.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
            else:
                b_temp = 0
        elif type(input_node) == BoundLinear:
            # TODO: consider if no bias in the BoundLinear layer
            b_temp = input_node.inputs[-1].param.detach()
        elif type(input_node) == BoundAdd:
            b_temp = 0
            # print(input_node.inputs)
            for l in input_node.inputs:
                if type(l) == BoundConv:
                    if len(l.inputs) > 2:
                        b_temp += l.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
                if type(l) == BoundBatchNormalization:
                    b_temp += 0  # l.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1) # TODO: bias of BN need refine
                if type(l) == BoundAdd:
                    for ll in l.inputs:
                        if type(ll) == BoundConv:
                            b_temp += ll.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
        else:
            b_temp = input_node.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1)  # for BN, bias is the -3th inputs

        # print(b_temp.shape, ratio_temp_0.shape, ratio.shape)
        b_temp = b_temp * ratio
        bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
        bias_candidate_2 = b_temp * ratio_temp_0
        bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)

        score_candidate = bias_candidate + intercept_candidate
        score.insert(0, (abs(score_candidate).view(batch, -1) * mask[relu_idx]).cpu())

        relu_idx -= 1

    decision = []
    for b in range(batch):
        new_score = [score[j][b] for j in range(len(score))]
        max_info = [torch.max(i, 0) for i in new_score]
        decision_layer = max_info.index(max(max_info))
        decision_index = max_info[decision_layer][1].item()

        if decision_layer != sparsest_layer and max_info[decision_layer][0].item() > decision_threshold:
            decision.append([decision_layer, decision_index])
        else:
            min_info = [[i, torch.min(intercept_tb[i][b], 0)] for i in range(len(intercept_tb)) if
                        torch.min(intercept_tb[i][b]) < -1e-4]
            # import pdb; pdb.set_trace()
            global Icp_score_counter
            if len(min_info) != 0 and Icp_score_counter < 2:
                intercept_layer = min_info[-1][0]
                intercept_index = min_info[-1][1][1].item()
                Icp_score_counter += 1
                decision.append([intercept_layer, intercept_index])
                if intercept_layer != 0:
                    Icp_score_counter = 0
                # else:
                #     print('using first layer split')
                # print('\tusing intercept score')
            else:
                # print('\t using a random choice')
                mask_item = [m[b] for m in mask]
                for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                    if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                        # print('random:', [preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                        decision.append([preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                        break
                Icp_score_counter = 0

    return decision


@torch.no_grad()
def choose_node_parallel_FSB(lower_bounds, upper_bounds, orig_mask, net, pre_relu_indices, lAs, branching_candidates=5, branching_reduceop='min', slopes=None, betas=None, history=None, use_beta=False):

    batch = len(orig_mask[0])
    # Mask is 1 for unstable neurons. Otherwise it's 0.
    mask = orig_mask
    reduce_op = get_branching_op(branching_reduceop)
    topk = branching_candidates

    score = []
    intercept_tb = []
    relu_idx = -1

    for layer in reversed(net.net.relus):
        ratio = lAs[relu_idx]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                   upper_bounds[pre_relu_indices[relu_idx]])
        # Intercept
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1
        intercept_tb.insert(0, intercept_candidate.view(batch, -1) * mask[relu_idx])

        # Bias
        input_node = layer.inputs[0]
        assert isinstance(input_node, (BoundConv, BoundLinear, BoundBatchNormalization, BoundAdd))
        if type(input_node) == BoundConv:
            if len(input_node.inputs) > 2:
                b_temp = input_node.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
            else:
                b_temp = 0
        elif type(input_node) == BoundLinear:
            # TODO: consider if no bias in the BoundLinear layer
            b_temp = input_node.inputs[-1].param.detach()
        elif type(input_node) == BoundAdd:
            b_temp = 0
            # print(input_node.inputs)
            for l in input_node.inputs:
                if type(l) == BoundConv:
                    if len(l.inputs) > 2:
                        b_temp += l.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
                if type(l) == BoundBatchNormalization:
                    b_temp += 0  # l.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1) # TODO
                if type(l) == BoundAdd:
                    for ll in l.inputs:
                        if type(ll) == BoundConv:
                            b_temp += ll.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
        else:
            b_temp = input_node.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1)  # for BN, bias is the -3th inputs

        # print(b_temp.shape, ratio_temp_0.shape, ratio.shape)
        b_temp = b_temp * ratio
        bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
        bias_candidate_2 = b_temp * ratio_temp_0
        bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)  # max for babsr by default
        score_candidate = bias_candidate + intercept_candidate
        score.insert(0, abs(score_candidate).view(batch, -1) * mask[relu_idx])

        relu_idx -= 1

    final_decision = []
    decision_tmp = {}
    tmp_ret = {}
    skip_layers = [0]

    # real batch = batch * 2, since we have two kinds of scores
    lbs = [torch.cat([i, i]) for i in lower_bounds]
    ups = [torch.cat([i, i]) for i in upper_bounds]
    if isinstance(slopes[0], dict):
        # per neuron slope.
        sps = slopes + slopes
    else:
        sps = [torch.cat([i, i]) for i in slopes]
    if use_beta:
        bs = [torch.cat([i, i]) for i in betas]
        history += history

    set_slope = True  # We only set the slope once.
    for i in range(1, len(score)):
        if (score[i].max(1).values <= 1e-4).all() and (intercept_tb[i].min(1).values >= -1e-4).all():
            print('{}th layer has no valid scores'.format(i))
            skip_layers.append(i)
            continue

        score_idx = torch.topk(score[i], topk)
        score_idx_indices = score_idx.indices.cpu()
        itb_idx = torch.topk(intercept_tb[i], topk, largest=False)
        itb_idx_indices = itb_idx.indices.cpu()

        k_ret = torch.empty(size=(topk, batch*2), device=lower_bounds[0].device, requires_grad=False)
        k_decision = []
        for k in range(topk):
            decision_index = score_idx_indices[:, k]
            decision_max_ = [[i, j.item()] for j in decision_index]  # add decision_index with layer's idx
            decision_index = itb_idx_indices[:, k]
            decision_min_ = [[i, j.item()] for j in decision_index]

            k_decision.append(decision_max_ + decision_min_)

            # only save the best lower bounds of the two splits
            if use_beta:
                k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], early_stop=False, betas=bs,
                                                       layer_set_bound=True, shortcut=True, history=history)
            else:
                k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], early_stop=False, beta=False,
                                                       layer_set_bound=True, shortcut=True)
            # No need to set slope next time; we do not optimize the slopes.
            set_slope = False
            # print(f'layer {i} k {k} decision {k_decision[-1]} bounds {k_ret_lbs.squeeze(-1).cpu()}')

            mask_score = (score_idx.values[:, k] <= 1e-4).float()  # build mask indicates invalid scores (stable neurons), batch wise, 1: invalid
            mask_itb = (itb_idx.values[:, k] >= -1e-4).float()
            # make the invalid lower bounds worse than normal lower bounds by minus 999999
            # we only consider the best lower bound across two splits by using min(0)
            k_ret[k] = reduce_op((k_ret_lbs.view(-1) - torch.cat([mask_score, mask_itb]).repeat(2) * 999999).reshape(2, -1), dim=0).values

        i_idx = k_ret.max(0)  # compare across topK
        tmp_ret[i] = i_idx.values
        decision_tmp[i] = [k_decision[i_idx.indices[ii]][ii] for ii in range(batch*2)]

    if len(tmp_ret) > 0:
        max_ret = torch.max(torch.stack([i for i in tmp_ret.values()]), dim=0)  # compare across layers
        rets, decision_layers = max_ret.values.cpu().numpy(), max_ret.indices.cpu().numpy()  # first batch: score; second batch: intercept_tb.

        # add index number for the skipped layers
        # for _, g in groupby(enumerate(skip_layers), lambda ix: ix[0] - ix[1]):
        #     decision_layers[decision_layers >= list(g)[-1][-1]] += 1
        for s in skip_layers:
            decision_layers[decision_layers >= s] += 1

        for b in range(batch):
            if max([i[b].max() for i in score]) > 1e-4 and min([i[b].min() for i in intercept_tb]) < -1e-4 \
                    and max(rets[b], rets[b + batch]) > -10000:  # make sure this potential split is valid
                if rets[b] > rets[b+batch]:  # score > intercept_tb
                    final_decision.append(decision_tmp[decision_layers[b].item()][b])
                else:
                    final_decision.append(decision_tmp[decision_layers[b+batch].item()][b+batch])
            else:
                # print('\t using a random choice')
                mask_item = [m[b] for m in mask]
                for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                    if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                        final_decision.append([preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                        break
    else:
        # all layers are split or has no improvement
        for b in range(batch):
            # print('\t using a random choice')
            mask_item = [m[b] for m in mask]
            # for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):  # random
            for preferred_layer in reversed(range(len(pre_relu_indices))):  # from last layer to first layer
                if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                    final_decision.append(
                        [preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                    break

    return final_decision



def branching_scores_kfsb(lower_bounds, upper_bounds, net, pre_relu_indices, lAs, batch,
                          mask, reduce_op, number_bounds, prioritize_slopes='none'):
    """Compute branching scores for kfsb."""
    score = []
    intercept_tb = []
    relu_idx = -1
    small_score_threshold = 1e-4
    big_constant = 1e6

    def normalize_scores(scores, normal_score_idx, reduced_score_idx, larger_is_better=True):
        #  We want to reduce all scores in the reduced_score_idx set, so they are no better than the scores in the normal_score_idx set.
        if larger_is_better:
            thresh = small_score_threshold
            get_best_score = lambda candidate, idx: torch.max(candidate * idx, dim=1).values  # idx is a mask, setting irrelevant scores to 0. Valid scores are positive.
            get_worst_score = lambda candidate, idx: torch.min(candidate * idx + (1.0 - idx) * big_constant, dim=1).values  # Setting irrelevant scores to inf.
            clamp_score = lambda candidate: torch.clamp_min(candidate, thresh)
        else:
            thresh = -small_score_threshold
            get_best_score = lambda candidate, idx: torch.min(candidate * idx, dim=1).values  # idx is a mask, setting irrelevant scores to 0. Valid scores are negative.
            get_worst_score = lambda candidate, idx: torch.max(candidate * idx - (1.0 - idx) * big_constant, dim=1).values  # Setting irrelevant scores to -inf.
            clamp_score = lambda candidate: torch.clamp_max(candidate, thresh)
        # Sizes are (batch,).
        best_score_in_reduced_set = get_best_score(scores, reduced_score_idx)
        worst_score_in_normal_set = clamp_score(get_worst_score(scores, normal_score_idx))
        ratio = torch.clamp_max(worst_score_in_normal_set / (best_score_in_reduced_set + thresh), 1.0)
        # Make the scores in the reduced_score_idx set smaller.
        adjusted_scores = scores * normal_score_idx + scores * reduced_score_idx * ratio.unsqueeze(1)
        return adjusted_scores

    # Compute BaBSR scores, starting from the last layer.
    for layer_i, layer in enumerate(reversed(net.net.relus)):
        this_layer_mask = mask[pre_relu_indices[relu_idx]].unsqueeze(1)
        # print('lAs[pre_relu_indices[relu_idx]]:', lAs[pre_relu_indices[relu_idx]].shape)
        # print('this_layer_mask:', this_layer_mask.shape)
        if prioritize_slopes == 'positive':
            # Prioritize splits with only positive lA.
            normal_score_mask = (lAs[pre_relu_indices[relu_idx]] >= 0).view(batch, number_bounds, -1) * this_layer_mask
            reduced_score_mask = (lAs[pre_relu_indices[relu_idx]] < 0).view(batch, number_bounds, -1) * this_layer_mask
        elif prioritize_slopes == 'negative':
            # Prioritize splits with only positive lA.
            normal_score_mask = (lAs[pre_relu_indices[relu_idx]] <= 0).view(batch, number_bounds, -1) * this_layer_mask
            reduced_score_mask = (lAs[pre_relu_indices[relu_idx]] > 0).view(batch, number_bounds, -1) * this_layer_mask
        elif prioritize_slopes != 'none':
            raise ValueError(f'Unknown prioritize_slopes parameter {prioritize_slopes}')

        ratio = lAs[pre_relu_indices[relu_idx]]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                   upper_bounds[pre_relu_indices[relu_idx]])

        # Intercept score, used as a backup score in BaBSR. A lower (more negative) score is better.
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1.unsqueeze(1)
        reshaped_intercept_candidate = intercept_candidate.view(batch, number_bounds, -1) * this_layer_mask
        reshaped_intercept_candidate = reshaped_intercept_candidate.mean(1)   # mean over number_bounds dim
        if prioritize_slopes != 'none':
            adjusted_intercept_candidate = normalize_scores(reshaped_intercept_candidate, normal_score_mask, reduced_score_mask, larger_is_better=False)
        else:
            adjusted_intercept_candidate = reshaped_intercept_candidate
        # intercept_tb is a list of intercept scores, each with a array of (batch, neuron).
        intercept_tb.insert(0, adjusted_intercept_candidate)

        # Bias
        input_node = layer.inputs[0]
        assert isinstance(input_node, (BoundConv, BoundLinear, BoundBatchNormalization, BoundAdd))
        if type(input_node) == BoundConv:
            if len(input_node.inputs) > 2:
                b_temp = input_node.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
            else:
                b_temp = 0
        elif type(input_node) == BoundLinear:
            # TODO: consider if no bias in the BoundLinear layer
            b_temp = input_node.inputs[-1].param.detach()
        elif type(input_node) == BoundAdd:
            b_temp = 0
            # print(input_node.inputs)
            for l in input_node.inputs:
                if type(l) == BoundConv:
                    if len(l.inputs) > 2:
                        b_temp += l.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
                if type(l) == BoundBatchNormalization:
                    b_temp += 0  # l.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1) # TODO
                if type(l) == BoundAdd:
                    for ll in l.inputs:
                        if type(ll) == BoundConv:
                            b_temp += ll.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
        else:
            # for BN, bias is the -3th inputs
            b_temp = input_node.inputs[-3].param.detach().view(-1, *([1] * (ratio.ndim - 3)))

        b_temp = b_temp * ratio
        # Estimated bounds of the two sides of the bounds.
        ratio_temp_0 = ratio_temp_0.unsqueeze(1)
        bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
        bias_candidate_2 = b_temp * ratio_temp_0
        bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)  # max for babsr by default
        score_candidate = bias_candidate + intercept_candidate
        score_candidate = score_candidate.abs().view(batch, number_bounds, -1) * this_layer_mask
        score_candidate = score_candidate.mean(1)  # mean over number_bounds dim
        if prioritize_slopes != 'none':
            adjusted_score_candidate = normalize_scores(score_candidate, normal_score_mask, reduced_score_mask, larger_is_better=True)
            remaining_branches = normal_score_mask.sum(dim=1, dtype=torch.int32)
            print(f'layer {len(net.net.relus) - layer_i} remaining preferred branching variables: {remaining_branches[:10].tolist()}, avg {remaining_branches.sum().item() / remaining_branches.numel()}')
        else:
            adjusted_score_candidate = score_candidate
        # Slope score, the main score in BaBSR. A higher (more positive) score is batter.
        score.insert(0, adjusted_score_candidate)

        relu_idx -= 1

    return score, intercept_tb



@torch.no_grad()
def choose_node_parallel_kFSB(lower_bounds, upper_bounds, orig_mask, net, pre_relu_indices, lAs, branching_candidates=5, branching_reduceop='min', slopes=None, betas=None, history=None, use_beta=False, keep_all_decision=False, prioritize_slopes='none'):

    batch = len(orig_mask[0])
    mask = orig_mask # Mask is 1 for unstable neurons. Otherwise it's 0.
    reduce_op = get_branching_op(branching_reduceop)
    topk = branching_candidates
    number_bounds = net.c.shape[1]

    score, intercept_tb = branching_scores_kfsb(lower_bounds, upper_bounds, net, pre_relu_indices, lAs, batch, mask, reduce_op, number_bounds, prioritize_slopes)
    
    final_decision = []

    # real batch = batch * 2, since we have two kinds of scores.
    lbs = [torch.cat([i, i]) for i in lower_bounds]
    ups = [torch.cat([i, i]) for i in upper_bounds]
    if isinstance(slopes[0], dict):
        # per neuron slope.
        sps = slopes + slopes
    else:
        sps = [torch.cat([i, i]) for i in slopes]
    if use_beta:
        bs = [torch.cat([i, i]) for i in betas]
        history += history

    # Use score_length to convert an index to its layer and offset.
    score_length = np.cumsum([len(score[i][0]) for i in range(len(score))])
    score_length = np.insert(score_length, 0, 0)

    # Flatten the scores vector.
    all_score = torch.cat(score, dim=1)
    all_itb = torch.cat(intercept_tb, dim=1)
    # Select top-k candidates among all layers for two kinds of scores.
    score_idx = torch.topk(all_score, topk)
    # These indices are the indices for the top-k scores in flatten
    score_idx_indices = score_idx.indices.cpu()
    itb_idx = torch.topk(all_itb, topk, largest=False)  # k-smallest elements.
    itb_idx_indices = itb_idx.indices.cpu()

    k_decision = []
    k_ret = torch.empty(size=(topk, batch * 2), device=lower_bounds[0].device, requires_grad=False)
    set_slope = True  # We only set the slope once.
    for k in range(topk):
        # top-k candidates from the slope scores.
        decision_index = score_idx_indices[:, k]
        # Find which layer and neuron this topk gradient belongs to.
        decision_max_ = []
        for l in decision_index:
            # Go over each element in this batch.
            l = l.item()
            # Recover the (layer, idx) from the flattend array.
            layer = np.searchsorted(score_length, l, side='right') - 1
            idx = l - score_length[layer]
            decision_max_.append([layer, idx])

        # top-k candidates from the intercept (backup) scores.
        decision_index = itb_idx_indices[:, k]
        # Find which layer and neuron this topk gradient belongs to.
        decision_min_ = []
        for l in decision_index:
            # Go over each element in this batch.
            l = l.item()
            layer = np.searchsorted(score_length, l, side='right') - 1
            idx = l - score_length[layer]
            decision_min_.append([layer, idx])

        # Stores the top-k decisions, so after finding the max we can go back to see which decision it is.
        k_decision.append(decision_max_ + decision_min_)

        # only save the best lower bounds of the two splits
        if use_beta:
            k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], early_stop=False, betas=bs, shortcut=True, history=history)
        else:
            k_ret_lbs = net.update_bounds_parallel(lbs, ups, k_decision[-1], sps if set_slope else [], early_stop=False, beta=False, shortcut=True)
        # print(f'k {k} decision {k_decision[-1]} bounds {k_ret_lbs.squeeze(-1).cpu()}')
        # No need to set slope next time; we do not optimize the slopes.
        rhs = net.rhs.repeat(k_ret_lbs.shape[0], 1)
        k_ret_lbs = (k_ret_lbs - rhs).max(-1).values
        set_slope = False

        mask_score = (score_idx.values[:, k] <= 1e-4).float()  # build mask indicates invalid scores (stable neurons), batch wise, 1: invalid
        mask_itb = (itb_idx.values[:, k] >= -1e-4).float()
        # We first make the invalid lower bounds worse than normal lower bounds by minus 999999.
        # Then we consider the best lower bound across two splits (in the first dimension after reshape) by using min(0) or max(0).
        k_ret[k] = reduce_op((k_ret_lbs.view(-1) - torch.cat([mask_score, mask_itb]).repeat(2) * 999999).reshape(2, -1), dim=0).values

    # k_ret has shape (top-k, batch) and we take the score eveluated using bound propagation based on the top-k choice.
    i_idx = k_ret.max(0)
    rets = i_idx.values.cpu().numpy()
    rets_indices = i_idx.indices.cpu().numpy()
    # Given the indices of the max score, find what is its corresponding decision.
    decision_tmp = [k_decision[rets_indices[ii]][ii] for ii in range(batch*2)]

    if not keep_all_decision:
        # regular kfsb, select the top 1 decision from k
        random_decision_list = []
        for b in range(batch):
            if max(rets[b], rets[b + batch]) > -10000:  # make sure this potential split is valid
                if rets[b] > rets[b + batch]:  # score > intercept_tb
                    final_decision.append(decision_tmp[b])
                else:
                    final_decision.append(decision_tmp[b + batch])
            else:
                # No valid scores, have to choose a neuron randomly.
                random_decision_list.append(b)
                mask_item = [m[b] for m in mask]
                random_decision = [None, None]
                for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                    if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                        random_decision = [preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()]
                        break
                assert random_decision[0] is not None
                final_decision.append(random_decision)
        if len(random_decision_list):
            logger.info(f'Random branching decision used for {len(random_decision_list)} domain(s)')
        return final_decision
    else:
        # keep all the k decisions
        # final_decision: batch -> k splits
        final_decision = [[] for _ in range(batch)]
        random_decision_dict = {}
        # customize kfsb but sometimes duplicate split for k_decision[ki][b] and k_decision[ki][b+batch]
        for b in range(batch):
            # use mask to check if a node is unstable when random selection
            # not detach for now, please check!!!
            mask_item = [m[b] for m in mask]
            for ki in range(topk):
                if max(k_ret[ki][b], k_ret[ki][b + batch]) > -10000:
                    if k_ret[ki][b] > k_ret[ki][b + batch] and mask_item[k_decision[ki][b][0]][k_decision[ki][b][1]] == 1:
                        decision = k_decision[ki][b]
                    else:
                        decision = k_decision[ki][b + batch]
                    if mask_item[decision[0]][decision[1]] == 1:
                        final_decision[b].append(decision)
                        # print(decision, mask_item[decision[0]][decision[1]])
                        # assert mask_item[decision[0]][decision[1]] == 1, "selected decision node should be unstable!"
                        mask_item[decision[0]][decision[1]] = 0
            
            if len(final_decision[b]) < topk:
                # No valid scores, have to choose a neuron randomly.
                random_decision_dict[b] = topk - len(final_decision[b])
                for i in range(topk - len(final_decision[b])):
                    for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                        if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                            final_decision[b].append([preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                            mask_item[final_decision[b][-1][0]][final_decision[b][-1][1]] = 0
                            break
            assert len(final_decision[b]) <= topk, f"{len(final_decision[b])} <= {topk}"
            # del mask_item
        if random_decision_dict:
            logger.info(f'Random branching decision used for {len(random_decision_dict)} domain(s)')
        return final_decision

