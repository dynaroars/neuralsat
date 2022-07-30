import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from modules import Flatten

Icp_score_counter = 0


def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound - F.relu(lower_bound)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio

    # if (lower >= 0):
    #    decision = [1,0]
    # elif (upper <=0):
    #    decision = [0, 0]
    # else:
    #    temp = upper/(upper - lower)
    #    decision = [temp, -temp*lower]
    # return decision
    return slope_ratio, intercept


def choose_node_conv(lower_bounds, upper_bounds, orig_mask, layers, pre_relu_indices, icp_score_counter, random_order,
                     sparsest_layer, decision_threshold=0.001, gt=False):
    """
    choose the dimension to split on
    based on each node's contribution to the cost function
    in the KW formulation.

    sparsest_layer: if all layers are dense, set it to -1
    decision_threshold: if the maximum score is below the threshold,
                        we consider it to be non-informative
    random_order: priority to each layer when making a random choice
                  with preferences. Increased preference for later elements                  in the list

    """

    mask = [(i == -1).float().view(-1) for i in orig_mask]
    score = []
    intercept_tb = []
    random_choice = random_order.copy()

    ratio = torch.ones(1).to(lower_bounds[0].device)
    # starting from 1, back-propogating: if the weight is negative
    # introduce bias; otherwise, intercept is 0
    # we are only interested in two terms for now: the slope x bias of the node
    # and bias x the amount of argumentation introduced by later layers.
    # From the last relu-containing layer to the first relu-containing layer

    # Record score in a dic
    # new_score = {}
    # new_intercept = {}
    relu_idx = -1

    for layer_idx, layer in reversed(list(enumerate(layers))):
        if type(layer) is nn.Linear:
            ratio = ratio.unsqueeze(-1)
            w_temp = layer.weight.detach()
            ratio = torch.t(w_temp) @ ratio
            ratio = ratio.view(-1)
            # import pdb; pdb.set_trace()

        elif type(layer) is nn.ReLU:
            # compute KW ratio
            ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                       upper_bounds[pre_relu_indices[relu_idx]])
            # Intercept
            intercept_temp = torch.clamp(ratio, max=0)
            intercept_candidate = intercept_temp * ratio_temp_1
            intercept_tb.insert(0, intercept_candidate.view(-1) * mask[relu_idx])

            # Bias
            b_temp = layers[layer_idx - 1].bias.detach()
            if type(layers[layer_idx - 1]) is nn.Conv2d:
                b_temp = b_temp.unsqueeze(-1).unsqueeze(-1)
            ratio_1 = ratio * (ratio_temp_0 - 1)
            bias_candidate_1 = b_temp * ratio_1
            ratio = ratio * ratio_temp_0
            bias_candidate_2 = b_temp * ratio
            bias_candidate = torch.max(bias_candidate_1, bias_candidate_2)
            # test = (intercept_candidate!=0).float()
            # ???if the intercept_candiate at a node is 0, we should skip this node
            #    (intuitively no relaxation triangle is introduced at this node)
            #    score_candidate = test*bias_candidate + intercept_candidate
            score_candidate = bias_candidate + intercept_candidate
            score.insert(0, abs(score_candidate).view(-1) * mask[relu_idx])

            relu_idx -= 1

        elif type(layer) is nn.Conv2d:
            # import pdb; pdb.set_trace()
            ratio = ratio.unsqueeze(0)
            ratio = F.conv_transpose2d(ratio, layer.weight, stride=layer.stride, padding=layer.padding)
            ratio = ratio.squeeze(0)

        elif type(layer) is Flatten:
            # import pdb; pdb.set_trace()
            ratio = ratio.reshape(lower_bounds[layer_idx].size())
        else:
            raise NotImplementedError

    max_info = [torch.max(i, 0) for i in score]
    decision_layer = max_info.index(max(max_info))
    decision_index = max_info[decision_layer][1].item()
    if decision_layer != sparsest_layer and max_info[decision_layer][0].item() > decision_threshold:
        # temp = torch.zeros(score[decision_layer].size())
        # temp[decision_index]=1
        # decision_index = torch.nonzero(temp.reshape(mask[decision_layer].shape))[0].tolist()
        decision = [decision_layer, decision_index]

    else:
        min_info = [[i, torch.min(intercept_tb[i], 0)] for i in range(len(intercept_tb)) if
                    torch.min(intercept_tb[i]) < -1e-4]
        # import pdb; pdb.set_trace()
        if len(min_info) != 0 and icp_score_counter < 2:
            intercept_layer = min_info[-1][0]
            intercept_index = min_info[-1][1][1].item()
            icp_score_counter += 1
            # inter_temp = torch.zeros(intercept_tb[intercept_layer].size())
            # inter_temp[intercept_index]=1
            # intercept_index = torch.nonzero(inter_temp.reshape(mask[intercept_layer].shape))[0].tolist()
            decision = [intercept_layer, intercept_index]
            if intercept_layer != 0:
                icp_score_counter = 0
            print('\tusing intercept score')
        else:
            print('\t using a random choice')
            undecided = True
            while undecided:
                preferred_layer = random_choice.pop(-1)
                if len(mask[preferred_layer].nonzero()) != 0:
                    decision = [preferred_layer, mask[preferred_layer].nonzero()[0].item()]
                    undecided = False
                else:
                    pass
            icp_score_counter = 0
    if gt is False:
        return decision, icp_score_counter
    else:
        return decision, icp_score_counter, score


def choose_node_parallel(lower_bounds, upper_bounds, orig_mask, layers, pre_relu_indices, random_order,
                         sparsest_layer, decision_threshold=0.001, gt=False, batch=5):
    batch = min(batch, len(orig_mask[0]))
    mask = [(i == -1).float() for i in orig_mask]

    score = []
    intercept_tb = []

    ratio = torch.ones((batch, 1), device=lower_bounds[0].device)
    relu_idx = -1

    for layer_idx, layer in reversed(list(enumerate(layers))):
        if type(layer) is nn.Linear:
            # ratio = ratio.unsqueeze(-1)
            w_temp = layer.weight.detach()
            ratio = ratio.matmul(w_temp)
            # ratio = ratio.view(-1)
            # import pdb; pdb.set_trace()

        elif type(layer) is nn.ReLU:
            # compute KW ratio
            ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]],
                                                       upper_bounds[pre_relu_indices[relu_idx]])
            # Intercept
            intercept_temp = torch.clamp(ratio, max=0)
            intercept_candidate = intercept_temp * ratio_temp_1
            intercept_tb.insert(0, intercept_candidate.view(batch, -1) * mask[relu_idx])

            # Bias
            b_temp = layers[layer_idx - 1].bias.detach()
            if type(layers[layer_idx - 1]) is nn.Conv2d:
                b_temp = b_temp.unsqueeze(-1).unsqueeze(-1)
            ratio_1 = ratio * (ratio_temp_0 - 1)
            bias_candidate_1 = b_temp * ratio_1
            ratio = ratio * ratio_temp_0
            bias_candidate_2 = b_temp * ratio
            bias_candidate = torch.max(bias_candidate_1, bias_candidate_2)
            # test = (intercept_candidate!=0).float()
            # ???if the intercept_candiate at a node is 0, we should skip this node
            #    (intuitively no relaxation triangle is introduced at this node)
            #    score_candidate = test*bias_candidate + intercept_candidate
            score_candidate = bias_candidate + intercept_candidate
            score.insert(0, abs(score_candidate).view(batch, -1) * mask[relu_idx])

            relu_idx -= 1

        elif type(layer) is nn.Conv2d:
            # import pdb; pdb.set_trace()
            # ratio = ratio.unsqueeze(0)
            ratio = F.conv_transpose2d(ratio, layer.weight, stride=layer.stride, padding=layer.padding)
            # ratio = ratio.squeeze(0)

        elif type(layer) is Flatten:
            # import pdb; pdb.set_trace()
            ratio = ratio.reshape(lower_bounds[layer_idx].size())
        elif type(layer) is nn.ZeroPad2d:
            pass

        else:
            raise NotImplementedError

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
                print('\t using a random choice')
                mask_item = [m[b] for m in mask]
                for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                    if len(mask_item[preferred_layer].nonzero()) != 0:
                        decision.append([preferred_layer, mask_item[preferred_layer].nonzero()[0].item()])
                        break
                Icp_score_counter = 0

    if gt is False:
        return decision
    else:
        return decision, score


