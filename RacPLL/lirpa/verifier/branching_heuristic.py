import torch

from auto_lirpa.operators import *

Icp_score_counter = 0


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

def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound.clamp(max=0)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio
    return slope_ratio, intercept

@torch.no_grad()
def choose_node_parallel_crown(lower_bounds, upper_bounds, orig_mask, net, pre_relu_indices, lAs, sparsest_layer=0, decision_threshold=0.001, batch=5, branching_reduceop='min'):
    batch = min(batch, len(orig_mask[0]))

    mask = orig_mask # 1 for unstable neurons. Otherwise it's 0.
    
    reduce_op = get_branching_op(branching_reduceop)

    score = []
    intercept_tb = []
    relu_idx = -1

    for layer in reversed(net.net.relus):
        ratio = lAs[relu_idx]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[pre_relu_indices[relu_idx]], upper_bounds[pre_relu_indices[relu_idx]])

        # Intercept
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1
        intercept_tb.insert(0, intercept_candidate.view(batch, -1) * mask[relu_idx])

        # Bias
        input_node = layer.inputs[0]
        if isinstance(input_node, BoundedConv):
            if len(input_node.inputs) > 2:
                b_temp = input_node.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
            else:
                b_temp = 0

        elif isinstance(input_node, BoundedLinear):
            # FIXME: consider if no bias in the BoundLinear layer
            b_temp = input_node.inputs[-1].param.detach()

        elif isinstance(input_node, BoundedAdd):
            raise NotImplementedError

        else:
            b_temp = input_node.inputs[-3].param.detach().unsqueeze(-1).unsqueeze(-1)  # for BN, bias is the -3th inputs

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
            min_info = [[i, torch.min(intercept_tb[i][b], 0)] for i in range(len(intercept_tb)) if torch.min(intercept_tb[i][b]) < -1e-4]

            global Icp_score_counter
            if len(min_info) != 0 and Icp_score_counter < 2:
                intercept_layer = min_info[-1][0]
                intercept_index = min_info[-1][1][1].item()
                Icp_score_counter += 1
                decision.append([intercept_layer, intercept_index])
                if intercept_layer != 0:
                    Icp_score_counter = 0
            else:
                mask_item = [m[b] for m in mask]
                for preferred_layer in np.random.choice(len(pre_relu_indices), len(pre_relu_indices), replace=False):
                    if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                        decision.append([preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                        break
                Icp_score_counter = 0

    return decision






