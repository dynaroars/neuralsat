import torch

def serialize_specs(x, cs, rhs):
    assert x.shape[0] == 1, print(f'Invalid batch size: {x.shape[0]}')
    cs_mat, rhs_mat, cond_mat = [], [], []
    for cs_i, rhs_i in zip(cs, rhs):
        cs_mat.append(torch.Tensor(cs_i).to(x.device))
        rhs_mat.append(torch.Tensor(rhs_i).to(x.device))
        cond_mat.append(rhs_i.shape[0]) # mark the `and` group
    cs_mat = torch.cat(cs_mat, dim=0).unsqueeze(0).to(x.device)
    rhs_mat = torch.cat(rhs_mat, dim=0).unsqueeze(0).to(x.device)
    return cs_mat, rhs_mat, [cond_mat]
    
    
def check_adv_multi(input, output, serialized_conditions, data_max, data_min):
    cs_mat, rhs_mat, cond_mat = serialized_conditions
    if len(cond_mat[0]) == rhs_mat.shape[-1]:
        assert all([cond_mat[0][0] == i for i in cond_mat[0]])
        cs_mat = cs_mat.view(cs_mat.shape[0], 1, len(cond_mat[0]), -1, cs_mat.shape[-1])
        # [batch_size, restarts, num_or_spec, num_and_spec, output_dim]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, len(cond_mat[0]), -1)
        cond = torch.matmul(cs_mat, output.unsqueeze(-1)).squeeze(-1) - rhs_mat

        valid = ((input <= data_max) & (input >= data_min))
        valid = valid.view(*valid.shape[:3], -1)
        # [num_example, restarts, num_all_spec, output_dim]
        valid = valid.all(-1).view(valid.shape[0], valid.shape[1], len(cond_mat[0]), -1)
        # [num_example, restarts, num_or_spec, num_and_spec]
        res = ((cond.amax(dim=-1, keepdim=True) < 0.0) & valid).any(dim=-1).any(dim=-1).any(dim=-1)  
    else:
        output = output.repeat_interleave(torch.tensor(cond_mat[0]).to(output.device), dim=2)
        # [num_example, num_restarts, num_spec, num_output]

        cs_mat = cs_mat.view(cs_mat.shape[0], 1, -1, cs_mat.shape[-1])
        # [num_example, 1, num_spec, num_output]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, -1)
        # [num_example, 1, num_spec]

        cond = torch.clamp((cs_mat * output).sum(-1) - rhs_mat, min=0.0)
        # [num_example, 1, num_spec]

        group_C = torch.zeros(len(cond_mat[0]), cs_mat.shape[2], device=cond.device, dtype=cond.dtype) # [num_or_spec, num_total_spec]
        x_index = []
        y_index = []
        index = 0
        
        for i, num_cond in enumerate(cond_mat[0]):
            x_index.extend([i] * num_cond)
            y_index.extend([index+j] for j in range(num_cond))
            index += num_cond

        group_C[x_index, y_index] = 1.0

        # loss shape: [batch_size, num_restarts, num_total_spec]
        cond = group_C.matmul(cond.unsqueeze(-1)).squeeze(-1)

        valid = ((input <= data_max) & (input >= data_min))
        valid = valid.view(*valid.shape[:3], -1)
        # [num_example, restarts, num_all_spec, output_dim]
        valid = valid.all(-1).view(valid.shape[0], valid.shape[1], len(cond_mat[0]), -1)
        # [num_example, restarts, num_or_spec, num_and_spec]

        valid = valid.all(-1)

        # [num_example, num_restarts, num_or_example]
        res = ((cond == 0.0) & valid).any(dim=-1).any(dim=-1)

    return res.all()


def get_loss(origin_out, output, serialized_conditions, gama_lambda=0, threshold=-1e-5):
    cs_mat, rhs_mat, cond_mat = serialized_conditions
    if rhs_mat.shape[-1] == len(cond_mat[0]):
        assert all([cond_mat[0][0] == i for i in cond_mat[0]])
        cs_mat = cs_mat.view(cs_mat.shape[0], 1, output.shape[2], -1, cs_mat.shape[-1])
        # [num_example, 1, num_or_spec, num_and_spec, num_output]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, output.shape[2], -1)
        loss = cs_mat.matmul(output.unsqueeze(-1)).squeeze(-1) - rhs_mat
        loss = torch.clamp(loss, min=threshold)
        # [num_example, num_restarts, num_or_spec, num_and_spec]
        loss = -loss
    else:
        output = output.repeat_interleave(torch.tensor(cond_mat[0]).to(output.device), dim=2)
        if origin_out is not None:
            origin_out = origin_out.repeat_interleave(torch.tensor(cond_mat[0]).to(output.device), dim=2)
        # [num_example, num_restarts, num_spec, num_output]

        cs_mat = cs_mat.view(cs_mat.shape[0], 1, -1, cs_mat.shape[-1])
        # [num_example, 1, num_spec, num_output]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, -1)
        # [num_example, 1, num_spec]

        loss = (cs_mat * output).sum(-1) - rhs_mat
        loss = torch.clamp(loss, min=threshold)
        loss = -loss

    if origin_out is not None:
        loss_gamma = loss.sum() + (gama_lambda * (output - origin_out)**2).sum(dim=3).sum()
    else:
        loss_gamma = loss.sum()
    # [num_example, num_restarts, num_or_spec, num_and_spec]
    return loss_gamma
