from torch.optim import Optimizer
import torch
import math

from util.misc.adam_clipping import AdamClipping

def preprocess_spec(spec, input_shape, device):
    list_target_label_arrays = [[]]
    data_min_repeat = []
    data_max_repeat = []

    data_range = torch.Tensor(spec[0])
    spec_num = len(spec[1])

    data_max_ = data_range[:,1].view(-1, *input_shape[1:]).to(device).expand(spec_num, *input_shape[1:]).unsqueeze(0)
    data_min_ = data_range[:,0].view(-1, *input_shape[1:]).to(device).expand(spec_num, *input_shape[1:]).unsqueeze(0)

    data_max_repeat.append(data_max_)
    data_min_repeat.append(data_min_)


    list_target_label_arrays[0].extend(list(spec[1]))

    data_min_repeat = torch.cat(data_min_repeat, dim=1)
    data_max_repeat = torch.cat(data_max_repeat, dim=1)

    return list_target_label_arrays, data_min_repeat, data_max_repeat



def build_conditions(x, list_target_label_arrays):
    '''
    parse C_mat, rhs_mat from the target_label_arrays
    '''
    batch_size = x.shape[0]

    cond_mat = [[] for _ in range(batch_size)]
    C_mat = [[] for _ in range(batch_size)]
    rhs_mat = [[] for _ in range(batch_size)]

    same_number_const = True
    const_num = None
    for i in range(batch_size):
        target_label_arrays = list_target_label_arrays[i]
        for prop_mat, prop_rhs in target_label_arrays:
            C_mat[i].append(torch.Tensor(prop_mat).to(x.device))
            rhs_mat[i].append(torch.Tensor(prop_rhs).to(x.device))
            cond_mat[i].append(prop_rhs.shape[0]) # mark the `and` group
            if const_num is not None and prop_rhs.shape[0] != const_num:
                same_number_const = False
            else:
                const_num = prop_rhs.shape[0]
                
        C_mat[i] = torch.cat(C_mat[i], dim=0).unsqueeze(0)
        rhs_mat[i] = torch.cat(rhs_mat[i], dim=0).unsqueeze(0)

        # C: [1, num_spec, num_output]
    try:
        # try to stack the specs for a batch of examples
        # C: [num_example, num_spec, num_output]
        C_mat = torch.cat(C_mat, dim=0)
        rhs_mat = torch.cat(rhs_mat, dim=0)
    except (RuntimeError, ValueError):
        # failed when the examples have different number of specs
        print("Only support batches when the examples have the same number of constraints.")
        assert False
    # C shape: [num_example, num_spec, num_output]
    # rhs shape: [num_example, num_spec]
    # cond_mat shape: [num_example, num_spec]

    return C_mat, rhs_mat, cond_mat, same_number_const
    
def test_conditions(input, output, C_mat, rhs_mat, cond_mat, same_number_const, data_max, data_min):
    '''
    Whether the output satisfies the specifiction conditions.
    If the output satisfies the specification for adversarial examples, this function returns True, otherwise False.

    input: [num_exampele, num_restarts, num_or_spec, *input_shape]
    output: [num_example, num_restarts, num_or_spec, num_output]
    C_mat: [num_example, num_restarts, num_spec, num_output] or [num_example, num_spec, num_output]
    rhs_mat: [num_example, num_spec]
    cond_mat: [[]] * num_examples
    same_number_const (bool): if same_number_const is True, it means that there are same number of and specifications in every or specification group.
    data_max & data_min: [num_example, num_spec, *input_shape]
    '''

    if same_number_const:
        C_mat = C_mat.view(C_mat.shape[0], 1, len(cond_mat[0]), -1, C_mat.shape[-1])
        # [batch_size, restarts, num_or_spec, num_and_spec, output_dim]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, len(cond_mat[0]), -1)

        cond = torch.matmul(C_mat, output.unsqueeze(-1)).squeeze(-1) - rhs_mat

        valid = ((input <= data_max) & (input >= data_min))

        valid = valid.view(*valid.shape[:3], -1)
        # [num_example, restarts, num_all_spec, output_dim]
        valid = valid.all(-1).view(valid.shape[0], valid.shape[1], len(cond_mat[0]), -1)
        # [num_example, restarts, num_or_spec, num_and_spec]
        
        res = ((cond.amax(dim=-1, keepdim=True) < 0.0) & valid).any(dim=-1).any(dim=-1).any(dim=-1)    
    else:
        output = output.repeat_interleave(torch.tensor(cond_mat[0]).to(output.device), dim=2)
        # [num_example, num_restarts, num_spec, num_output]

        C_mat = C_mat.view(C_mat.shape[0], 1, -1, C_mat.shape[-1])
        # [num_example, 1, num_spec, num_output]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, -1)
        # [num_example, 1, num_spec]

        cond = torch.clamp((C_mat * output).sum(-1) - rhs_mat, min=0.0)
        # [num_example, 1, num_spec]

        group_C = torch.zeros(len(cond_mat[0]), C_mat.shape[2], device=cond.device) # [num_or_spec, num_total_spec]
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

    return res


def build_loss(origin_out, output, C_mat, rhs_mat, cond_mat, same_number_const, gama_lambda=0, threshold=-1e-5, mode='hinge'):
    '''
    output: [num_example, num_restarts, num_or_spec, num_output]
    C_mat: [num_example, num_restarts, num_spec, num_output]
    rhs_mat: [num_example, num_spec]
    cond_mat: [[]] * num_examples
    gama_lambda: weight factor for gama loss. If true, sum the loss and return the sum of loss
    threshold: the threshold for hinge loss
    same_number_const (bool): if same_number_const is True, it means that there are same number of and specifications in every or specification group.
    '''

    if same_number_const:
        C_mat = C_mat.view(C_mat.shape[0], 1, output.shape[2], -1, C_mat.shape[-1])
        # [num_example, 1, num_or_spec, num_and_spec, num_output]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, output.shape[2], -1)
        loss = C_mat.matmul(output.unsqueeze(-1)).squeeze(-1) - rhs_mat
        loss = torch.clamp(loss, min=threshold)
        # [num_example, num_restarts, num_or_spec, num_and_spec]
        loss = -loss
    else:
        output = output.repeat_interleave(torch.tensor(cond_mat[0]).to(output.device), dim=2)
        if origin_out is not None:
            origin_out = origin_out.repeat_interleave(torch.tensor(cond_mat[0]).to(output.device), dim=2)
        # [num_example, num_restarts, num_spec, num_output]

        C_mat = C_mat.view(C_mat.shape[0], 1, -1, C_mat.shape[-1])
        # [num_example, 1, num_spec, num_output]
        rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, -1)
        # [num_example, 1, num_spec]

        loss = (C_mat * output).sum(-1) - rhs_mat
        loss = torch.clamp(loss, min=threshold)
        loss = -loss

    if origin_out is not None:
        loss_gamma = loss.sum() + (gama_lambda * (output - origin_out)**2).sum(dim=3).sum()
    else:
        loss_gamma = loss.sum()
    # [num_example, num_restarts, num_or_spec, num_and_spec]
    
    if mode == "sum":
        loss[loss >= 0] = 1.0
    # loss is returned for best loss selection, loss_gamma is for gradient descent.
    return loss, loss_gamma


def OSI_init_C(model, X, alpha, output_dim, iter_steps=50, lower_limit=0.0, upper_limit=1.0):
    # the general version of OSI initialization.
    input_shape = X.shape
    # [batch_size, num_restarts, num_or_spec, *X_shape[1:]]
    X_init = X.clone().detach()
    # [batch_size, num_restarts, num_or_spec, *X_shape[1:]]
    X_init = X_init.view(-1, *X_init.shape[3:])
    X = X.view(-1, *X.shape[3:])
    # [batch_size, * num_restarts * num_or_spec, *X_shape[1:]]

    w_d = (torch.rand([X.shape[0], output_dim], device=X.device) - 0.5) * 2

    assert torch.is_grad_enabled()

    for i in range(iter_steps):
        X_init = X_init.detach().requires_grad_()
        output = model(X_init)
        # test whether we need to early stop here.
        dot = torch.einsum('...,...->', w_d, output)
        dot.backward()

        with torch.no_grad():
            X_init = X_init + alpha * torch.sign(X_init.grad)
            X_init = X_init.view(input_shape)
            X_init = torch.max(torch.min(X_init, upper_limit), lower_limit)
            X_init = X_init.view(-1, *X_init.shape[3:])

    X_init = X_init.view(input_shape)
    X = X.view(input_shape)

    assert (X_init <= upper_limit).all()
    assert (X_init >= lower_limit).all()

    return X_init


def gen_adv_example(model_ori, x, best_deltas, data_max, data_min, C_mat, rhs_mat, cond_mat):
    # x and best_deltas has shape (batch, c, h, w).
    # data_min and data_max have shape (batch, spec, c, h, w).
    attack_image = torch.max(torch.min((x + best_deltas).unsqueeze(1), data_max), data_min)
    assert (attack_image >= data_min).all()
    assert (attack_image <= data_max).all()

    attack_output = model_ori(attack_image.view(-1, *x.shape[1:])).view(*attack_image.shape[:2], -1)
    # [batch_size, num_or_spec, out_dim]

    # only print out the first two random start outputs of the first two examples.
    # print("Adv example prediction (first 2 examples and 2 restarts):\n", attack_output[:2,:2])

    attack_output_repeat = attack_output.unsqueeze(1).repeat_interleave(torch.tensor(cond_mat[0]).to(x.device), dim=2)
    # [num_example, num_restarts, num_spec, num_output]

    C_mat = C_mat.view(C_mat.shape[0], 1, -1, C_mat.shape[-1])
    # [num_example, 1, num_spec, num_output]
    rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, -1)
    # [num_example, 1, num_spec]

    attack_margin = (C_mat * attack_output_repeat).sum(-1) - rhs_mat
    # [num_example, num_restarts, num_spec]

    # print("PGD attack margin (first 2 examles and 10 specs):\n", attack_margin[:2, :, :10])
    # print("number of violation: ", (attack_margin < 0).sum().item())

    return attack_image, attack_output, attack_margin




def translate_constraints_to_label(GT_specs):
    labels = []
    for and_list in GT_specs:
        label = None
        for or_list in and_list:
            if len(or_list) > 1:
                label = None
                break
            if label is None:
                label = or_list[0][0]
            elif label != or_list[0][0]:
                label = None
                break
        labels.append(label)
    return labels



def evaluate_cstr(constraints, net_out, torch_input=False):
    if len(net_out.shape) <= 1:
        net_out = net_out.reshape(1, -1)

    n_samp = net_out.shape[0]

    and_holds = (
        torch.ones(n_samp, dtype=bool, device=net_out.device)
        if torch_input
        else np.ones(n_samp, dtype=np.bool)
    )
    for or_list in constraints:
        or_holds = (
            torch.zeros(n_samp, dtype=bool, device=net_out.device)
            if torch_input
            else np.zeros(n_samp, dtype=np.bool)
        )
        for cstr in or_list:
            if cstr[0] == -1:
                or_holds = or_holds.__or__(cstr[2] > net_out[:, cstr[1]])
            elif cstr[1] == -1:
                or_holds = or_holds.__or__(net_out[:, cstr[0]] > cstr[2])
            else:
                or_holds = or_holds.__or__(
                    net_out[:, cstr[0]] - net_out[:, cstr[1]] > cstr[2]
                )
            if or_holds.all():
                break
        and_holds = and_holds.__and__(or_holds)
        if not and_holds.any():
            break
    return and_holds



def constraint_loss(logits, constraints, and_idx=None):
    loss = 0
    for i, or_list in enumerate(constraints):
        or_loss = 0
        for cstr in or_list:
            if cstr[0] == -1:
                or_loss += -logits[:, cstr[1]]
            elif cstr[1] == -1:
                or_loss += logits[:, cstr[0]]
            else:
                or_loss += logits[:, cstr[0]] - logits[:, cstr[1]]
        if and_idx is not None:
            loss += torch.where(and_idx == i, or_loss, torch.zeros_like(or_loss))
        else:
            loss += or_loss
    return -loss



class step_lr_scheduler:

    def __init__(self, initial_step_size, gamma=0.1, interval=10):
        self.initial_step_size = initial_step_size
        self.gamma = gamma
        self.interval = interval
        self.current_step = 0

    def step(self, k=1):
        self.current_step += k

    def get_lr(self):
        if isinstance(self.interval, int):
            return self.initial_step_size * self.gamma ** (np.floor(self.current_step / self.interval))
        phase = len([x for x in self.interval if self.current_step >= x])
        return self.initial_step_size * self.gamma ** (phase)
