


def get_adv_string(inputs, outputs):
    x = inputs.flatten().detach().cpu().numpy()
    y = outputs.flatten().detach().cpu().numpy()
    string_x = [f'(X_{i} {x[i]})' for i in range(len(x))]
    string_y = [f'(Y_{i} {y[i]})' for i in range(len(y))]
    string = '\n '.join(string_x + string_y)
    return f"({string})"