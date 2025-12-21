import torch


def split_neuron(args, split_index, state_dict):
    """
    The function takes in a state_dict of a two-layer network, and a split_index, and returns a new state_dict of a
    wider two-layer network with the same width as the "sum" of the split_index.

    :param args: the arguments of the model
    :param split_index: a list of integers or lists of integers
    :param state_dict: the state_dict of the original model
    :return: The new width of the network and the new state dictionary.
    """

    assert args.hidden_layers_width[0] == len(
        split_index), 'len(split_index) must equal to hidden layer width. '

    new_width = 0

    for i in split_index:
        if isinstance(i, list):

            new_width += len(i)

        elif isinstance(i, int):

            new_width += i

    state_dict_new = {
        'features.0.weight': torch.zeros(new_width, 1),
        'features.0.bias': torch.zeros(new_width),
        'features.2.weight': torch.zeros(1, new_width)
    }
    index = 0
    for ind, i in enumerate(split_index):
        if isinstance(i, list):
            state_dict_new['features.0.weight'][index:index +
                                                len(i), 0] = state_dict['features.0.weight'][ind, 0].expand(len(i))

            state_dict_new['features.0.bias'][index:index +
                                              len(i)] = state_dict['features.0.bias'][ind].expand(len(i))

            state_dict_new['features.2.weight'][0, index:index+len(
                i)] = state_dict['features.2.weight'][0, ind].expand(len(i))*torch.Tensor(i)/sum(i)

            index += len(i)

        if isinstance(i, int):

            state_dict_new['features.0.weight'][index:index+i,
                                                0] = state_dict['features.0.weight'][ind, 0].expand(i)

            state_dict_new['features.0.bias'][index:index +
                                              i] = state_dict['features.0.bias'][ind].expand(i)

            state_dict_new['features.2.weight'][0, index:index +
                                                i] = state_dict['features.2.weight'][0, ind].expand(i)*1/i

            index += i

    return new_width, state_dict_new
