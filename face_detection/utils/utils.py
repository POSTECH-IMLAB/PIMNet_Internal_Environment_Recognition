import torch


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    return len(used_pretrained_keys) > 0


def load_model(model, checkpoint_path, is_train=False):
    print('Loading pretrained model from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint["net_state_dict"]
    assert check_keys(model, state_dict), 'load NONE from pretrained checkpoint'
    model.load_state_dict(state_dict, strict=False)
    if not is_train:
        model.eval().requires_grad_(False)
    return model