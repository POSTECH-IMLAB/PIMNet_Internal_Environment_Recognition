class Config(object):
    lr = 0.001

    # 'LIGHT' or 'HEAVY' or 'HEAVY+ATT' or 'MORE_LIGHT'
    #use_model_type = 'HEAVY+ATT'
    use_model_type = 'MORE_LIGHT'

    alpha = 2
    batch_size = 200
    global_img_size = [100, 120]
    local_img_size = [100, 80]
    schedule = [150, 225]
    gamma = 0.1
    print_iter = 5
    save_epoch = 10

    data_path = 'D:/-----/cropped_fld_and_face'
    save_path = 'save_checks_more_light'

    max_epoch = 200
    gpus = "0"
    class_num = 6
    momentum= 0.9
    weight_decay = 5e-4