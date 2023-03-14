cfg = dict(
    # data params
    train_sessions=list(range(0, 6)),  # 0-indexing
    valid_sessions=list(range(6, 7)),
    event_name_cls_map=dict(left=0, right=1),
    include_chans_name=['C3', 'Cz', 'C4'],
    #include_chans_name = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'],
    epoch_t_interval=None,  # if None, (0, np.inf)
    epoch_type='epochs_on_task',
    load_to_mem=True,
    match_t_and_freq_dim=max,  # None | min | max | ...

    # model/training params
    epochs=100,
    batch_size=64,
    accumulate_grad_batches=1,
    dev='cuda',
    ndev=1,
    multi_dev_strat=None,
    precision=32,
    gradient_clip_val=None,
    num_workers=0,
    prefetch_factor=2,
)
