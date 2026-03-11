"""
ForestFormer LitePT - Training from Scratch (Full Dataset)

This config trains ForestFormer with LitePT backbone from scratch.
"""

_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

# Model settings
num_channels = 72  # LitePT default output channels
num_instance_classes = 3
num_semantic_classes = 3
radius = 16
grid_size = 0.2  # Matches PTV3 config

model = dict(
    type='ForAINetV2OneFormer3D_LitePT',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=3,  # Input feature channels (xyz centered coords)
    num_channels=num_channels,
    grid_size=grid_size,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    stuff_classes=[0],
    thing_cls=[1, 2],
    radius=radius,
    backbone=dict(
        type='PT-v3m1-gridpool-xcpe-xattn-rope-pretrain',
        in_channels=3,  # 3D input (xyz) different from https://github.com/prs-eth/LitePT/blob/main/configs/scannet200/insseg-litept-small-v1m2.py 
        grid_size=grid_size,
        # LitePT-Small configuration
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_cpe=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_cpe=(True, True, True, False),
        dec_attn=(False, False, False, True),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
    ),
    decoder=dict(
        type='ForAINetv2QueryDecoder',
        num_layers=6,
        num_classes=1,
        num_instance_queries=300,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True),
    criterion=dict(
        type='ForAINetv2UnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='S3DISSemanticCriterion',
            loss_weight=0.2),
        inst_criterion=dict(
            type='InstanceCriterionForAI',
            matcher=dict(
                type='HungarianMatcher',
                costs=[
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)]),
            loss_weight=[1.0, 1.0, 0.5],
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=250,
        inst_score_thr=0.4,
        pan_score_thr=0.4,
        npoint_thr=10,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[0],
        thing_cls=[0]))

# Dataset settings
dataset_type = 'ForAINetV2SegDataset_'
data_root_forainetv2 = 'data/ForAINetV2/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='CylinderCrop', radius=radius),
    dict(type='GridSample', grid_size=grid_size),  # Use LitePT grid_size
    dict(
        type='PointSample_',
        num_points=640000),
    dict(type='SkipEmptyScene_'),
    dict(type='PointInstClassMapping_',
        num_classes=num_instance_classes),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask', 'ratio_inspoint'
        ])
]
val_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='CylinderCrop', radius=radius),
    dict(type='GridSample', grid_size=grid_size),  # Use LitePT grid_size
    dict(
        type='PointSample_',
        num_points=640000),
    dict(type='PointInstClassMapping_',
        num_classes=num_instance_classes),
    dict(type='Pack3DDetInputs_', keys=['points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='Pack3DDetInputs_', keys=['points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask'])
]

# Dataloader settings
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_train.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        filter_empty_gt=True,
        box_type_3d='Depth',
        backend_args=None))
val_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=val_pipeline,
        box_type_3d='Depth',
        test_mode=True,
        backend_args=None))
test_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_test.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        box_type_3d='Depth',
        test_mode=True,
        backend_args=None))

# Evaluation settings
class_names = ['ground', 'wood', 'leaf']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[],
    classes=class_names,
    dataset_name='ForAINetV2')

sem_mapping = [0, 1, 2]
inst_mapping = sem_mapping[1:]
val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0],
    thing_class_inds=list(range(1, num_semantic_classes)),
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

# Optimizer settings
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

# Learning rate scheduler
param_scheduler = dict(type='PolyLR', begin=0, end=3000, power=0.9, by_epoch=True)

# Hooks
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
]
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=200,
        max_keep_ckpts=3,
        save_optimizer=True),
    logger=dict(type='LoggerHook', interval=20),
    visualization=dict(type='Det3DVisualizationHook', draw=False))

# Visualization
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Training settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=3000,
    val_interval=100)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
find_unused_parameters = True
