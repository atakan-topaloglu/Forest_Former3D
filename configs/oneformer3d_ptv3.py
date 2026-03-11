"""
ForestFormer with Point Transformer V3 Backbone Configuration

This config replaces the SpConvUNet backbone with PTV3 for ForAINetV2 dataset.

Key Design Choices:
1. Input channels: 6 (centered_xyz + zero padding for pretrained weight compatibility)
2. Output channels: 64 (PTV3's dec_channels[0])
3. Grid size: 0.02 (from PTV3 config)
4. PDNorm: Disabled by default (pdnorm_bn=False, pdnorm_ln=False)
5. Flash Attention: Enabled for efficiency
6. Serialization: Default orders (z, z-trans, hilbert, hilbert-trans)

Usage:
    python tools/train.py configs/oneformer3d_ptv3.py
"""

_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

# Model settings
num_channels = 64  # PTV3 output channels (dec_channels[0])
num_instance_classes = 3
num_semantic_classes = 3
radius = 12
grid_size = 0.2  # PTV3 voxel size

model = dict(
    type='ForAINetV2OneFormer3D_PTV3',
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
        type='PTV3Backbone',
        in_channels=3,  # Will be zero-padded to 6 internally
        grid_size=grid_size,
        out_channels=num_channels,
        # Pretrained weights (set path to load pretrained PTV3 weights)
        # Example: pretrained='path/to/ptv3_scannet.pth'
        pretrained=None,
        # Serialization settings
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # Encoder settings
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # Decoder settings
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),  # First value = out_channels
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        # Transformer settings
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        # Attention settings
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        # PDNorm settings
        # NOTE: Set pdnorm_bn=False and pdnorm_ln=False when using pretrained weights
        # The pretrained checkpoint was trained without PDNorm
        pdnorm_bn=False,
        pdnorm_ln=False,  # Disabled for pretrained weight compatibility
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ForAINet",),
    ),
    decoder=dict(
        type='ForAINetv2QueryDecoder_XAwarequery',
        num_layers=6,
        num_classes=1,
        num_instance_queries=300,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,  # Match backbone output: 64
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
    dict(type='GridSample', grid_size=grid_size),  # Use PTV3 grid_size
    dict(
        type='PointSample_',
        num_points=320000),
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
    dict(type='GridSample', grid_size=grid_size),  # Use PTV3 grid_size
    dict(
        type='PointSample_',
        num_points=320000),
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
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
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
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

# Learning rate scheduler
param_scheduler = dict(type='PolyLR', begin=0, end=450000, power=0.9, by_epoch=False)

# Hooks
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    # Fix spconv weight format for SpConvUNet during validation + checkpoint saving
    # NOTE: PTV3Backbone is automatically skipped (doesn't have this issue)
    # dict(type='SpConvWeightFixHook', verbose=False),
]
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=100,
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
