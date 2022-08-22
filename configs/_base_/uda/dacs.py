uda = dict(
    type='DACS',
    alpha=0.9,
    pseudo_threshold=0.9,
    dynamic_class_weight=False,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    pseudo_kernal_size=None,
    local_ps_weight_type=None,
    debug_img_interval=1000,
    print_grad_magnitude=False,
)
use_ddp_wrapper = True
