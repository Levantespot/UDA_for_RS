uda = dict(
    type='DACS',
    alpha=0.9,
    pseudo_threshold=0.9,
    mix_class_threshold=0.1,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    pseudo_kernal_size=5,
    local_ps_weight_type='label',
    debug_img_interval=1000,
    print_grad_magnitude=False,
)
use_ddp_wrapper = True
