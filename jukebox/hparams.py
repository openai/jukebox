HPARAMS_REGISTRY = {}
DEFAULTS = {}

class Hyperparams(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

def setup_hparams(hparam_set_names, kwargs):
    H = Hyperparams()
    if not isinstance(hparam_set_names, tuple):
        hparam_set_names = hparam_set_names.split(",")
    hparam_sets = [HPARAMS_REGISTRY[x.strip()] for x in hparam_set_names if x] + [kwargs]
    for k, v in DEFAULTS.items():
        H.update(v)
    for hps in hparam_sets:
        for k in hps:
            if k not in H:
                raise ValueError(f"{k} not in default args")
        H.update(**hps)
    H.update(**kwargs)
    return H

# Teeny for testing
teeny = Hyperparams(
)
HPARAMS_REGISTRY["teeny"] = teeny

easy = Hyperparams(
    sr=22050,
)
HPARAMS_REGISTRY["easy"] = easy

REMOTE_PREFIX = 'https://openaipublic.azureedge.net/'

# Model hps
vqvae = Hyperparams(
    levels = 3,
    downs_t = (3, 2, 2),
    strides_t = (2, 2, 2),
    emb_width = 64,
    l_bins = 2048,
    l_mu = 0.99,
    commit = 0.02,
    spectral = 0.0,
    multispectral = 1.0,
    hvqvae_multipliers = (2, 1, 1),
    loss_fn = 'lmix',
    lmix_l2 = 1.0,
    lmix_linf=0.02,
    width = 32,
    depth = 4,
    m_conv = 1.0,
    dilation_growth_rate = 3,
    restore_vqvae=REMOTE_PREFIX + 'jukebox/models/5b/vqvae.pth.tar',
)
HPARAMS_REGISTRY["vqvae"] = vqvae

labels = Hyperparams(
    y_bins=(120, 4111),
    t_bins=128,
    max_bow_genre_size=5,
    n_vocab=80,
)

upsamplers = Hyperparams(
    n_ctx=8192,
    prior_width=1920,
    prior_depth=72,
    heads=1,
    attn_order=2,
    blocks=128,
    init_scale=0.4,
    c_res=1,
    cond_width=1024,
    cond_depth=16,
    cond_dilation_growth_rate=3,
    cond_dilation_cycle=8,
    cond_c_res=1,
    use_tokens=False,
    prime_loss_fraction=0.0,
    fp16_params=False,
)
upsamplers.update(labels)

upsampler_level_0 = Hyperparams(
    level=0,
    restore_prior=REMOTE_PREFIX + 'jukebox/models/5b/prior_level_0.pth.tar'
)
upsampler_level_0.update(upsamplers)
HPARAMS_REGISTRY["upsampler_level_0"] = upsampler_level_0

upsampler_level_1 = Hyperparams(
    level=1,
    cond_res_scale=True,
    restore_prior=REMOTE_PREFIX + 'jukebox/models/5b/prior_level_1.pth.tar'
)
upsampler_level_1.update(upsamplers)
HPARAMS_REGISTRY["upsampler_level_1"] = upsampler_level_1

prior_5b = Hyperparams(
    level=2,
    n_ctx=8192,
    prior_width=4800,
    prior_depth=72,
    heads=8,
    attn_order=2,
    blocks=128,
    init_scale=0.1,
    c_res=1,
    beta2=0.925,
    min_duration=60.0,
    max_duration=600.0,
    use_tokens=False,
    n_tokens=0,
    prime_loss_fraction=0.0,
    merged_decoder=True,
    restore_prior=REMOTE_PREFIX + 'jukebox/models/5b/prior_level_2.pth.tar',
    fp16_params=True,
)
prior_5b.update(labels)
HPARAMS_REGISTRY["prior_5b"] = prior_5b


prior_5b_lyrics = Hyperparams(
    level=2,
    n_ctx=8192,
    prior_width=4800,
    prior_depth=79,
    heads=8,
    attn_order=10,
    blocks=128,
    init_scale=0.1,
    c_res=1,
    prime_width=1280,
    prime_depth=18,
    prime_heads=4,
    prime_attn_order=2,
    prime_blocks=32,
    prime_init_scale=0.7,
    prime_c_res=1,
    min_duration=23.8,
    max_duration=600.0,
    use_tokens=True,
    n_tokens=512,
    prime_loss_fraction=0.4,
    merged_decoder=True,
    restore_prior=REMOTE_PREFIX + 'jukebox/models/5b_lyrics/prior_level_2.pth.tar',
    fp16_params=True,
    alignment_layer=68,
    alignment_head=2,
)
prior_5b_lyrics.update(labels)
HPARAMS_REGISTRY["prior_5b_lyrics"] = prior_5b_lyrics

labels_v3 = Hyperparams(
    y_bins=(604, 7898),
    t_bins=64,
    max_bow_genre_size=1,
    n_vocab=79,
)

prior_1b_lyrics = Hyperparams(
    level=2,
    n_ctx=6144,
    prior_width=2048,
    prior_depth=72,
    heads=2,
    attn_order=12,
    blocks=64,
    init_scale=0.2,
    c_res=1,
    labels_v3=True,
    min_duration=17.84,
    max_duration=600.0,
    use_tokens=True,
    n_tokens=384,
    prime_loss_fraction=0.4,
    single_enc_dec=True,
    restore_prior=REMOTE_PREFIX + 'jukebox/models/1b_lyrics/prior_level_2.pth.tar',
    fp16_params=False,
    alignment_layer=63,
    alignment_head=0,
)
prior_1b_lyrics.update(labels_v3)
HPARAMS_REGISTRY["prior_1b_lyrics"] = prior_1b_lyrics

# Small models
small_vqvae = Hyperparams(
    sr = 22050,
    levels = 2,
    downs_t = (5, 3),
    strides_t = (2, 2),
    emb_width = 64,
    l_bins = 1024,
    l_mu = 0.99,
    commit = 0.02,
    spectral = 0.0,
    multispectral = 1.0,
    loss_fn = 'l2',
    width = 32,
    depth = 4,
    m_conv = 1.0,
    dilation_growth_rate = 3,
)
HPARAMS_REGISTRY["small_vqvae"] = small_vqvae

small_prior = Hyperparams(
    n_ctx=8192,
    prior_width=1024,
    prior_depth=48,
    heads=1,
    c_res=1,
    attn_order=2,
    blocks=64,
    init_scale=0.7,
)
HPARAMS_REGISTRY["small_prior"] = small_prior

small_labelled_prior = Hyperparams(
    labels=True,
    labels_v3=True,
    y_bins=(10,100), # Set this to (genres, artists) for your dataset
    max_bow_genre_size=1,
    min_duration=60.0,
    max_duration=600.0,
    t_bins=64,
)
small_labelled_prior.update(small_prior)
HPARAMS_REGISTRY["small_labelled_prior"] = small_labelled_prior

small_single_enc_dec_prior = Hyperparams(
    n_ctx=6144,
    prior_width=1024,
    prior_depth=48,
    heads=2,
    attn_order=12,
    blocks=64,
    init_scale=0.7,
    c_res=1,
    prime_loss_fraction=0.4,
    single_enc_dec=True,
    labels=True,
    labels_v3=True,
    y_bins=(10,100), # Set this to (genres, artists) for your dataset
    max_bow_genre_size=1,
    min_duration=60.0,
    max_duration=600.0,
    t_bins=64,
    use_tokens=True,
    n_tokens=384,
    n_vocab=79,
)
HPARAMS_REGISTRY["small_single_enc_dec_prior"] = small_single_enc_dec_prior

small_sep_enc_dec_prior = Hyperparams(
    n_ctx=6144,
    prior_width=1024,
    prior_depth=50,
    heads=2,
    attn_order=8,
    blocks=64,
    init_scale=0.7,
    c_res=1,
    prime_width=256,
    prime_depth=9,
    prime_heads=2,
    prime_attn_order=2,
    prime_blocks=32,
    prime_init_scale=0.7,
    prime_c_res=1,
    prime_loss_fraction=0.4,
    labels=True,
    labels_v3=True,
    y_bins=(10,100), # Set this to (genres, artists) for your dataset
    max_bow_genre_size=1,
    min_duration=60.0,
    max_duration=600.0,
    t_bins=64,
    use_tokens=True,
    n_tokens=384,
    n_vocab=79,
)
HPARAMS_REGISTRY["small_sep_enc_dec_prior"] = small_sep_enc_dec_prior

small_upsampler = Hyperparams(
    n_ctx=8192,
    prior_width=1024,
    prior_depth=48,
    heads=1,
    c_res=1,
    attn_order=2,
    blocks=64,
    init_scale=0.7,
    cond_width=512,
    cond_depth=16,
    cond_dilation_growth_rate=3,
    cond_dilation_cycle=8,
    cond_c_res=1,
)

HPARAMS_REGISTRY["small_upsampler"] = small_upsampler

all_fp16 = Hyperparams(
    fp16=True,
    fp16_params=True,
    fp16_opt=True,
    fp16_scale_window=250,
)
HPARAMS_REGISTRY["all_fp16"] = all_fp16

cpu_ema = Hyperparams(
    ema=True,
    cpu_ema=True,
    cpu_ema_freq=100,
    ema_fused=False,
)
HPARAMS_REGISTRY["cpu_ema"] = cpu_ema


DEFAULTS["rcall"] = Hyperparams(
    rcall_command="<unknown_rcall_command>",
    git_commit="<unknown_git_commit>",
)

DEFAULTS["script"] = Hyperparams(
    name='',
    debug_mem=False,
    debug_eval_files=False,
    debug_speed=False,
    debug_iters=100,
    debug_batch=False,
    debug_grad_accum=False,
    debug_inputs=False,
    local_path='',
    local_logdir='logs',
    max_len=24,
    max_log=32,
    save=True,
    save_iters=20000,
    seed=0,
    prior=False,
    log_steps=100,
    func='',
)

DEFAULTS["data"] = Hyperparams(
    audio_files_dir='',
    finetune='',
    english_only=False,
    bs=1,
    bs_sample=1,
    nworkers=1,
    aug_shift=False,
    aug_blend=False,
    train_test_split=0.9,
    train_shrink_factor=1.0,
    test_shrink_factor=1.0,
    p_unk=0.1,
    min_duration=None,
    max_duration=None,
    n_tokens=0,
    n_vocab=0,
    use_tokens=False,
    curr_epoch=-1,
)

DEFAULTS["vqvae"] = Hyperparams(
    restore_vqvae='',
    levels=2,
    downs_t=(1,1),
    strides_t=(2,2),
    hvqvae_multipliers=None,
    revival_threshold=1.0,
    emb_width=64,
    l_bins=512,
    l_mu=0.99,
    commit=1.0,
    spectral=0.0,
    multispectral=1.0,
    loss_fn='l2',
    linf_k=2048,
    lmix_l1=0.0,
    lmix_l2=0.0,
    lmix_linf=0.0,
    use_bottleneck=True,
)

DEFAULTS["vqvae_conv_block"] = Hyperparams(
    depth=3,
    width=128,
    m_conv=1.0,
    dilation_growth_rate=1,
    dilation_cycle=None,
    vqvae_reverse_decoder_dilation=True,
)

DEFAULTS["prior"] = Hyperparams(
    restore_prior='',
    restore_prior_ddp=False,
    max_bow_genre_size=None,
    y_bins=0,
    level=0,
    cond_levels=None,
    t_bins=64,
    y_cond_as_bias=False,
    copy_input=False,
    merged_decoder=False,
    single_enc_dec=False,
    alignment_layer=None,
    alignment_head=None,
)

DEFAULTS["prior_attn_block"] = Hyperparams(
    n_ctx=1024,
    prior_depth=3,
    prior_width=128,
    heads=1,
    attn_order=0,
    blocks=None,
    spread=None,
    attn_dropout=0.0,
    resid_dropout=0.0,
    emb_dropout=0.0,
    zero_out=False,
    res_scale=False,
    pos_init=False,
    init_scale=1.0,
    m_attn=0.25,
    m_mlp=1.0,
    c_res=0,
    c_attn=0,
    c_mlp=0,
)

DEFAULTS["cond_conv_block"] = Hyperparams(
    cond_depth=3,
    cond_width=128,
    cond_m_conv=1.0,
    cond_zero_out=False,
    cond_res_scale=False,
    cond_dilation_growth_rate=1,
    cond_dilation_cycle=None,
    cond_c_res=0,
)

DEFAULTS["sample"] = Hyperparams(
    primed_chunk_size=None,
    selected_artists='',
    temp_top=1.0,
    temp_rest=0.99,
    sample_length_in_seconds=24,
    total_sample_length_in_seconds=240,
)

DEFAULTS["prime"] = Hyperparams(
    #encoder_kv_width=128,
    prime_loss_fraction=0.1,
    restore_decoder='',
)
DEFAULTS["prime_attn_block"] = Hyperparams(
    prime_depth=3,
    prime_width=128,
    prime_heads=1,
    prime_attn_order=0,
    prime_blocks=None,
    prime_spread=None,
    prime_attn_dropout=0.0,
    prime_resid_dropout=0.0,
    prime_emb_dropout=0.0,
    prime_zero_out=False,
    prime_res_scale=False,
    prime_pos_init=False,
    prime_init_scale=1.0,
    prime_m_attn=0.25,
    prime_m_mlp=1.0,
    prime_c_res=0,
    prime_c_attn=0,
    prime_c_mlp=0,
    prime_rel_attn=False,
    prime_posemb_timescale=10000,
)

DEFAULTS["opt"] = Hyperparams(
    epochs=10000,
    lr=0.0003,
    clip=1.0,
    beta1=0.9,
    beta2=0.999,
    ignore_grad_norm=0,
    weight_decay=0.0,
    eps=1e-08,
    lr_warmup=100.0,
    lr_decay=10000000000.0,
    lr_gamma=1.0,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
    lr_use_cosine_decay=False,
)

DEFAULTS["fp16"] = Hyperparams(
    fp16=False,
    fp16_params=False,
    fp16_loss_scale=None,
    fp16_scale_window=1000.0,
    fp16_opt=False,
)

DEFAULTS["train_test_eval"] = Hyperparams(
    labels=True,
    labels_v3=False,
    dump=False,
    ema=True,
    ema_fused=True,
    cpu_ema=False,
    cpu_ema_freq=100,
    reset_best_loss=False,
    reset_step=False,
    reset_opt=False,
    reset_shd=False,
    train=False,
    test=False,
    sample=False,
    sampler='ancestral',
    codes_logdir='',
    date=None,
    labeller='top_genres',
    label_line=0,
    iters_before_update=1,
    grad_accum_iters=0,
    mu=None,
    piped=False,
    pipe_depth=8,
    break_train=1e10,
    break_test=1e10,
    exit_train=1e10,
)

DEFAULTS["audio"] = Hyperparams(
    n_fft=1024,
    hop_length=256,
    window_size=1024,
    sr=44100,
    channels=2,
    wav='',
    n_inps=1,
    n_hops=2,
    n_segment=1,
    n_total_segment=1,
    n_segment_each=1,
    prime_chunks=4,
    sample_length=0,
    sample_hop_length=30000,
    max_silence_pad_length=0,
    ignore_boundaries=False,
    use_nonrelative_specloss=True,
    multispec_loss_n_fft=(2048,1024,512),
    multispec_loss_hop_length=(240,120,50),
    multispec_loss_window_size=(1200,600,240),
)

DEFAULTS["distributed"] = Hyperparams(
    bucket=128
)
