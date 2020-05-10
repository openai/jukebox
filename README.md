**Status:** Archive (code is provided as-is, no updates expected)

# Jukebox
Code for "Jukebox: A Generative Model for Music"

[Paper](https://arxiv.org/abs/2005.00341) 
[Blog](https://openai.com/blog/jukebox) 
[Explorer](http://jukebox.openai.com/) 
[Colab](https://colab.research.google.com/github/openai/jukebox/blob/master/jukebox/Interacting_with_Jukebox.ipynb)

# Install
``` 
# Required: Sampling
conda create --name jukebox python=3.7.5
conda activate jukebox
conda install mpi4py=3.0.3
conda install pytorch=1.4 torchvision=0.5 cudatoolkit=10.0 -c pytorch
git clone https://github.com/openai/jukebox.git
cd jukebox
pip install -r requirements.txt
pip install -e .

# Required: Training
conda install av=7.0.01 -c conda-forge 
pip install ./tensorboardX
 
# Optional: Apex for faster training with fused_adam
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=10.0 -c pytorch
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

# Sampling
## Sampling from scratch
To sample normally, run the following command. Model can be `5b`, `5b_lyrics`, `1b_lyrics`
``` 
python jukebox/sample.py --model=5b_lyrics --name=sample_5b --levels=3 --sample_length_in_seconds=20 --total_sample_length_in_seconds=180 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
``` 
python jukebox/sample.py --model=1b_lyrics --name=sample_1b --levels=3 --sample_length_in_seconds=20 --total_sample_length_in_seconds=180 --sr=44100 --n_samples=16 --hop_fraction=0.5,0.5,0.125
```
The above generates the first `sample_length_in_seconds` seconds of audio from a song of total length `total_sample_length_in_seconds`.

The samples decoded from each level are stored in `{name}/level_{level}`. 
You can also view the samples as an html with the aligned lyrics under `{name}/level_{level}/index.html`. Run `python -m http.server` and open the html through the server to see the lyrics animate as the song plays.  
A summary of all sampling data including zs, x, labels and sampling_kwargs is stored in `{name}/level_{level}/data.pth.tar`.

The hps are for a V100 GPU with 16 GB GPU memory. The `1b_lyrics`, `5b`, and `5b_lyrics` top-level priors take up 3.8 GB, 10.3 GB, and 11.5 GB, respectively. The peak memory usage to store transformer key, value cache is about 400 MB for `1b_lyrics` and 1 GB for `5b_lyrics` per sample. If you are having trouble with CUDA OOM issues, try `1b_lyrics` or decrease `max_batch_size` in sample.py, and `--n_samples` in the script call.

On a V100, it takes about 3 hrs to fully sample 20 seconds of music. Since this is a long time, it is recommended to use `n_samples > 1` so you can generate as many samples as possible in parallel. The 1B lyrics and upsamplers can process 16 samples at a time, while 5B can fit only up to 3. Since the vast majority of time is spent on upsampling, we recommend using a multiple of 3 less than 16 like `--n_samples 15` for `5b_lyrics`. This will make the top-level generate samples in groups of three while upsampling is done in one pass.

To continue sampling from already generated codes for a longer duration, you can run
```
python jukebox/sample.py --model=5b_lyrics --name=sample_5b --levels=3 --mode=continue --codes_file=sample_5b/level_0/data.pth.tar --sample_length_in_seconds=40 --total_sample_length_in_seconds=180 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
Here, we take the 20 seconds samples saved from the first sampling run at `sample_5b/level_0/data.pth.tar` and continue by adding 20 more seconds. 

You could also continue directly from the level 2 saved outputs, just pass `--codes_file=sample_5b/level_2/data.pth.tar`. Note this will upsample the full 40 seconds song at the end.

If you stopped sampling at only the first level and want to upsample the saved codes, you can run
```
python jukebox/sample.py --model=5b_lyrics --name=sample_5b --levels=3 --mode=upsample --codes_file=sample_5b/level_2/data.pth.tar --sample_length_in_seconds=20 --total_sample_length_in_seconds=180 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
Here, we take the 20 seconds samples saved from the first sampling run at `sample_5b/level_2/data.pth.tar` and upsample the lower two levels.

## Prompt with your own music
If you want to prompt the model with your own creative piece or any other music, first save them as wave files and run
```
python jukebox/sample.py --model=5b_lyrics --name=sample_5b_prompted --levels=3 --mode=primed --audio_file=path/to/recording.wav,awesome-mix.wav,fav-song.wav,etc.wav --prompt_length_in_seconds=12 --sample_length_in_seconds=20 --total_sample_length_in_seconds=180 --sr=44100 --n_samples=6 --hop_fraction=0.5,0.5,0.125
```
This will load the four files, tile them to fill up to `n_samples` batch size, and prime the model with the first `prompt_length_in_seconds` seconds.

# Training
## VQVAE
To train a small vqvae, run
```
mpiexec -n {ngpus} python jukebox/train.py --hps=small_vqvae --name=small_vqvae --sample_length=262144 --bs=4 --nworkers=4 --audio_files_dir={audio_files_dir} --labels=False --train --aug_shift --aug_blend
```
Here, `{audio_files_dir}` is the directory in which you can put the audio files for your dataset, and `{ngpus}` is number of GPU's you want to use to train. 
The above trains a two-level VQ-VAE with `downs_t = (5,3)`, and `strides_t = (2, 2)` meaning we downsample the audio by `2**5 = 32` to get the first level of codes, and `2**8 = 256` to get the second level codes.  
Checkpoints are stored in the `logs` folder. You can monitor the training by running Tensorboard
```
tensorboard --logdir logs
```
    
## Prior
### Train prior or upsamplers
Once the VQ-VAE is trained, we can restore it from its saved checkpoint and train priors on the learnt codes. 
To train the top-level prior, we can run

```
mpiexec -n {ngpus} python jukebox/train.py --hps=small_vqvae,small_prior,all_fp16,cpu_ema --name=small_prior --sample_length=2097152 --bs=4 --nworkers=4 --audio_files_dir={audio_files_dir} --labels=False --train --test --aug_shift --aug_blend --restore_vqvae=logs/small_vqvae/checkpoint_latest.pth.tar --prior --levels=2 --level=1 --weight_decay=0.01 --save_iters=1000
```

To train the upsampler, we can run
```
mpiexec -n {ngpus} python jukebox/train.py --hps=small_vqvae,small_upsampler,all_fp16,cpu_ema --name=small_upsampler --sample_length 262144 --bs 4 --nworkers 4 --audio_files_dir {audio_files_dir} --labels False --train --test --aug_shift --aug_blend --restore_vqvae logs/small_vqvae/checkpoint_latest.pth.tar --prior --levels 2 --level 0 --weight_decay 0.01 --save_iters 1000
```
We pass `sample_length = n_ctx * downsample_of_level` so that after downsampling the tokens match the n_ctx of the prior hps. 
Here, `n_ctx = 8192` and `downsamples = (32, 256)`, giving `sample_lengths = (8192 * 32, 8192 * 256) = (65536, 2097152)` respectively for the bottom and top level. 

### Reuse pre-trained VQ-VAE and retrain top level prior on new dataset.
#### No labels
Our pre-trained VQ-VAE can produce compressed codes for a wide variety of genres of music, and the pre-trained upsamplers can upsample them back to audio that sound very similar to the original audio.
To re-use these for a new dataset of your choice, you can retrain just the top-level  

To retrain top-level on a new dataset, run
```
mpiexec -n {ngpus} python jukebox/train.py --hps=vqvae,small_prior,all_fp16,cpu_ema --name=pretrained_vqvae_small_prior --sample_length=1048576 --bs=4 --nworkers=4 --bs_sample=4 --aug_shift --aug_blend --audio_files_dir={audio_files_dir} --labels=False --train --test --prior --levels=3 --level=2 --weight_decay=0.01 --save_iters=1000
```
You can then run sample.py with the top-level of our models replaced by your new model. To do so, add an entry `my_model` in MODELs (in `make_models.py`) with the (vqvae hps, upsampler hps, top-level prior hps) of your new model, and run sample.py with `--model=my_model`. 

#### With labels 
To train with you own metadata for your audio files, pass `--labels=True --labels_v3=True` and implement `get_metadata` in `data/files_dataset.py` to return the `artist`, `genre` and `lyrics` for a given audio file. 

We use 3 kinds of labels information:
- Artist/Genre: 
  - For each file, we return an artist_id and a list of genre_ids. The reason we have a list and not a single genre_id is that in v2, we split genres like `blues_rock` into a bag of words `[blues, rock]`, and we pass atmost `max_bow_genre_size` of those, in `v3` we consider it as a single word and just set `max_bow_genre_size=1`.
  - Update the `v3_artist_ids` and `v3_genre_ids` to use ids from your new dataset. Pass the hps `y_bins = (number_of_artists, number_of_genres)` and `max_bow_genre_size=1`. 
- Timing: 
  - For each chunk of audio, we return the `total_length` of the song, the `offset` the current audio chunk is at and the `sample_length` of the audio chunk. We have three timing embeddings: total_length, our current position, and our current position as a fraction of the total length, and we divide the range of these values into `t_bins` discrete bins. 
  - Pass the hps `min_duration` and `max_duration` to be the shortest/longest duration of audio files, and `t_bins` for how many bins you want to discretize timing information into. 
- Lyrics: 
  - For each file, we linearly align the lyric characters to the audio, find the position in lyric that corresponds to the midpoint of our audio chunk, and pass a window of `n_tokens` lyric characters centred around that. 
  - Pass the hps `use_tokens=True` and `n_tokens` to be the number of lyric characters to use for an audio chunk. Set it according to the `sample_length` you're training on so that its large enough that the lyrics for an audio chunk are almost always found inside a window of that size.
  - If you use a non-English vocabulary, update `text_processor.py` with your new vocab and pass `n_vocab = number of characters in vocabulary` accordingly. In v2, we had a `n_vocab=80` and in v3 we missed `+` and so `n_vocab=79` of characters. 

After these modifications, to train a top-level with labels and lyrics, run
```
mpiexec -n {ngpus} python jukebox/train.py --hps=vqvae,small_lyric_prior,all_fp16,cpu_ema --name=pretrained_vqvae_small_lyric_prior --sample_length=1048576 --bs=4 --nworkers=4 --bs_sample=4 --aug_shift --aug_blend --audio_files_dir={audio_files_dir} --labels=True --train --test --prior --levels=3 --level=2 --weight_decay=0.01 --save_iters=1000 --labels_v3=True --y_bins=({artists},{genres}) --max_bow_genre_size=1 --min_duration=60.0 --max_duration=600.0 --t_bins=64 --use_tokens=True --n_tokens=384 --n_vocab=79
```
To simplify hps choices, here we used a `single_enc_dec` model that combines both encoder and decoder of the transformer into a single model. We do so by merging the lyric vocab and vq-vae vocab into a single larger vocab, and flattening the lyric tokens and the vq-vae codes into a single sequence of length `n_ctx + n_tokens`.   

# Citation

Please cite using the following bibtex entry:

```
@article{dhariwal2020jukebox,
  title={Jukebox: A Generative Model for Music},
  author={Dhariwal, Prafulla and Jun, Heewoo and Payne, Christine and Kim, Jong Wook and Radford, Alec and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2005.00341},
  year={2020}
}
```

# License 
[Noncommercial Use License](./LICENSE) 

It covers both released code and weights. 

