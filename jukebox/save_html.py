import os
import json
import numpy as np
from PIL import Image, ImageFilter
import soundfile

def save_html(logdir, x, zs, labels, alignments, hps):
    level = hps.levels - 1 # Top level used
    z = zs[level]
    bs, total_length = z.shape[0], z.shape[1]

    with open(f'{logdir}/index.html', 'w') as html:
        print(f"<html><head><title>{logdir}</title></head><body style='font-family: sans-serif; font-size: 1.4em; font-weight: bold; text-align: center; max-width:1024px; width: 100%; margin: auto;'>",
            file=html)
        print("<link rel='icon' href='data:;base64,iVBORw0KGgo='>", file=html)

        for item in range(bs):
            data = dict(wav=x[item].cpu().numpy(), sr=hps.sr,
                        info=labels['info'][item],
                        total_length=total_length,
                        total_tokens=len(labels['info'][item]['full_tokens']),
                        alignment=alignments[item] if alignments is not None else None)
            item_dir = f'{logdir}/item_{item}'
            _save_item_html(item_dir, item, item, data)
            print(f"<iframe style='height: 100%; width: 100%;' frameborder='0' scrolling='no' src='item_{item}/index.html'></iframe>", file=html)
        print("</body></html>", file=html)  

def _save_item_html(item_dir, item_id, item_name, data):
    # replace gs:// with /root/samples/

    # an html for each sample. Main html has a selector to get us id of this?
    if not os.path.exists(item_dir):
        os.makedirs(item_dir)

    with open(f'{item_dir}/index.html', 'w') as html:
        print(f"<html><head><title>{item_name}</title></head><body style='font-family: sans-serif; font-size: 1.4em; font-weight: bold; text-align: center; max-width:1024px; width: 100%; margin: auto;'>",
            file=html)
        print("<link rel='icon' href='data:;base64,iVBORw0KGgo='>", file=html)
        total_length = data['total_length']
        total_tokens = data['total_tokens']
        alignment = data['alignment']
        lyrics = data["info"]["lyrics"]
        wav, sr = data['wav'], data['sr']
        genre, artist = data["info"]["genre"], data["info"]["artist"]

        # Strip unused columns
        if alignment is not None:
            assert alignment.shape == (total_length, total_tokens)
            assert len(lyrics) == total_tokens, f'Total_tokens: {total_tokens}, Lyrics Len: {len(lyrics)}. Lyrics: {lyrics}'
            max_attn_at_token = np.max(alignment, axis=0)
            assert len(max_attn_at_token) == total_tokens
            for token in reversed(range(total_tokens)):
                if max_attn_at_token[token] > 0:
                    break
            alignment = alignment[:,:token+1]
            lyrics = lyrics[:token+1]
            total_tokens = token+1

            # Small alignment image
            im = Image.fromarray(np.uint8(alignment * 255)).resize((512, 1024)).transpose(Image.ROTATE_90)
            img_src = f'align.png'
            im.save(f'{item_dir}/{img_src}')
            print(f"<img id='{img_src}' src='{img_src}' \>", file=html)

            # Smaller alignment json for animation
            total_alignment_length = total_length // 16
            alignment = Image.fromarray(np.uint8(alignment * 255)).resize((total_tokens, total_alignment_length))
            alignment = alignment.filter(ImageFilter.GaussianBlur(radius=1.5))
            alignment = np.asarray(alignment).tolist()
            align_src = f'align.json'
            with open(f'{item_dir}/{align_src}', 'w') as f:
                json.dump(alignment, f)

        # Audio
        wav_src = f'audio.wav'
        soundfile.write(f'{item_dir}/{wav_src}', wav, samplerate=sr, format='wav')
        print(f"<audio id='{wav_src}' src='{wav_src}' style='width: 100%;' controls></audio>", file=html)


        # Labels and Lyrics
        print(f"<pre style='white-space: pre-wrap;'>", end="", file=html)
        print(f"<div>Artist {artist}, Genre {genre}</div>", file=html)
        lyrics = [c for c in lyrics]  # already characters actually
        lyrics = [''] + lyrics[:-1]  # input lyrics are shifted by 1
        for i, c in enumerate(lyrics):
            print(f"<span id='{item_id}/{i}'>{c}</span>", end="", file=html)
        print(f"</pre>", file=html)
        with open(f'{item_dir}/lyrics.json', 'w') as f:
            json.dump(lyrics, f)

        if alignment is not None:
            # JS for alignment animation
            print("""<script>
            async function fetchAsync (url) {
                let response = await fetch(url);
                let data = await response.json();
                return data;
            }
    
            var audio = document.getElementById('""" + f'{wav_src}' + """');
            audio.onplay = function () {
                track = '""" + f'{item_id}' + """'
                fetchAsync('""" + f'{align_src}' + """')
                .then(data => animateLyrics(data, track, this))
                .catch(reason => console.log(reason.message))
            }; 
    
            function animateLyrics(data, track, audio) {
                var animate = setInterval(function () {
                    var time = Math.floor(audio.currentTime*""" + f'{total_alignment_length}' + """/audio.duration);
                    if (!(time == 0 || time == """ + f'{total_alignment_length}' + """)) {
                        console.log(time);
                        changeColor(data, track, audio, time);
                    }
                    if (audio.paused) {
                        clearInterval(animate);
                    }
                }, 50);
            }
    
            function changeColor(data, track, audio, time) {
                colors = data[time]
                for (i = 0; i < colors.length; i++){
                    character = document.getElementById(track + '/' + i.toString());
                    color = Math.max(230 - 10*colors[i], 0).toString();
                    character.style.color = 'rgb(255,' + color + ',' + color + ')';
                }
            }
            </script>""", file=html)
        print("</body></html>", file=html)
