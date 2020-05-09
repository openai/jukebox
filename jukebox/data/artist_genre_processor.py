import os
import re

accepted = frozenset([chr(i) for i in range(ord('a'), ord('z') + 1)] +
                     [chr(i) for i in range(ord('A'), ord('Z') + 1)] +
                     [chr(i) for i in range(ord('0'), ord('9') + 1)])

rex = re.compile(r'_+')

def norm(s):
    s = ''.join([c if c in accepted else '_' for c in s.lower()])
    s = rex.sub('_', s).strip('_')
    return s

def create_reverse_lookup(atoi):
    # Multiple entries could go to the same artist_id/genre_id
    itoa = {}
    for a, i in atoi.items():
        if i not in itoa:
            itoa[i] = []
        itoa[i].append(a)
    indices = sorted(list(itoa.keys()))
    for i in indices:
        itoa[i] = '_'.join(sorted(itoa[i]))
    return itoa

class ArtistGenreProcessor():
    def __init__(self, v3=False):
        self.v3 = v3
        dirname = os.path.dirname(__file__)
        if self.v3:
            self.artist_id_file = f"{dirname}/ids/v3_artist_ids.txt"
            self.genre_id_file = f"{dirname}/ids/v3_genre_ids.txt"
        else:
            self.artist_id_file = f"{dirname}/ids/v2_artist_ids.txt"
            self.genre_id_file = f"{dirname}/ids/v2_genre_ids.txt"
        self.load_artists()
        self.load_genres()

    def get_artist_id(self, artist):
        input_artist = artist
        if self.v3:
            artist = artist.lower()
        else:
            artist = norm(artist)
        if artist not in self.artist_ids:
            print(f"Input artist {input_artist} maps to {artist}, which is not present in {self.artist_id_file}. "
                  f"Defaulting to (artist_id, artist) = (0, unknown), if that seems wrong please format artist correctly")
        return self.artist_ids.get(artist, 0)

    def get_genre_ids(self, genre):
        if self.v3:
            genres = [genre.lower()]
        else:
            # In v2, we convert genre into a bag of words
            genres = norm(genre).split("_")
        for word in genres:
            if word not in self.genre_ids:
                print(f"Input genre {genre} maps to the list {genres}. {word} is not present in {self.genre_id_file}. "
                      f"Defaulting to (word_id, word) = (0, unknown), if that seems wrong please format genre correctly")
        return [self.genre_ids.get(word, 0) for word in genres]

    # get_artist/genre throw error if we ask for non-present values
    def get_artist(self, artist_id):
        return self.artists[artist_id]

    def get_genre(self, genre_ids):
        if self.v3:
            assert len(genre_ids) == 1
            genre = self.genres[genre_ids[0]]
        else:
            genre = '_'.join([self.genres[genre_id] for genre_id in genre_ids if genre_id >= 0])
        return genre

    def load_artists(self):
        print(f'Loading artist IDs from {self.artist_id_file}')
        self.artist_ids = {}
        with open(self.artist_id_file, 'r', encoding="utf-8") as f:
            for line in f:
                artist, artist_id = line.strip().split(';')
                self.artist_ids[artist.lower()] = int(artist_id)
        self.artists = create_reverse_lookup(self.artist_ids)

    def load_genres(self):
        print(f'Loading artist IDs from {self.genre_id_file}')
        self.genre_ids = {}
        with open(self.genre_id_file, 'r', encoding="utf-8") as f:
            for line in f:
                genre, genre_id = line.strip().split(';')
                self.genre_ids[genre.lower()] = int(genre_id)
        self.genres = create_reverse_lookup(self.genre_ids)


