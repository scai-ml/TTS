import multiprocessing
import os.path
from argparse import ArgumentParser

from tqdm.contrib.concurrent import process_map

from TTS.config import BaseAudioConfig
from TTS.utils.audio import AudioProcessor

conf = BaseAudioConfig(
    sample_rate=16000,
    resample=False,
    do_rms_norm=True,
    db_level=-27,
    do_trim_silence=False
)

global ap, out_fld, in_fld
ap = AudioProcessor(**conf)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--wavs_fld', type=str, required=True)
    parser.add_argument('--out_fld', type=str, required=True)
    return parser.parse_args()


def normalize_wav(file_name):
    global ap
    wav = ap.load_wav(os.path.join(in_fld, file_name))
    ap.save_wav(wav, path=os.path.join(out_fld, file_name))


def main():
    global out_fld, in_fld
    args = get_args()
    out_fld = args.out_fld
    in_fld = args.wavs_fld
    os.makedirs(out_fld, exist_ok=True)

    wav_names = os.listdir(args.wavs_fld)

    num_threads = multiprocessing.cpu_count()
    process_map(normalize_wav, wav_names, max_workers=num_threads, chunksize=15)


if __name__ == '__main__':
    main()
