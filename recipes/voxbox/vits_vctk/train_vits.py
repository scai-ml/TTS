import os
from typing import List

from dataclasses import dataclass

from TTS.config.shared_configs import BaseAudioConfig
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text import TTSTokenizer
from TTS.utils.audio import AudioProcessor

RESUME_EPOCH = -1
output_path = "/mnt/FastData/dl_workspace/ws_other/voxbox/checkpoints/"
wavs_fld = "/mnt/FastData/dl_workspace/ws_other/voxbox/datasets/resampled/ds_v2/"
transcripts_fld = "/mnt/FastData/dl_workspace/ws_other/voxbox/datasets/transcripts/ds_v2/"

speakers_config_path = "/mnt/FastData/dl_workspace/ws_other/voxbox/datasets/speakers_config.json"


@dataclass
class RoDatasetConfig(BaseDatasetConfig):
    used_speakers: List[str] = None


dataset_config = RoDatasetConfig(
    name="voxbox",
    meta_file_train=f"{transcripts_fld}/metadata_train_filtered.txt",
    meta_file_val=f"{transcripts_fld}/metadata_test_filtered.txt",
    path=wavs_fld,
    language="ro",
    ignored_speakers=["tss"],
    used_speakers=["adr", "mara", "ele", "sgs", "pmm", "pss", "fds", "eme", "cau"]
)
audio_config = BaseAudioConfig(
    sample_rate=16000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=False,
    trim_db=23.0,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
    resample=False,
)

vitsArgs = VitsArgs(
    use_language_embedding=False,
    use_speaker_embedding=True,
    use_sdp=True,
    spec_segment_size=32
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_voxbox",
    use_speaker_embedding=True,
    batch_size=28,
    eval_batch_size=16,
    batch_group_size=10,
    start_by_longest=False,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100,
    text_cleaner="romanian_cleaners",
    use_phonemes=False,
    lr_disc=0.0002,
    lr_gen=0.0002,
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    sort_by_audio_len=True,
    min_audio_len=8000,
    max_audio_len=180000,
    output_path=output_path,
    datasets=[dataset_config],
    characters=CharactersConfig(
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZÂÎĂȘȚabcdefghijklmnopqrstuvwxyzâîășț",
        punctuations="!',-.? ",
        is_unique=True
    ),
    test_sentences=[
        ["Vreau să zic că-mi place despre tine."],
        ["Deși lumea zice că voxbox e o aroganță, noi știm mai bine."],
        ["Îmi pare rău, Alex. Nu cred că pot să fac asta."],
        ["Crezi că te descurci cu logistica?"],
        ["Înainte de noiembrie douăzeci și doi, o mie nouă sute șaizeci și cinci, lumea era mai veselă."],
        ["Cele cinci sute cincizeci și cinci de ciuperci ciuruite pentru ciorbă"],
        [
            "Duc în bac sac de dac, aud crac, o fi rac? "
            "O fi drac? Face pac, aud mac, aud oac, nu e rac, nu-i gândac, nu e cuc, nu-i brotac, îl apuc, îl hurduc. "
            "E tot drac."
        ],
        [
            "Unilateralitatea colocviilor desolidarizează conștinciozitatea energeticienilor care manifestă o "
            "imperturbabilitate indiscriptibilă în locul nabucodonosorienei ireproșabilități."
        ],
        [
            "O mierliță fuflendiță fuflendi-fuflendariță nu poate să fuflendească fuflendi-fluflendărească "
            "pe mierloiul fuflendoiul fuflendi-fluflendaroiul."
        ],
        [
            "Dar mierloiul fuflendoiul fuflendi-fluflendaroiul poate că să fuflendească "
            "fuflendi-fluflendărească pe mierlița fuflendița fuflendi-fuflendărița"
        ],
        [
            "Înțeleg că dorești să vorbești cu un operator uman. "
            "Dar te rog hai să încercăm să o rezolvăm împreună, că mă scot ăștia din priză dacă nu."
        ],
        ["Apăsați tasta doi pentru a afla promoțiile pe care vi le-am pregătit."],
        [
            'Sunteți in căutarea unui smartphone? '
            'Descoperiți în magazinele noastre ofertele promoționale special pentru dumneavoastră.'
        ]
    ],

)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

tokenizer, config = TTSTokenizer.init_from_config(config)

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
config.model_args.num_speakers = speaker_manager.num_speakers

# init model
model = Vits(config, ap, tokenizer, speaker_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
