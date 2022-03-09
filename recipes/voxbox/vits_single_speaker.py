import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.config import RoDatasetConfig

RESUME_EPOCH = -1
output_path = "/home/bogdan/workspace/ws_voxbox/checkpoints/"
wavs_fld = "/home/bogdan/workspace/ws_voxbox/datasets/resampled/ds_v2/"
transcripts_fld = "/home/bogdan/workspace/ws_voxbox/datasets/transcripts/ds_v2/"
speakers_config_path = "/home/bogdan/workspace/ws_voxbox/datasets/speakers_config.json"

dataset_config = RoDatasetConfig(
    name="voxbox",
    meta_file_train=f"{transcripts_fld}/metadata_train_filtered.txt",
    meta_file_val=f"{transcripts_fld}/metadata_test_filtered.txt",
    path=wavs_fld,
    language="ro",
    ignored_speakers=["tss"],
    used_speakers=["mara"]
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
    trim_db=45,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
)

config = VitsConfig(
    audio=audio_config,
    run_name="vits_mara",
    batch_size=16,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    lr_disc=0.0001,
    lr_gen=0.0001,
    lr=0.001,
    epochs=100,
    text_cleaner="romanian_cleaners",
    use_phonemes=False,
    phoneme_language="ro",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    min_audio_len=8000,
    max_audio_len=180000,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
     characters=CharactersConfig(
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂÎĂȘȚâîășț",
        punctuations="!'(),-.:;? ",
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

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

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
