import os

from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager

RESUME_EPOCH = -1
output_path = "/mnt/FastData/dl_workspace/ws_other/voxbox/checkpoints/"
wavs_fld = "/mnt/FastData/dl_workspace/ws_other/voxbox/datasets/resampled/ds_v2/"
transcripts_fld = "/mnt/FastData/dl_workspace/ws_other/voxbox/datasets/transcripts/ds_v2_dla"

speakers_config_path = "/mnt/FastData/dl_workspace/ws_other/voxbox/datasets/speakers_config.json"
dataset_config = BaseDatasetConfig(
    name="voxbox",
    meta_file_train=f"{transcripts_fld}/metadata_train.txt",
    meta_file_val=f"{transcripts_fld}/metadata_test.txt",
    path=wavs_fld, language="ro"
)
audio_config = BaseAudioConfig(
    sample_rate=16000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=23.0,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
)

config = VitsConfig(
    audio=audio_config,
    run_name="vits_voxbox",
    batch_size=10,
    eval_batch_size=12,
    batch_group_size=5,
    num_loader_workers=1,
    num_eval_loader_workers=1,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=200,
    text_cleaner="romanian_cleaners",
    use_phonemes=False,
    phoneme_language="ro",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    save_step=1000,
    keep_after=1000,
    lr_gen=0.000001,
    lr_scheduler_gen_params={'gamma': 0.999875, "last_epoch": RESUME_EPOCH},
    lr_disc=0.000001,
    lr_scheduler_disc_params={'gamma': 0.999875, "last_epoch": RESUME_EPOCH},
    speakers_file=speakers_config_path,
    use_speaker_embedding=True,
    print_eval=True,
    mixed_precision=True,
    min_seq_len=14000,
    max_seq_len=180000,
    output_path=output_path,
    datasets=[dataset_config],
    characters=CharactersConfig(
        pad="_",
        eos="&",
        bos="*",
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZÂÎĂȘabcdefghijklmnopqrstțuvwxyzâîăș!'(),-.:;? ",
        punctuations="!'(),-.:;? ",
        unique=True
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

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)
meta_data = train_samples + eval_samples

# init model
model = Vits(config, speaker_manager=SpeakerManager(
    # data_items=meta_data,
    speaker_id_file_path=os.path.join('/mnt/FastData/dl_workspace/ws_other/voxbox/checkpoints/vits_ft_delia_5/speakers.json')
))
# speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
# config.model_args.num_speakers = speaker_manager.num_speakers

# init the trainer and 🚀
trainer = Trainer(
    args=TrainingArgs(),
    config=config,
    output_path=output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
    parse_command_line_args=True
)
trainer.fit()
