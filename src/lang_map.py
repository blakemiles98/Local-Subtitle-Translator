WHISPER_TO_NLLB = {
    # Afrikaans, Amharic, Arabic
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "ar": "arb_Arab",

    # Assamese, Azerbaijani
    "as": "asm_Beng",
    "az": "azj_Latn",

    # Bashkir, Belarusian, Bulgarian, Bengali, Tibetan, Breton, Bosnian
    "ba": "bak_Cyrl",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "bo": "bod_Tibt",
    "br": "bre_Latn",
    "bs": "bos_Latn",

    # Catalan, Czech, Welsh, Danish, German, Greek, English, Spanish, Estonian, Basque
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",

    # Persian, Finnish, Faroese, French, Galician
    "fa": "pes_Arab",   # NLLB uses pes_Arab for Persian
    "fi": "fin_Latn",
    "fo": "fao_Latn",
    "fr": "fra_Latn",
    "gl": "glg_Latn",

    # Gujarati, Hausa, Hawaiian, Hebrew, Hindi, Croatian, Haitian Creole, Hungarian, Armenian
    "gu": "guj_Gujr",
    "ha": "hau_Latn",
    "haw": "haw_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "ht": "hat_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",

    # Indonesian, Icelandic, Italian, Japanese, Javanese, Georgian, Kazakh, Khmer, Kannada, Korean
    "id": "ind_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "jv": "jav_Latn",   # some Whisper builds use "jw"; see below
    "jw": "jav_Latn",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",

    # Latin, Luxembourgish, Lingala, Lao, Lithuanian, Latvian
    "la": "lat_Latn",
    "lb": "ltz_Latn",
    "ln": "lin_Latn",
    "lo": "lao_Laoo",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",   # Latvian in NLLB is lvs_Latn

    # Malagasy, Maori, Macedonian, Malayalam, Mongolian, Marathi, Malay, Maltese, Burmese
    "mg": "plt_Latn",   # Malagasy (Plateau Malagasy) is plt_Latn in NLLB
    "mi": "mri_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",   # Halh Mongolian
    "mr": "mar_Deva",
    "ms": "zsm_Latn",   # IMPORTANT: fixes your 'ms' warning (Malay -> Standard Malay)
    "mt": "mlt_Latn",
    "my": "mya_Mymr",

    # Nepali, Dutch
    "ne": "npi_Deva",
    "nl": "nld_Latn",

    # Norwegian (Bokmål/Generic) + Norwegian Nynorsk
    "no": "nob_Latn",
    "nb": "nob_Latn",
    "nn": "nno_Latn",   # IMPORTANT: fixes your 'nn' warning

    # Occitan, Punjabi, Polish, Pashto, Portuguese, Romanian, Russian
    "oc": "oci_Latn",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "ps": "pbt_Arab",   # Pashto
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",

    # Sanskrit, Sindhi, Sinhala, Slovak, Slovenian, Shona, Somali, Albanian, Serbian, Sundanese, Swedish, Swahili
    "sa": "san_Deva",
    "sd": "snd_Arab",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sn": "sna_Latn",
    "so": "som_Latn",
    "sq": "als_Latn",   # Albanian in NLLB is als_Latn
    "sr": "srp_Cyrl",   # default to Cyrillic (you can switch to srp_Latn if you prefer)
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",

    # Tamil, Telugu, Tajik, Thai, Turkmen, Tagalog, Turkish, Tatar, Ukrainian, Urdu, Uzbek, Vietnamese
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "tg": "tgk_Cyrl",
    "th": "tha_Thai",
    "tk": "tuk_Latn",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "tt": "tat_Cyrl",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",

    # Yiddish, Yoruba
    "yi": "ydd_Hebr",
    "yo": "yor_Latn",

    # Chinese + Cantonese
    "zh": "zho_Hans",   # default Simplified; switch to zho_Hant if you want Traditional default
    "yue": "yue_Hant",  # Cantonese (if your Whisper build returns yue)
}