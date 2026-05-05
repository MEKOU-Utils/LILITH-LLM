use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};
use std::thread;
use rodio::{Decoder, OutputStream, Sink};

pub struct TalkAi {
    sink: Arc<Mutex<Option<Sink>>>,
    _stream: Arc<Mutex<Option<OutputStream>>>,
    wav_cache: Arc<HashMap<String, Vec<u8>>>,
}

impl TalkAi {
    pub fn new(teto_dir: &str) -> Self {
        // Init audio output
        let (stream, stream_handle) = OutputStream::try_default().unwrap();
        let sink = Sink::try_new(&stream_handle).unwrap();
        
        // Preload memory
        let mut cache = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(teto_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("wav") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        let name = stem.trim_start_matches('_').to_string();
                        if let Ok(data) = std::fs::read(&path) {
                            cache.insert(name, data);
                        }
                    }
                }
            }
        }
        
        Self {
            sink: Arc::new(Mutex::new(Some(sink))),
            _stream: Arc::new(Mutex::new(Some(stream))),
            wav_cache: Arc::new(cache),
        }
    }

    /// Converts a Japanese string into an array of romaji strings matching the wav files
    fn tokenize_kana(text: &str) -> Vec<String> {
        let mapping = [
            ("きゃ", "kya"), ("きゅ", "kyu"), ("きょ", "kyo"), ("きぇ", "kye"),
            ("ぎゃ", "gya"), ("ぎゅ", "gyu"), ("ぎょ", "gyo"), ("ぎぇ", "gye"),
            ("しゃ", "sya"), ("しゅ", "syu"), ("しょ", "syo"), ("しぇ", "sye"),
            ("じゃ", "zya"), ("じゅ", "zyu"), ("じょ", "zyo"), ("じぇ", "zye"),
            ("ちゃ", "tya"), ("ちゅ", "tyu"), ("ちょ", "tyo"), ("ちぇ", "tye"),
            ("にゃ", "nya"), ("にゅ", "nyu"), ("にょ", "nyo"), ("にぇ", "nye"),
            ("ひゃ", "hya"), ("ひゅ", "hyu"), ("ひょ", "hyo"), ("ひぇ", "hye"),
            ("びゃ", "bya"), ("びゅ", "byu"), ("びょ", "byo"), ("びぇ", "bye"),
            ("ぴゃ", "pya"), ("ぴゅ", "pyu"), ("ぴょ", "pyo"), ("ぴぇ", "pye"),
            ("みゃ", "mya"), ("みゅ", "myu"), ("みょ", "myo"), ("みぇ", "mye"),
            ("りゃ", "rya"), ("りゅ", "ryu"), ("りょ", "ryo"), ("りぇ", "rye"),
            ("ヴぁ", "va"), ("ヴぃ", "vi"), ("ヴぇ", "ve"), ("ヴぉ", "vo"),
            ("あ", "a"), ("い", "i"), ("う", "u"), ("え", "e"), ("お", "o"),
            ("か", "ka"), ("き", "ki"), ("く", "ku"), ("け", "ke"), ("こ", "ko"),
            ("が", "ga"), ("ぎ", "gi"), ("ぐ", "gu"), ("げ", "ge"), ("ご", "go"),
            ("さ", "sa"), ("し", "si"), ("す", "su"), ("せ", "se"), ("そ", "so"),
            ("ざ", "za"), ("じ", "zi"), ("ず", "zu"), ("ぜ", "ze"), ("ぞ", "zo"),
            ("た", "ta"), ("ち", "ti"), ("つ", "tu"), ("て", "te"), ("と", "to"),
            ("だ", "da"), ("ぢ", "di"), ("づ", "du"), ("で", "de"), ("ど", "do"),
            ("な", "na"), ("に", "ni"), ("ぬ", "nu"), ("ね", "ne"), ("の", "no"),
            ("は", "ha"), ("ひ", "hi"), ("ふ", "fu"), ("へ", "he"), ("ほ", "ho"),
            ("ば", "ba"), ("び", "bi"), ("ぶ", "bu"), ("べ", "be"), ("ぼ", "bo"),
            ("ぱ", "pa"), ("ぴ", "pi"), ("ぷ", "pu"), ("ぺ", "pe"), ("ぽ", "po"),
            ("ま", "ma"), ("み", "mi"), ("む", "mu"), ("め", "me"), ("も", "mo"),
            ("や", "ya"), ("ゆ", "yu"), ("よ", "yo"),
            ("ら", "ra"), ("り", "ri"), ("る", "ru"), ("れ", "re"), ("ろ", "ro"),
            ("わ", "wa"), ("を", "wo"), ("ん", "nn"),
        ];

        let mut result = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();
        while i < chars.len() {
            let mut matched = false;
            if i + 1 < chars.len() {
                let s: String = chars[i..=i+1].iter().collect();
                for (k, v) in mapping.iter() {
                    if *k == s {
                        result.push(v.to_string());
                        matched = true;
                        i += 2;
                        break;
                    }
                }
            }
            if !matched {
                let s = chars[i].to_string();
                for (k, v) in mapping.iter() {
                    if *k == s {
                        result.push(v.to_string());
                        matched = true;
                        break;
                    }
                }
                i += 1;
            }
        }
        result
    }

    /// Speaks the given text asynchronously
    pub fn speak(&self, text: &str) {
        let tokens = Self::tokenize_kana(text);
        let cache = Arc::clone(&self.wav_cache);
        let sink_lock = Arc::clone(&self.sink);

        thread::spawn(move || {
            if let Ok(guard) = sink_lock.lock() {
                if let Some(sink) = &*guard {
                    for token in tokens {
                        if let Some(data) = cache.get(&token) {
                            let cursor = std::io::Cursor::new(data.clone());
                            if let Ok(decoder) = Decoder::new(cursor) {
                                sink.append(decoder);
                            }
                        }
                    }
                }
            }
        });
    }
}
