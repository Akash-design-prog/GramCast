"""Text-to-speech for advisory text, per the guide's own framing (section 8b): "A standard
text-to-speech API... reading out the same advisory text is enough for the pilot" - but explicitly in
the LOCAL LANGUAGE, not English (the guide's own example is a Hindi sentence). Since the full advisory
text is a small, fixed, enumerable set (~10 strings total from backend/advisory/rules.py), it's
hand-translated here rather than run through a live translation API - cheaper, more reliable, and
reviewable, at the cost of only covering exactly those known strings.

HONEST LIMITATION, stated plainly: these Hindi/Marathi translations were written without a native
speaker review. They're a genuine best-effort at correct, natural phrasing, not a certified translation
- flag this to a Marathi/Hindi speaker before this reaches a real farmer, same as any other pilot-stage
content. Falls back to the English text (rather than failing) for any advisory string not in the table -
this only happens if backend/advisory/rules.py's text changes without updating this table alongside it.

HYBRID ENGINE, per Akash's request to prefer government infrastructure where practical: if Bhashini
(bhashini_tts.py) is configured (BHASHINI_USER_ID/BHASHINI_API_KEY env vars set - registration needs
DIBD team approval, not instant), synthesize_speech() tries it FIRST. Any failure there (not
configured, network error, approval not through yet, malformed response) falls back to gTTS silently -
a farmer getting a gTTS voice note is far better than the endpoint failing because a government API
had a bad day. The engine actually used is returned alongside the audio so callers/logs can tell which
one ran, without that ever being a caller-visible failure mode.
"""
import io
import logging

from gtts import gTTS

from bhashini_tts import BhashiniError, is_configured as bhashini_is_configured, synthesize_speech_bhashini

logger = logging.getLogger(__name__)

# gTTS language codes this project supports - 'mr' (Marathi, Maharashtra's state language) is the
# primary target per the Pune pilot; 'hi' (Hindi) as a widely-understood alternative; 'en' as the
# always-available fallback.
SUPPORTED_LANGUAGES = {"en", "hi", "mr"}

# English advisory_text (exactly as backend/advisory/rules.py produces it) -> {lang: translated text}
_TRANSLATIONS: dict[str, dict[str, str]] = {
    "Heavy rain likely. Delay spraying and fertiliser application; clear field drainage; secure harvested produce.": {
        "hi": "भारी बारिश की संभावना है। छिड़काव और खाद डालना टालें; खेत की नालियाँ साफ करें; कटी हुई फसल सुरक्षित रखें।",
        "mr": "जोरदार पावसाची शक्यता आहे. फवारणी आणि खत टाकणे पुढे ढकला; शेतातील पाणी वाहून जाण्याचा मार्ग मोकळा ठेवा; काढलेले पीक सुरक्षित ठेवा.",
    },
    "Moderate rain likely. Good window for sowing or transplanting where soil allows.": {
        "hi": "मध्यम बारिश की संभावना है। जहाँ मिट्टी अनुकूल हो वहाँ बुवाई या रोपाई के लिए अच्छा समय है।",
        "mr": "मध्यम पावसाची शक्यता आहे. जमीन योग्य असल्यास पेरणी किंवा लावणीसाठी चांगली वेळ आहे.",
    },
    "Light rain likely. Generally safe for field work; check conditions before spraying.": {
        "hi": "हल्की बारिश की संभावना है। खेत का काम करना सामान्यतः सुरक्षित है; छिड़काव से पहले स्थिति जाँच लें।",
        "mr": "हलक्या पावसाची शक्यता आहे. शेतातील काम करणे साधारणपणे सुरक्षित आहे; फवारणीपूर्वी परिस्थिती तपासा.",
    },
    "Little or no rain expected today. Normal field operations; irrigate if soil is dry.": {
        "hi": "आज बारिश की संभावना कम या नहीं है। सामान्य खेती का काम करें; मिट्टी सूखी हो तो सिंचाई करें।",
        "mr": "आज पाऊस पडण्याची शक्यता कमी किंवा नाही. नेहमीप्रमाणे शेतीची कामे करा; जमीन कोरडी असल्यास पाणी द्या.",
    },
    "Dry spell expected over the next several days. Plan irrigation; good time for harvesting or drying.": {
        "hi": "अगले कई दिनों तक सूखा रहने की संभावना है। सिंचाई की योजना बनाएं; कटाई या सुखाने के लिए अच्छा समय है।",
        "mr": "पुढील अनेक दिवस कोरडे राहण्याची शक्यता आहे. पाण्याचे नियोजन करा; काढणी किंवा वाळवण्यासाठी चांगली वेळ आहे.",
    },
    "Heavy rain likely during flowering - risk of flower/pod drop. Delay spraying; provide drainage to protect the crop.": {
        "hi": "फूल आने के समय भारी बारिश की संभावना - फूल/फली गिरने का खतरा। छिड़काव टालें; फसल बचाने के लिए जल निकासी करें।",
        "mr": "फुलोऱ्याच्या वेळी जोरदार पावसाची शक्यता - फुले/शेंगा गळण्याचा धोका. फवारणी पुढे ढकला; पीक वाचवण्यासाठी पाणी वाहून जाण्याची व्यवस्था करा.",
    },
    "Heavy rain likely with crop ready for harvest - prioritise harvesting and securing produce before rain arrives if there is lead time; otherwise protect stored produce from moisture.": {
        "hi": "फसल कटाई के लिए तैयार है और भारी बारिश की संभावना है - समय हो तो बारिश से पहले कटाई और उपज सुरक्षित करना प्राथमिकता दें; नहीं तो भंडारित उपज को नमी से बचाएं।",
        "mr": "पीक काढणीसाठी तयार आहे आणि जोरदार पावसाची शक्यता आहे - वेळ असल्यास पावसाआधी काढणी करून पीक सुरक्षित ठेवण्यास प्राधान्य द्या; अन्यथा साठवलेले पीक ओलाव्यापासून वाचवा.",
    },
    "Moderate rain likely - favourable moisture window for sowing or transplanting.": {
        "hi": "मध्यम बारिश की संभावना - बुवाई या रोपाई के लिए अनुकूल नमी का समय।",
        "mr": "मध्यम पावसाची शक्यता - पेरणी किंवा लावणीसाठी अनुकूल ओलावा आहे.",
    },
    "Little or no rain expected - soil may be too dry for sowing; irrigate first or wait for rain.": {
        "hi": "बारिश की संभावना कम या नहीं है - बुवाई के लिए मिट्टी बहुत सूखी हो सकती है; पहले सिंचाई करें या बारिश का इंतज़ार करें।",
        "mr": "पाऊस पडण्याची शक्यता कमी किंवा नाही - पेरणीसाठी जमीन खूप कोरडी असू शकते; आधी पाणी द्या किंवा पावसाची वाट पाहा.",
    },
    "Dry conditions expected - good window for harvesting and drying the crop.": {
        "hi": "सूखा मौसम रहने की संभावना - फसल की कटाई और सुखाने के लिए अच्छा समय।",
        "mr": "कोरडे हवामान राहण्याची शक्यता - पीक काढणी आणि वाळवण्यासाठी चांगली वेळ आहे.",
    },
}


def translate_advisory(advisory_text: str, lang: str) -> str:
    """Returns advisory_text translated to `lang`, or the original English text if either the language
    isn't supported or this specific advisory string has no translation on file (graceful degradation,
    not a crash - a farmer getting an English voice note is far better than the endpoint failing)."""
    if lang == "en" or lang not in SUPPORTED_LANGUAGES:
        return advisory_text
    return _TRANSLATIONS.get(advisory_text, {}).get(lang, advisory_text)


def synthesize_speech(advisory_text: str, lang: str = "mr") -> tuple[bytes, str, str]:
    """Returns (audio_bytes, engine, media_type). engine is "bhashini" or "gtts" - useful for logging/
    demoing which one actually ran. media_type differs (Bhashini returns WAV, gTTS returns MP3), so
    callers must use the returned media_type, not assume one.

    Raises whatever gTTS raises on a real failure there too - if BOTH engines fail, the caller needs to
    know synthesis genuinely failed rather than silently getting nothing."""
    text = translate_advisory(advisory_text, lang)

    if bhashini_is_configured():
        try:
            audio = synthesize_speech_bhashini(text, lang)
            return audio, "bhashini", "audio/wav"
        except BhashiniError as e:
            logger.warning("Bhashini TTS failed, falling back to gTTS: %s", e)

    tts = gTTS(text, lang=lang if lang in SUPPORTED_LANGUAGES else "en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue(), "gtts", "audio/mpeg"
