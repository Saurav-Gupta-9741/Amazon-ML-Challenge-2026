"""
transliterate.py — High-speed Multi-Script Indic to Latin Phonetic Romanizer.

Handles Hindi, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, and Gurmukhi
business entity names and address components in Source 2 and 3.
Zero external dependencies (pure Python).
"""

import re
import unicodedata

# =========================================================================
# AUTHENTIC REGIONAL BUSINESS TERMS TO CANONICAL ENGLISH
# =========================================================================

HINDI_BUSINESS_TERMS = {
    'प्राइवेट लिमिटेड': 'pvt ltd', 'प्राइवेट': 'pvt', 'प्रा. लि.': 'pvt ltd',
    'प्रा लि': 'pvt ltd', 'लिमिटेड': 'ltd', 'लि.': 'ltd', 'एलएलपी': 'llp',
    'कंपनी': 'company', 'एंटरप्राइजेज': 'enterprises', 'इंटरप्राइजेज': 'enterprises',
    'उद्योग': 'udhyog', 'सॉल्यूशंस': 'solutions', 'सोल्यूशंस': 'solutions',
    'टेक्नोलॉजीज': 'technologies', 'टेक्नोलॉजी': 'technology',
    'कंस्ट्रक्शन': 'constructions', 'कंस्ट्रक्शंस': 'constructions',
    'इंफ्रास्ट्रक्चर': 'infrastructure', 'मार्केटिंग': 'marketing',
    'ट्रेडर्स': 'traders', 'ट्रेडिंग': 'trading', 'सर्विसेज': 'services',
    'सर्विस': 'services', 'एसोसिएट्स': 'associates', 'प्रॉपर्टीज': 'properties',
    'डेवलपर्स': 'developers', 'सिक्योरिटी': 'security', 'इंटरनेशनल': 'international',
    'ग्लोबल': 'global', 'इन्वेस्टमेंट': 'investment', 'इन्वेस्टमेंट्स': 'investments',
    'फाइनेंस': 'finance', 'फूड्स': 'foods', 'फूड': 'food',
    'हॉस्पिटल': 'hospital', 'हेल्थकेयर': 'healthcare', 'फार्मा': 'pharma',
    'फार्मास्यूटिकल्स': 'pharmaceuticals', 'एजेंसी': 'agency', 'एजेंसीज': 'agencies',
    'ऑटोमोबाइल्स': 'automobiles', 'इलेक्ट्रिकल्स': 'electricals',
    'इलेक्ट्रॉनिक्स': 'electronics', 'फैशन': 'fashion', 'टेक्सटाइल्स': 'textiles',
    'स्टोर': 'store', 'मार्ट': 'mart', 'बाजार': 'bazaar', 'केंद्र': 'kendra',
    'संस्थान': 'sansthan', 'फाउंडेशन': 'foundation', 'ट्रस्ट': 'trust',
}

TAMIL_BUSINESS_TERMS = {
    'பிரைவேட் லிமிடெட்': 'pvt ltd', 'பிரைவேட் லிமிட்டெட்': 'pvt ltd',
    'பிரைவேட் லிமிட்டட்': 'pvt ltd', 'பிரைவேட்': 'pvt', 'லிமிடெட்': 'ltd',
    'லிமிட்டெட்': 'ltd', 'எல்எல்பி': 'llp', 'எல்.எல்.பி': 'llp', 'எல் எல் பி': 'llp',
    'கம்பெனி': 'company', 'என்டர்பிரைசஸ்': 'enterprises', 'என்டர்பிரைஸஸ்': 'enterprises',
    'இன்வெஸ்ட்மெண்ட்ஸ்': 'investments', 'இன்வெஸ்ட்மென்ட்ஸ்': 'investments',
    'இன்வெஸ்ட்மெண்ட்': 'investment', 'இன்வெஸ்ட்மென்ட்': 'investment',
    'சொல்யூஷன்ஸ்': 'solutions', 'சொல்யூசன்ஸ்': 'solutions',
    'டெக்னாலஜிஸ்': 'technologies', 'டெக்னாலஜீஸ்': 'technologies',
    'இண்டஸ்ட்ரீஸ்': 'industries', 'இண்டஸ்ட்ரிஸ்': 'industries',
    'கன்ஸ்ட்ரக்ஷன்ஸ்': 'constructions', 'கன்ஸ்ட்ரக்ஷன்': 'constructions',
    'இன்போசிஸ்': 'infosys', 'சர்வீசஸ்': 'services', 'சர்வீஸ்': 'services',
    'டிரேடிங்': 'trading', 'டிரேடர்ஸ்': 'traders', 'அசோசியேட்ஸ்': 'associates',
    'ப்ராபர்ட்டீஸ்': 'properties', 'பிராபர்டீஸ்': 'properties',
    'டெவலப்பர்ஸ்': 'developers', 'செக்யூரிட்டி': 'security',
    'இன்டர்நேஷனல்': 'international', 'குளோபல்': 'global', 'பைனான்ஸ்': 'finance',
    'பினான்ஸ்': 'finance', 'புட்ஸ்': 'foods', 'ஹாஸ்பிடல்': 'hospital',
    'ஹெல்த்கேர்': 'healthcare', 'பார்மா': 'pharma', 'ஏஜென்சி': 'agency',
    'ஆட்டோமொபைல்ஸ்': 'automobiles', 'எலக்ட்ரிக்கல்ஸ்': 'electricals',
    'எலக்ட்ரானிக்ஸ்': 'electronics', 'டெக்ஸ்டைல்ஸ்': 'textiles', 'ஸ்டோர்': 'store',
    'மார்ட்': 'mart', 'பஜார்': 'bazaar', 'பவுண்டேஷன்': 'foundation', 'டிரஸ்ட்': 'trust',
}

TELUGU_BUSINESS_TERMS = {
    'ప్రైవేట్ లిమిటెడ్': 'pvt ltd', 'ప్రైవేట్': 'pvt', 'లిమిటెడ్': 'ltd',
    'ఎల్ఎల్పీ': 'llp', 'ఎల్.ఎల్.పి': 'llp', 'కంపెనీ': 'company',
    'ఎంటర్‌ప్రైజెస్': 'enterprises', 'ఎంటర్ప్రైజెస్': 'enterprises',
    'ఇన్వెస్ట్‌మెంట్స్': 'investments', 'ఇన్వెస్ట్‌మెంట్': 'investment',
    'ఇన్వెస్ట్మెంట్': 'investment', 'సొల్యూషన్స్': 'solutions',
    'టెక్నాలజీస్': 'technologies', 'ఇండస్ట్రీస్': 'industries',
    'కన్‌స్ట్రక్షన్స్': 'constructions', 'మార్కెటింగ్': 'marketing',
    'ట్రేడర్స్': 'traders', 'ట్రేడింగ్': 'trading', 'సర్వీసెస్': 'services',
    'అసోసియేట్స్': 'associates', 'ప్రాపర్టీస్': 'properties',
    'డెవలపర్స్': 'developers', 'ఇంటర్నేషనల్': 'international',
    'గ్లోబల్': 'global', 'ఫైనాన్స్': 'finance', 'హాస్పిటల్': 'hospital',
    'హెల్త్‌కేర్': 'healthcare', 'ఫార్మా': 'pharma', 'ఏజెన్సీ': 'agency',
    'ఆటోమొబైల్స్': 'automobiles', 'ఎలక్ట్రానిక్స్': 'electronics',
    'ఫౌండేషన్': 'foundation', 'ట్రస్ట్': 'trust',
}

KANNADA_BUSINESS_TERMS = {
    'ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್': 'pvt ltd', 'ಪ್ರೈವೇಟ್': 'pvt', 'ಲಿಮಿಟೆಡ್': 'ltd',
    'ಎಲ್‌ಎಲ್‌ಪಿ': 'llp', 'ಎಲ್ಎಲ್ಪಿ': 'llp', 'ಕಂಪನಿ': 'company',
    'ಎಂಟರ್‌ಪ್ರೈಸಸ್': 'enterprises', 'ಇನ್ವೆಸ್ಟ್‌ಮೆಂಟ್ಸ್': 'investments',
    'ಇನ್ವೆಸ್ಟ್ಮೆಂಟ್ಸ್': 'investments', 'ಇನ್ವೆಸ್ಟ್‌ಮೆಂಟ್': 'investment',
    'ಇನ್ವೆಸ್ಟ್ಮೆಂಟ್': 'investment', 'ಸೊಲ್ಯೂಷನ್ಸ್': 'solutions',
    'ಟೆಕ್ನಾಲಜೀಸ್': 'technologies', 'ಇಂಡಸ್ಟ್ರೀಸ್': 'industries',
    'ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ಸ್': 'constructions', 'ಟ್ರೇಡರ್ಸ್': 'traders',
    'ಟ್ರೇಡಿಂಗ್': 'trading', 'ಸರ್ವಿಸಸ್': 'services', 'ಅಸೋಸಿಯೇಟ್ಸ್': 'associates',
    'ಪ್ರಾಪರ್ಟೀಸ್': 'properties', 'ಡೆವಲಪರ್ಸ್': 'developers',
    'ಇಂಟರ್‌ನ್ಯಾಷನಲ್': 'international', 'ಗ್ಲೋಬಲ್': 'global', 'ಫೈನಾನ್ಸ್': 'finance',
    'ಹಾಸ್ಪಿಟಲ್': 'hospital', 'ಹೆಲ್ತ್‌ಕೇರ್': 'healthcare', 'ಫಾರ್ಮಾ': 'pharma',
    'ಫೌಂಡೇಶನ್': 'foundation', 'ಟ್ರಸ್ಟ್': 'trust',
}

GUJARATI_BUSINESS_TERMS = {
    'પ્રાઇવેટ લિમિટેડ': 'pvt ltd', 'પ્રાઇવેટ': 'pvt', 'લિમીટેડ': 'ltd',
    'લિમિટેડ': 'ltd', 'એલએલપી': 'llp', 'એલ.એલ.પી': 'llp', 'કંપની': 'company',
    'એન્ટરપ્રાઇઝિસ': 'enterprises', 'એન્ટરપ્રાઇઝ': 'enterprises',
    'ઇન્વેસ્ટમેન્ટ્સ': 'investments', 'ઇન્વેસ્ટમેન્ટ': 'investment',
    'સોલ્યુશન્સ': 'solutions', 'ટેકનોલોજી': 'technology', 'ટેકનોલોજીસ': 'technologies',
    'ઇન્ડસ્ટ્રીઝ': 'industries', 'કન્સ્ટ્રક્શન': 'constructions',
    'ટ્રેડર્સ': 'traders', 'ટ્રેડિંગ': 'trading', 'સર્વિસીસ': 'services',
    'એસોસિએટ્સ': 'associates', 'પ્રોપર્ટીઝ': 'properties', 'ડેવલપર્સ': 'developers',
    'ઇન્ટરનેશનલ': 'international', 'ગ્લોબલ': 'global', 'ફાઇનાન્સ': 'finance',
    'હોસ્પિટલ': 'hospital', 'હેલ્થકેર': 'healthcare', 'ફાર્મા': 'pharma',
    'ફાઉન્ડેશન': 'foundation', 'ટ્રસ્ટ': 'trust',
}

BENGALI_BUSINESS_TERMS = {
    'প্রাইভেট লিমিটেড': 'pvt ltd', 'প্রাইভেট': 'pvt', 'লিমিটেড': 'ltd',
    'এলএলপি': 'llp', 'এল.এল.পি': 'llp', 'কোম্পানি': 'company',
    'এন্টারপ্রাইজ': 'enterprises', 'এন্টারপ্রাইজেস': 'enterprises',
    'ইনভেস্টমেন্টস': 'investments', 'ইনভেস্টমেন্ট': 'investment',
    'সলিউশনস': 'solutions', 'সলিউশন': 'solutions', 'টেকনোলজিস': 'technologies',
    'ইন্ডাস্ট্রিজ': 'industries', 'কনস্ট্রাকশন': 'constructions',
    'ট্রেডার্স': 'traders', 'ট্রেডিং': 'trading', 'সার্ভিসেস': 'services',
    'অ্যাসোসিয়েটস': 'associates', 'প্রপার্টিজ': 'properties',
    'ডেভেলপার্স': 'developers', 'আন্তর্জাতিক': 'international',
    'গ্লোবাল': 'global', 'ফাইন্যান্স': 'finance', 'হাসপাতাল': 'hospital',
    'হেলথকেয়ার': 'healthcare', 'ফার্মা': 'pharma', 'ট্রাস্ট': 'trust',
}

GURMUKHI_BUSINESS_TERMS = {
    'ਪ੍ਰਾਈਵੇਟ ਲਿਮਿਟੇਡ': 'pvt ltd', 'ਪ੍ਰਾਈਵੇਟ': 'pvt', 'ਲਿਮਿਟੇਡ': 'ltd',
    'ਐੱਲਐੱਲਪੀ': 'llp', 'ਐਲ.ਐਲ.ਪੀ': 'llp', 'ਕੰਪਨੀ': 'company',
    'ਇੰਟਰਪ੍ਰਾਈਜਿਜ਼': 'enterprises', 'ਇਨਵੈਸਟਮੈਂਟਸ': 'investments',
    'ਇਨਵੈਸਟਮੈਂਟ': 'investment', 'ਸਲਿਊਸ਼ਨਜ਼': 'solutions',
    'ਤਕਨਾਲੋਜੀ': 'technology', 'ਇੰਡਸਟਰੀਜ਼': 'industries',
    'ਕੰਸਟਰੱਕਸ਼ਨ': 'constructions', 'ਟਰੇਡਰਜ਼': 'traders', 'ਟਰੇਡਿੰਗ': 'trading',
    'ਸਰਵਿਸਿਜ਼': 'services', 'ਐਸੋਸੀਏਟਸ': 'associates', 'ਪ੍ਰਾਪਰਟੀਜ਼': 'properties',
    'ਡਿਵੈਲਪਰਜ਼': 'developers', 'ਇੰਟਰਨੈਸ਼ਨਲ': 'international', 'ਗਲੋਬਲ': 'global',
    'ਫਾਇਨਾਂਸ': 'finance', 'ਹਸਪਤਾਲ': 'hospital', 'ਟਰੱਸਟ': 'trust',
}

MALAYALAM_BUSINESS_TERMS = {
    'പ്രൈവറ്റ് ലിമിറ്റഡ്': 'pvt ltd', 'പ്രൈവറ്റ്': 'pvt', 'ലിമിറ്റഡ്': 'ltd',
    'എൽഎൽപി': 'llp', 'കമ്പനി': 'company', 'എന്റർപ്രൈസസ്': 'enterprises',
    'ഇൻവെസ്റ്റ്മെന്റ്സ്': 'investments', 'ഇൻവെസ്റ്റ്മെന്റ്': 'investment',
    'സൊല്യൂഷൻസ്': 'solutions', 'ടെക്നോളജീസ്': 'technologies',
    'ഇൻഡസ്ട്രീസ്': 'industries', 'കൺസ്ട്രക്ഷൻസ്': 'constructions',
    'ട്രേഡേഴ്സ്': 'traders', 'ട്രേഡിംഗ്': 'trading', 'സർവീസസ്': 'services',
    'അസോസിയേറ്റ്സ്': 'associates', 'പ്രോപ്പർട്ടീസ്': 'properties',
    'ഇന്റർനാഷണൽ': 'international', 'ഗ്ലോബൽ': 'global', 'ഫിനാൻസ്': 'finance',
    'ഹോസ്പിറ്റൽ': 'hospital', 'ഹെൽത്ത്കെയർ': 'healthcare', 'ട്രസ്റ്റ്': 'trust',
}

# =========================================================================
# INDIAN STATES IN REGIONAL SCRIPTS
# =========================================================================

INDIC_STATES = {
    # Hindi
    'महाराष्ट्र': 'maharashtra', 'मध्य प्रदेश': 'madhya pradesh', 'उत्तर प्रदेश': 'uttar pradesh',
    'राजस्थान': 'rajasthan', 'गुजरात': 'gujarat', 'कर्नाटक': 'karnataka',
    'तमिलनाडु': 'tamil nadu', 'पश्चिम बंगाल': 'west bengal', 'बिहार': 'bihar',
    'पंजाब': 'punjab', 'हरियाणा': 'haryana', 'दिल्ली': 'delhi', 'नई दिल्ली': 'new delhi',
    'ओडिशा': 'odisha', 'केरल': 'kerala', 'आंध्र प्रदेश': 'andhra pradesh',
    'तेलंगाना': 'telangana', 'झारखंड': 'jharkhand', 'छत्तीसगढ़': 'chhattisgarh',
    'उत्तराखंड': 'uttarakhand', 'हिमाचल प्रदेश': 'himachal pradesh', 'असम': 'assam', 'गोवा': 'goa',
    # Tamil
    'தமிழ்நாடு': 'tamil nadu', 'சென்னை': 'chennai', 'கர்நாடகா': 'karnataka',
    'கேரளா': 'kerala', 'ஆந்திரா': 'andhra pradesh', 'தெலுங்கானா': 'telangana',
    'மகாராஷ்டிரா': 'maharashtra', 'டெல்லி': 'delhi', 'மும்பை': 'mumbai',
    # Telugu
    'తెలంగాణ': 'telangana', 'ఆంధ్ర ప్రదేశ్': 'andhra pradesh', 'హైదరాబాద్': 'hyderabad',
    'కర్ణాటక': 'karnataka', 'మహారాష్ట్ర': 'maharashtra', 'తమిళనాడు': 'tamil nadu',
    # Kannada
    'ಕರ್ನಾಟಕ': 'karnataka', 'ಬೆಂಗಳೂರು': 'bangalore', 'ಮಹಾರಾಷ್ಟ್ರ': 'maharashtra',
    'ತಮಿಳುನಾಡು': 'tamil nadu', 'ಕೇರಳ': 'kerala',
    # Gujarati
    'ગુજરાત': 'gujarat', 'અમદાવાદ': 'ahmedabad', 'મહારાષ્ટ્ર': 'maharashtra', 'રાજસ્થાન': 'rajasthan',
    # Bengali
    'পশ্চিমবঙ্গ': 'west bengal', 'পশ্চিম বঙ্গ': 'west bengal', 'কলকাতা': 'kolkata',
    # Gurmukhi
    'ਪੰਜਾਬ': 'punjab', 'ਹਰਿਆਣਾ': 'haryana', 'ਚੰਡੀਗੜ੍ਹ': 'chandigarh',
    # Malayalam
    'കേരളം': 'kerala', 'കേരള': 'kerala', 'കൊച്ചി': 'kochi',
}

# Combine all business term dictionaries for fastest multi-script scan
ALL_INDIC_BUSINESS_TERMS = {}
for d in [
    TAMIL_BUSINESS_TERMS, TELUGU_BUSINESS_TERMS, KANNADA_BUSINESS_TERMS,
    GUJARATI_BUSINESS_TERMS, BENGALI_BUSINESS_TERMS, GURMUKHI_BUSINESS_TERMS,
    MALAYALAM_BUSINESS_TERMS, HINDI_BUSINESS_TERMS
]:
    ALL_INDIC_BUSINESS_TERMS.update(d)

# Sort by length descending to match longest phrases first (e.g. 'private limited' before 'private')
SORTED_PHRASES = sorted(ALL_INDIC_BUSINESS_TERMS.items(), key=lambda x: len(x[0]), reverse=True)
SORTED_STATES = sorted(INDIC_STATES.items(), key=lambda x: len(x[0]), reverse=True)

# =========================================================================
# CHARACTER-LEVEL PHONETIC MAPPINGS (DEVANAGARI BASE + DRAVIDIAN EXTENSIONS)
# =========================================================================

DEV_VOWELS = {
    '\u0905': 'a', '\u0906': 'aa', '\u0907': 'i', '\u0908': 'ee', '\u0909': 'u',
    '\u090a': 'oo', '\u090b': 'ri', '\u090e': 'e', '\u090f': 'e', '\u0910': 'ai',
    '\u0912': 'o', '\u0913': 'o', '\u0914': 'au',
}

DEV_MATRAS = {
    '\u093e': 'a', '\u093f': 'i', '\u0940': 'ee', '\u0941': 'u', '\u0942': 'oo',
    '\u0943': 'ri', '\u0946': 'e', '\u0947': 'e', '\u0948': 'ai',
    '\u094a': 'o', '\u094b': 'o', '\u094c': 'au',
}

DEV_CONSONANTS = {
    '\u0915': 'k', '\u0916': 'kh', '\u0917': 'g', '\u0918': 'gh', '\u0919': 'ng',
    '\u091a': 'ch', '\u091b': 'chh', '\u091c': 'j', '\u091d': 'jh', '\u091e': 'ny',
    '\u091f': 't', '\u0920': 'th', '\u0921': 'd', '\u0922': 'dh', '\u0923': 'n',
    '\u0924': 't', '\u0925': 'th', '\u0926': 'd', '\u0927': 'dh', '\u0928': 'n',
    '\u0929': 'n',  # Tamil alveolar n (0x0ba9 - 0x280)
    '\u092a': 'p', '\u092b': 'ph', '\u092c': 'b', '\u092d': 'bh', '\u092e': 'm',
    '\u092f': 'y', '\u0930': 'r',
    '\u0931': 'r',  # Tamil alveolar r (0x0bb1 - 0x280)
    '\u0932': 'l',
    '\u0933': 'l',  # Retroflex l (0x0bb3 - 0x280)
    '\u0934': 'l',  # Tamil retroflex zha (0x0bb4 - 0x280)
    '\u0935': 'v', '\u0936': 'sh', '\u0937': 'sh', '\u0938': 's', '\u0939': 'h',
    '\u0958': 'q', '\u0959': 'kh', '\u095a': 'gh', '\u095b': 'z', '\u095c': 'r',
    '\u095d': 'rh', '\u095e': 'f',
}

VIRAMA = '\u094d'  # Halant
ANUSVARA = '\u0902'  # Bindu
VISARGA = '\u0903'


def contains_indic(text: str) -> bool:
    """Fast check if string contains any Indic script character (0x0900 to 0x0D7F)."""
    if not text:
        return False
    return any('\u0900' <= ch <= '\u0d7f' for ch in text)


def get_script_offset(ch: str):
    """Returns Unicode offset to normalize regional Indic scripts to Devanagari equivalent."""
    code = ord(ch)
    if 0x0900 <= code <= 0x097F: return 0       # Devanagari
    if 0x0980 <= code <= 0x09FF: return 0x0080 # Bengali
    if 0x0A00 <= code <= 0x0A7F: return 0x0100 # Gurmukhi
    if 0x0A80 <= code <= 0x0AFF: return 0x0180 # Gujarati
    if 0x0B00 <= code <= 0x0B7F: return 0x0200 # Oriya
    if 0x0B80 <= code <= 0x0BFF: return 0x0280 # Tamil
    if 0x0C00 <= code <= 0x0C7F: return 0x0300 # Telugu
    if 0x0C80 <= code <= 0x0CFF: return 0x0380 # Kannada
    if 0x0D00 <= code <= 0x0D7F: return 0x0400 # Malayalam
    return None


def transliterate_indic_word(word: str) -> str:
    """Transliterates a single Indic word into Latin phonetic text."""
    out = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        offset = get_script_offset(ch)

        if offset is not None:
            base_ch = chr(ord(ch) - offset) if offset else ch

            if base_ch in DEV_VOWELS:
                out.append(DEV_VOWELS[base_ch])
                i += 1
                continue

            if base_ch in DEV_CONSONANTS:
                base_lat = DEV_CONSONANTS[base_ch]
                if i + 1 < n:
                    next_ch = word[i + 1]
                    next_offset = get_script_offset(next_ch)
                    if next_offset is not None:
                        next_base = chr(ord(next_ch) - next_offset) if next_offset else next_ch
                        if next_base == VIRAMA:
                            out.append(base_lat)
                            i += 2
                            continue
                        elif next_base in DEV_MATRAS:
                            out.append(base_lat + DEV_MATRAS[next_base])
                            i += 2
                            continue
                        elif next_base == ANUSVARA:
                            out.append(base_lat + 'an')
                            i += 2
                            continue

                # Implicit 'a' if next is a consonant or vowel
                if i + 1 < n:
                    next_ch = word[i + 1]
                    next_off = get_script_offset(next_ch)
                    if next_off is not None:
                        n_b = chr(ord(next_ch) - next_off) if next_off else next_ch
                        if n_b in DEV_CONSONANTS or n_b in DEV_VOWELS:
                            out.append(base_lat + 'a')
                            i += 1
                            continue
                out.append(base_lat)
                i += 1
                continue

            if base_ch in DEV_MATRAS:
                out.append(DEV_MATRAS[base_ch])
            elif base_ch == ANUSVARA:
                out.append('n')
            elif base_ch == VISARGA:
                out.append('h')
            elif base_ch == VIRAMA:
                pass  # Halant suppresses implicit vowel
        elif ch.isascii():
            out.append(ch)

        i += 1

    return ''.join(out)


def transliterate_text(text: str) -> str:
    """
    Translates/transliterates Indic scripts (Hindi, Tamil, Telugu, Kannada, etc.)
    into normalized Latin text.
    1. Replaces business entity types with canonical English ('pvt ltd', 'llp', 'investments', etc.)
    2. Replaces regional state/city names with English ('tamil nadu', 'maharashtra', etc.)
    3. Phonetically romanizes remaining words.
    """
    if not text or not contains_indic(text):
        return text

    result = text

    # Step 1: Phrase-level business entity replacements (longest first)
    for phrase, canonical in SORTED_PHRASES:
        if phrase in result:
            result = result.replace(phrase, ' ' + canonical + ' ')

    # Step 2: Regional state replacements
    for state_term, en_state in SORTED_STATES:
        if state_term in result:
            result = result.replace(state_term, ' ' + en_state + ' ')

    # Step 3: Word-level phonetic transliteration for remaining Indic words
    words = result.split()
    processed_words = []
    for w in words:
        if contains_indic(w):
            processed_words.append(transliterate_indic_word(w))
        else:
            processed_words.append(w)

    return ' '.join(processed_words)


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    test_cases = [
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "आदित्य प्रॉपर्टीज एलएलपी",
        "PLOT NO B-78/1, THANE, महाराष्ट्र",
        "Orelee's Barbershop",
        "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
        "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு",
        "బాలాజీ ఇన్వెస్ట్‌మెంట్ ప్రైవేట్ లిమిటెడ్",
        "గురు ಸೊಲ್ಯೂಷನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
        "রেড গ্লোবাল লিমিটেড",
        "ગોલ્ડ મીડિયા પ્રાઇવેટ લિમિટેડ",
        "ജയ് ട്രേഡിംഗ് പ്രൈവറ്റ് ലിമിറ്റഡ്",
        "ਵਿਜ਼ਨ ਇੰਡਸਟ੍ਰੀਜ਼ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ"
    ]
    for tc in test_cases:
        print(f"RAW : {tc}")
        print(f"TRAN: {transliterate_text(tc)}")
        print()
