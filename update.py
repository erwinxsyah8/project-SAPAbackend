# =========================
# EMOSI NEGATIF / KESAL, SEDIH, CEMAS
# =========================
EMO_NEGATIVE = [
    "sedih","kesedihan","kecewa","kekecewaan","frustasi","putus asa","kesal","kemarahan",
    "marah","gelisah","cemas","kekhawatiran","menangis","stress","stresnya","terpuruk",
    "bosan","capek","kelelahan","tertekan","frustrasi","murung","gundah","kehilangan",
    "tak berdaya","jengkel","panik","khawatir berlebihan","curiga","resah","was-was",
    "overthinking","terluka","terluka hati","kebingungan","sepi","minder","terasing",
    "canggung","grogi","ragu","keraguan","malu","malunya","terasingkan","pemalu","anti-sosial",
    "bingung","gelisah hati","deg-degan","frustasi sosial","putus harapan","galau","merana",
    "hampa","pilu","sayu","sengsara","duka","luka hati","meresah","pilu hati","melankolis",
    "sendu","tersiksa","kecewanya","menyesal","menangisi","meratap","terluka",
    "frustrasi berat","kehilangan harapan","gelisah mental","emosional","putus asa berat",
    "murung mendalam","stress berat","panik mendadak","curiga berlebihan",
    # 2 kata
    "tidak bahagia","sangat sedih","kecewa berat","sangat frustasi","cemas berlebihan",
    "murung mendalam","khawatir berlebihan","gelisah mental","overthinking berlebihan",
    "putus asa","tertekan berat","kehilangan besar","sedih mendalam","gelisah parah",
    "merasa sedih","murung berat","frustasi sosial","emosional berat","stress berat",

    # 3 kata
    "putus asa berat","kehilangan harapan besar","gelisah hati mendalam","terluka hati berat",
    "merasa sangat sedih","khawatir dan cemas","murung dan tertekan","frustasi sangat berat",
    "sangat sedih dan kecewa","cemas berlebihan dan khawatir","kecewa dan putus asa"
]

# Variants imbuhan
EMO_NEGATIVE += [f"me{w}" for w in ["marah","gelisah","frustasi","cemas","jengkel","murung"]]
EMO_NEGATIVE += [f"ter{w}" for w in ["tekan","murung","sedih","cemas","curiga","galau","frustrasi"]]
EMO_NEGATIVE += [f"ke{w}" for w in ["cewa","sedih","luka","kehilangan","kekecewaan"]]

# ====== ANGER / MARAH / FRUSTRASI
ANGER_EMO = [
    "marah","kesal","jengkel","geram","murka","emosi","frustrasi","panik","mendidih","berang",
    "berbenci","bete","jengkel hati","marah-marah","termarah","memarah","geramnya","mendongkol",
    "menggeram","geram hati","amarah","emosinya","keselnya","resah","kebencian",
    "membanting","menyerang","mendam","mengeram","mencak-mencak","frustrasi berat","marah mendadak",
    # 2 kata
    "sangat marah","amarah besar","frustrasi berat","marah mendadak","geram luar biasa",
    "kesal berat","emosi negatif","jengkel berat","marah dan frustrasi",

    # 3 kata
    "marah tidak terkendali","frustrasi sangat berat","kesal luar biasa","jengkel dan marah",
    "emosi sangat tinggi","marah dan kesal"
]
ANGER_EMO += [f"me{w}" for w in ["marah","jengkel","geram","murka","benci"]]
ANGER_EMO += [f"ter{w}" for w in ["marah","jengkel","geram","murka"]]

# ====== SADNESS / SEDIH
SAD_EMO = [
    "sedih","kesedihan","kecewa","kekecewaan","putus asa","menangis","murung","terpuruk",
    "merana","galau","hampa","pilu","sayu","sengsara","duka","luka hati","meresah",
    "kehilangan","pilu hati","melankolis","sendu","tersiksa","kecewanya","menyesal",
    "menangisi","meratap","terluka","terluka hati","frustasi emosional","murung mendalam",
    "putus asa berat","kehilangan harapan",
     # 2 kata
    "sangat sedih","kecewa berat","murung mendalam","putus asa berat","gelisah hati",
    "sedih parah","merasa sedih","hati sedih","kehilangan besar","murung dan sedih",

    # 3 kata
    "merasa kehilangan besar","sedih dan frustasi","murung dan tertekan","sedih dan kecewa",
    "putus asa sangat berat","gelisah dan cemas"
]
SAD_EMO += [f"ke{w}" for w in ["cewa","sedih","luka","kehilangan","kekecewaan"]]
SAD_EMO += [f"ter{w}" for w in ["sedih","kecewa","murung","galau"]]

# ====== ANXIETY / CEMAS / TAKUT
ANXIETY_EMO = [
    "cemas","gelisah","khawatir","kekhawatiran","was-was","overthinking","bingung",
    "takut","takutnya","panik","resah","gugup","grogi","ragu","keraguan","tertekan",
    "tekanan","deg-degan","gelisah hati","khawatir berlebihan","curiga","waswas","cemasnya",
    "gelisahnya","cemas berlebihan","tergugup","tertekan","tercemas","tercuriga",
    "grogi berlebihan","kekhawatiran mendalam","gelisah mental","takut berat",
    # 2 kata
    "cemas berlebihan","khawatir berlebihan","gelisah hati","takut berat","deg-degan parah",
    "cemas parah","khawatir tinggi","gelisah berlebihan","was-was berat",

    # 3 kata
    "cemas dan gelisah","khawatir terus menerus","takut sangat berat","gelisah dan khawatir",
    "deg-degan sangat parah","cemas dan takut"
]
ANXIETY_EMO += [f"ter{w}" for w in ["tekan","gugup","cemas","curiga","khawatir"]]
ANXIETY_EMO += [f"me{w}" for w in ["cemas","gelisah","khawatir"]]

# ====== EMOSI POSITIF / SENANG / BAHAGIA
EMO_POSITIVE = [
    "senang","kesenangan","bahagia","kebahagiaan","puas","kepuasan","bangga","kebanggaan",
    "gembira","ceria","lega","ketenangan","termotivasi","motivasinya","tenang","optimis",
    "antusias","relaks","positif","semangat","senyum","puas hati","riang","bersemangat",
    "terinspirasi","syukur","damai","berenergi","senyum-senyum","bermotif positif",
    "senangnya","bahagianya","lega hati","bahagia sekali","puas banget","termotivasi","bersemangat",
    "riang gembira","senyum lebar","puas luar biasa","optimis tinggi","bahagia mendalam","antusiasme tinggi",
    # 2 kata
    "sangat senang","bahagia sekali","puas hati","riang gembira","senyum lebar",
    "senang dan puas","termotivasi tinggi","tenang dan damai","optimis tinggi","riang dan bahagia",

    # 3 kata
    "sangat bahagia sekali","riang dan gembira","puas dan senang","senang dan bersemangat",
    "bahagia dan tenang","termotivasi dan antusias"
]

# ====== SOSIAL / TRUST / RELATIONSHIP
NEGATIVE_SOCIAL = [
    "takut","takutnya","ketakutan","cemas","kecemasan","tidak percaya diri","percaya diri rendah",
    "menyendiri","sendiri","menjauhi","malu","malunya","grogi","ragu","keraguan","tertekan",
    "tekanan","khawatir","kekhawatiran","menjauh","isolasi","mengasingkan","bingung","kebingungan",
    "terasing","canggung","gelisah","was-was","sepi","minder","terasingkan","curiga","pemalu",
    "anti-sosial","resah","menghindar","overthinking","terluka","terluka hati","frustrasi sosial",
    "terasing dari kelompok","menjauh dari teman","isolasi sosial","tidak nyaman berinteraksi",
    # 2 kata
    "tidak suka","tidak senang","benci pada","tidak nyaman","menjauhi orang",
    "tidak percaya","menghindar dari","tidak mau berinteraksi","tidak ingin berinteraksi",
    "tidak peduli","tidak menghargai","mengabaikan orang",

    # 3 kata
    "tidak suka orang","tidak nyaman berinteraksi","menjauhi orang lain","tidak percaya diri",
    "tidak peduli orang","mengabaikan orang lain","tidak ingin berinteraksi"
]

POSITIVE_SOCIAL = [
    "bertemu","bertemunya","ngobrol","ngobrolin","berbagi","membantu","hangout","bersosialisasi",
    "berinteraksi","teman","teman-teman","komunikasi","diskusi","kerjasama","bergaul","acara",
    "saling","kerabat","mendekat","bersenda","tertawa bersama","berkenalan","jaringan","teamwork",
    "ramah","ceria","humoris","aktif","berpartisipasi","sosial","berteman","berkumpul",
    "kolaborasi","kerjasama tim","mendukung","memotivasi","bekerjasama","menghargai teman",
    
]
EXTRAVERSION_E = [
    "aktif",
    "diskusi",
    "ngobrol",
    "ramai",
    "keramaian",
    "sosial",
    "interaksi",
    "bergaul",
    "berkumpul",
    "kerja tim",
    "kolaboratif",
    "presentasi",
    "depan umum",
    "kenal orang baru",
    "akrab",
    "nimbrung",
    "komunikasi",
    "berbincang",
    "bercerita",
    "berbagi cerita",
    "suasana hidup",
    "pusat perhatian",
    "banyak orang",
    "aktif berdiskusi","aktif berpartisipasi","aktif dalam diskusi","aktif diskusi kelompok",
    "aktif dalam kegiatan bersama","aktif dalam kegiatan kelompok","aktif dalam kegiatan sosial",
    "aktif bersosialisasi","aktif dalam acara sosial","aktif dalam lingkungan sosial",
    "aktif dalam percakapan kelompok","aktif ikut diskusi","aktif menyapa","aktif menyapa orang",
    "aktif saat berkumpul","aktif saat kerja tim","aktif saat ada acara",
    "energi meningkat saat berkumpul","energi naik saat bersama orang lain",
    "lebih hidup kalau ada teman","lebih hidup saat bersama orang lain",
    "lebih aktif kalau ada teman","lebih aktif saat ada teman",

    "mudah bergaul","mudah berinteraksi","mudah membangun komunikasi",
    "mudah menjalin komunikasi","mudah nyambung kalau diajak ngobrol",
    "mudah beradaptasi secara sosial","mudah bergaul di lingkungan baru",

    "senang berinteraksi","senang berinteraksi langsung","senang berinteraksi dengan orang baru",
    "senang berinteraksi dengan banyak orang","senang ngobrol","senang ngobrol lama",
    "senang ngobrol panjang lebar","senang ngobrol rame","senang ngobrol santai",
    "senang berbagi cerita","senang berbagi cerita ke orang lain",
    "senang berbagi pengalaman","senang bertemu banyak orang",
    "senang ikut kegiatan sosial","senang terlibat aktivitas sosial",
    "senang terlibat diskusi","senang diskusi santai",

    "suka ngobrol","suka ngobrol dengan banyak orang","suka ngobrol santai",
    "suka ngobrol sambil bercanda","suka ngobrol di berbagai situasi",
    "suka berbagi cerita","suka berbagi pengalaman",
    "suka aktivitas sosial","suka bergaul","suka interaksi langsung",
    "suka ikut acara kumpul","suka ikut acara rame",

    "lebih nyaman kerja tim","lebih nyaman kerja bareng tim",
    "lebih suka kerja tim","lebih suka diskusi langsung",
    "lebih suka kerja sambil ngobrol","lebih senang ngobrol daripada diam",

    "paling semangat kalau diskusi","paling semangat kalau ngobrol rame",
    "ngerasa lebih hidup saat ngobrol","ngerasa semangat kalau ngobrol lama",
    "kalau ada acara kumpul pasti ikut","kalau ada acara rame pasti ikut"
]
TRUST = [
    "peduli","menolong","percaya","percaya diri","loyal","setia","mendukung","mempercayai",
    "terbuka","saling percaya","menghargai","mengandalkan","solid","ramah","toleran",
    "mengerti","memaafkan","kooperatif","sopan","baik hati","humane","bekerjasama",
    "percaya penuh","percaya satu sama lain","percaya tim",
    "integritas","bertanggung jawab","mengayomi","memimpin","membimbing"
]

RELATIONSHIP_AFFECTION = [
    "sayang","cinta","kasih","peduli","rindu","kamu","kita","hubungan","bersama","pasangan",
    "pacar","kekasih","teman dekat","teman sejati","kekasih hati","hubungan romantis",
    "affeksi","kehangatan","kedekatan","kasih sayang","memelihara hubungan","intim"
]
# ====== IDE TERBUKA / DISKUSI MENDALAM / KREATIF SOSIAL (AGREEABLENESS DOMINANT)
CREATIVE_DISCUSSION_A = [
    "ide segar","pikiran terbuka","berpikir terbuka","diskusi ide","diskusi mendalam",
    "diskusi yang mendalam","diskusi lintas bidang","berbagi ide","bertukar ide",
    "kolaborasi ide","pemikiran terbuka","open minded","kreatif","kreativitas",
    "inovasi","solusi kreatif","pemikiran inovatif","ide kreatif",
    "menyukai diskusi","menikmati diskusi","diskusi intelektual",
    "mencari solusi bersama","pemecahan masalah bersama",
    "ide membantu","kreativitas membantu","berpikir terbuka membantu",
    "inovasi solusi","pemikiran lintas disiplin"
]
DISCIPLINE_C = [
    "disiplin",
  "tepat waktu",
  "tanggung jawab",
  "rapi",
  "teratur",
  "terstruktur",
  "sistematis",
  "konsisten",
  "teliti",
  "fokus",
  "kualitas",
  "deadline",
  "perencanaan",
  "rencana",
  "to-do list",
  "efisien",
  "produktif",
  "kerapihan",
  "komitmen",
  "target",
  "hasil maksimal",
  "tidak suka menunda",
  "tidak suka asal-asalan",
  "pekerjaan selesai",
  "sesuai rencana",
    "disiplin","bertanggung jawab","tanggung jawab","fokus","target","deadline",
    "tepat waktu","ketepatan waktu","konsisten","konsistensi","rapi","teratur",
    "terstruktur","terorganisir","sistematis","perencanaan","rencana","jadwal",
    "to-do list","daftar tugas","prioritas","efisien","produktif","teliti",
    "ketelitian","kualitas","hasil maksimal","hasil optimal","komitmen",
    "anti menunda","tidak suka menunda","tidak suka menunda pekerjaan",
    "menyelesaikan","menyelesaikan tugas","menyelesaikan pekerjaan",
    "selesai tepat waktu","sesuai rencana","standar kerja","kerja serius",
    "pekerjaan rapi","kerja rapi","kerja terencana","kerja sistematis",
    "kerja terstruktur","kerja terorganisir","fokus bekerja",
    "cek ulang","mengecek ulang","detail","setiap detail",
    "pekerjaan tuntas","sampai selesai","tidak setengah-setengah",
    "tidak asal-asalan","tidak terburu-buru","kualitas lebih penting",
    "menjaga kualitas","menjaga konsistensi","menjaga komitmen"
]
# ====== MENTAL UNSTABLE / OVERTHINKING / ANXIETY CORE (NEUROTICISM)
MENTAL_UNSTABLE_N = [
    "gugup","mudah gugup","mudah merasa gugup",
    "tidak tenang","merasa tidak tenang","sering tidak tenang",
    "perasaan tidak stabil","emosi tidak stabil",
    "pikiran kacau","pikiran mudah kacau","pikiran sering kacau",
    "pikiran negatif","pikiran negatif sering muncul",
    "pikiran negatif muncul berulang kali",
    "pikiran negatif muncul terus-menerus",
    "pikiran negatif sulit dikendalikan",
    "pikiran sering negatif","pikiran sering mengganggu",
    "pikiran tidak terkendali","pikiran terasa berat",
    "pikiran tidak pernah tenang",
    "stres","mudah stres","mudah merasa stres",
    "tidak aman","merasa tidak aman","sering merasa tidak aman",
    "cemas","kecemasan","gelisah","resah",
    "sulit rileks","sulit merasa rileks",
    "sulit tenang","sulit merasa tenang",
    "sulit merasa damai","sulit merasa nyaman",
    "sulit menenangkan diri","sulit menenangkan pikiran",
    "sulit merasa aman","sulit tenang sepenuhnya",
    "tidak nyaman tanpa sebab","gelisah tanpa sebab"
]

E_SOCIAL_DEPENDENCY = [
  "lebih suka kerja bareng",
  "lebih suka kerja kelompok",
  "diskusi ramai",
  "kurang semangat kalau sepi",
  "gelisah kalau sendirian",
  "lebih aktif kalau ada teman",
  "lebih nyaman kerja bareng"
]
EMPATHY_HARMONY_A = [
    "empati","berempati","simpati","iba","peduli",
    "tidak suka melihat orang lain sedih",
    "tidak suka membuat orang lain kecewa",
    "tidak suka menyakiti orang lain",
    "tidak suka menyakiti perasaan orang",
    "tidak suka menyalahkan orang",
    "tidak tega melihat orang kesulitan",

    "damai","berdamai","suasana damai","harmonis","keharmonisan",
    "menghindari konflik","menghindari pertengkaran","menghindari perselisihan",
    "tidak suka konflik","tidak suka pertengkaran",
    "lebih memilih berdamai","lebih memilih kompromi",
    "lebih memilih mengalah","mengalah daripada konflik",
    "mengalah daripada bertengkar",

    "kerja sama","bekerja sama","kolaboratif","kebersamaan",
    "mudah bekerja sama","senang bekerja sama",
    "lebih memilih kerja sama daripada bersaing",

    "menjaga hubungan baik","menjaga perasaan orang",
    "menjadi pendengar yang baik","memahami perasaan orang",
    "memahami sudut pandang orang lain",
    "mudah tersentuh","mudah memahami perasaan",
    "selalu berusaha bersikap baik",
    "selalu berusaha bersikap adil",
    "senang membantu tanpa mengharapkan balasan",

    "suasana nyaman","suasana rukun","suasana harmonis",
    "menciptakan suasana positif","menjaga suasana nyaman",
    "tidak enak hati","tidak enak menolak bantuan"
]
# ====== INTROSPECTION / ANALYTICAL / OCEAN
INTROSPECTION = [
    "merenung","berpikir","refleksi","evaluasi","mengamati","menganalisis","mengingat","menyadari",
    "kontemplasi","renungan","introspeksi","memikirkan","mencermati","merenungkan","berandai-andai",
    "filosofis","menghayati","meneliti","mendalami","menafsirkan","merenungi","observasi","perenungan",
    "pemikiran mendalam","kritis","analisis","evaluasi","menganalisis","mempertimbangkan","memeriksa","menguji",
    "analitis","rasional","logis","problem solving","menganalisis data",
    "observasi mendalam","pemikiran kritis","refleksi mendalam","analisis terperinci"
]

# ====== ACHIEVEMENT / DISCIPLINE / PRODUCTIVITY
ACHIEVEMENT = [
    "disiplin","tekun","bertanggung jawab","menyelesaikan","goal","target","berusaha","gigih",
    "produktif","rajin","fokus","komitmen","dedikasi","berprestasi","inisiatif","teliti","rapi",
    "terorganisir","mengikuti aturan","persisten","berorientasi hasil","bertekad","kemauan keras",
    "capai target","usaha maksimal","hasil maksimal","hasil optimal","pencapaian","goal oriented",
    "proaktif","inisiatif","mandiri","berinisiatif","work hard","determinasi",
    "menyelesaikan tugas tepat waktu","produktif tinggi","komitmen penuh","mencapai milestone","berorientasi prestasi"
]
COLLABORATION = [
    "bekerja sama","teamwork","kolaborasi","bersama","kooperatif","mendukung tim",
    "gotong royong","tim","kerja tim","kerjasama","saling membantu"
]
# =========================
# EXTREME NEGATIVE / SELF-HARM / SUICIDAL
# =========================
EXTREME_NEGATIVE = [
    "ingin mati", "rasanya ingin menyerah", "mati saja", "tidak ingin hidup", 
    "putus asa ingin mati", "bunuh diri", "tidak tahan hidup", "ingin bunuh diri", 
    "akhiri hidupku", "sudah tidak kuat", "tidak sanggup lagi", "lepaskan nyawa"
]

# Variants imbuhan sederhana
EXTREME_NEGATIVE += [f"ter{w}" for w in ["tekan","putus asa","stres","sakit"]]
EXTREME_NEGATIVE += [f"me{w}" for w in ["mati","bunuh","menyerah"]]



# ==========================
# OCEAN ADJUSTMENT REFINED
# ==========================
from collections import Counter
import re

def adjust_ocean_by_keywords(scores: dict, text: str):
    adjusted = scores.copy()
    counter = Counter(re.findall(r'\w+', text.lower()))

    # NEGATIVE_SOCIAL → menurunkan E, menaikkan N, sedikit turunkan A
    for word in NEGATIVE_SOCIAL:
        if word in counter:
            f = counter[word]
            adjusted["E"] -= 0.2 * f
            adjusted["N"] += 0.5 * f
            adjusted["A"] -= 0.1 * f

    # POSITIVE_SOCIAL → menaikkan E & A, sedikit menurunkan N jika sangat negatif
    for word in POSITIVE_SOCIAL:
        if word in counter:
            f = counter[word]
            adjusted["A"] += 0.4 * f   # fokus ke agreeableness
            adjusted["E"] += 0.3 * f
            adjusted["N"] -= 0.05 * f


    # EMO_POSITIVE → tingkatkan E & O
    for word in EMO_POSITIVE:
        if word in counter:
            f = counter[word]
            adjusted["E"] += 0.3 * f
            adjusted["O"] += 0.15 * f

    # EMO_NEGATIVE → tingkatkan N & sedikit O
    for word in EMO_NEGATIVE:
        if word in counter:
            f = counter[word]
            adjusted["N"] += 0.35 * f
            adjusted["O"] += 0.1 * f

    # INTROSPECTION → tingkatkan O
    for word in INTROSPECTION:
        if word in counter:
            if not any(n in text.lower() for n in MENTAL_UNSTABLE_N):
                f = counter[word]
            adjusted["O"] += 0.35 * f

    # ACHIEVEMENT → tingkatkan C
    for word in ACHIEVEMENT:
        if word in counter:
            f = counter[word]
            adjusted["C"] += 0.5 * f

    # TRUST → tingkatkan A
    for word in TRUST:
        if word in counter:
            f = counter[word]
            adjusted["A"] += 0.5 * f

    # RELATIONSHIP_AFFECTION → naikkan A, sedikit turunkan N
    for word in RELATIONSHIP_AFFECTION:
        if word in counter:
            f = counter[word]
            adjusted["A"] += 0.6 * f
            adjusted["N"] -= 0.1 * f

    for word in COLLABORATION:
        if word in counter:
            f = counter[word]
            adjusted["A"] += 0.9 * f   # fokus ke agreeableness
            adjusted["E"] += 0.7 * f
            adjusted["N"] -= 0.05 * f
     # EXTREME_NEGATIVE → N naik lebih tinggi, E turun sedikit
    for phrase in EXTREME_NEGATIVE:
        if phrase in text.lower():  # periksa frasa utuh
            adjusted["N"] += 1.0   # tingkatkan Neuroticism signifikan
            adjusted["E"] -= 0.2   # ekstrim → lebih introvert
            # Bisa juga set alert
            adjusted["EXTREME_ALERT"] = True
    # CREATIVE DISCUSSION (A-dominant)
    for phrase in CREATIVE_DISCUSSION_A:
        if phrase in text.lower():
            adjusted["A"] += 0.8
            adjusted["O"] += 0.4
            adjusted["E"] += 0.2
    for word in DISCIPLINE_C:
        if word in counter:
            f = counter[word]
            adjusted["C"] += 0.8 * f
    for word in EXTRAVERSION_E:
        if word in counter:
            f = counter[word]
            adjusted["E"] += 0.7 * f
            adjusted["A"] += 0.3 * f
            adjusted["N"] -= 0.05 * f
    for word in E_SOCIAL_DEPENDENCY:
        if word in counter:
            f = counter[word]
            adjusted["E"] += 0.6 * f
            adjusted["A"] += 0.3 * f
            adjusted["N"] -= 0.1 * f
    for word in EMPATHY_HARMONY_A:
        if word in counter:
            f = counter[word]
            adjusted["A"] += 0.7 * f
            adjusted["N"] -= 0.1 * f
    # MENTAL UNSTABLE / OVERTHINKING
    for phrase in MENTAL_UNSTABLE_N:
        if phrase in text.lower():
            adjusted["N"] += 8.0
            adjusted["E"] -= 0.2
            adjusted["O"] -= 0.1


    # Clamp ke skala 1–5 secara lebih smooth
    for k in adjusted:
        adjusted[k] = round(min(5.0, max(1.0, adjusted[k])), 3)

    return max(adjusted, key=adjusted.get), adjusted

# ==========================
# KEYWORD → OCEAN MAPPING
# ==========================
KEYWORD_TRAIT_MAP = {
    "NEGATIVE_SOCIAL": {"E": -0.2, "N": 0.5, "A": -0.1},
    "POSITIVE_SOCIAL": {"A": 0.3, "E": 0.6, "N": -0.05},
    "EMO_POSITIVE": {"E": 0.3, "O": 0.15},
    "EMO_NEGATIVE": {"N": 0.35, "O": 0.1},
    "INTROSPECTION": {"O": 0.35},
    "ACHIEVEMENT": {"C": 0.5},
    "CREATIVE_DISCUSSION_A": {"A": 0.8, "E": 0.4, "C": 0.2, "N": -0.05},
    "TRUST": {"A": 0.5},
    "RELATIONSHIP_AFFECTION": {"A": 0.6, "N": -0.1},
    "COLLABORATION": {"A": 0.8, "E": 0.5, "C": 0.2, "N": -0.05},
    "ANGER_EMO": {"N": 0.5},
    "SAD_EMO": {"N": 0.3, "O": 0.05},
    "ANXIETY_EMO": {"N": 0.35, "E": -0.1},
    "EXTREME_NEGATIVE": {"N": 1.0, "E": -0.2},
    "DISCIPLINE_C": {"C": 0.8},
    "EXTRAVERSION_E": {"E": 0.7, "A": 0.5, "N": -0.05},
    "E_SOCIAL_DEPENDENCY": {"E": 0.6, "A": 0.4, "N": -0.1},
    "EMPATHY_HARMONY_A": {"A": 0.7, "N": -0.1},
    "MENTAL_UNSTABLE_N": {"N": 8.0, "E": -0.2, "O": -0.1},

}

# ==========================
# EMOTIONAL KEYWORD ADJUSTMENT REFINED
# ==========================
def apply_emotional_keyword_adjustment(text: str, scores: dict):
    adjusted = scores.copy()
    counter = Counter(re.findall(r'\w+', text.lower()))

    # ANGER → kuatkan N, tidak ubah O
    for word in ANGER_EMO:
        if word in counter:
            f = counter[word]
            adjusted["N"] += 0.5 * f

    # SADNESS → naikkan N & sedikit O
    for word in SAD_EMO:
        if word in counter:
            f = counter[word]
            adjusted["N"] += 0.3 * f
            adjusted["O"] += 0.05 * f

    # ANXIETY → naikkan N, turunkan E sedikit
    for word in ANXIETY_EMO:
        if word in counter:
            f = counter[word]
            adjusted["N"] += 0.35 * f
            adjusted["E"] -= 0.1 * f

     # Social, Achievement, Trust, Relationship
    for group_name, keywords in {
        "NEGATIVE_SOCIAL": NEGATIVE_SOCIAL,
        "POSITIVE_SOCIAL": POSITIVE_SOCIAL,
        "EMO_POSITIVE": EMO_POSITIVE,
        "EMO_NEGATIVE": EMO_NEGATIVE,
        "INTROSPECTION": INTROSPECTION,
        "ACHIEVEMENT": ACHIEVEMENT,
        "TRUST": TRUST,
        "RELATIONSHIP_AFFECTION": RELATIONSHIP_AFFECTION,
        "COLLABORATION": COLLABORATION,
        "CREATIVE_DISCUSSION_A": CREATIVE_DISCUSSION_A,
        "DISCIPLINE_C": DISCIPLINE_C,
        "EXTRAVERSION_E": EXTRAVERSION_E,
        "E_SOCIAL_DEPENDENCY": E_SOCIAL_DEPENDENCY,
        "EMPATHY_HARMONY_A": EMPATHY_HARMONY_A,
        "MENTAL_UNSTABLE_N": MENTAL_UNSTABLE_N,
    }.items():
        for word in keywords:
            if word in counter:
                f = math.log(1 + counter[word])
                for trait, weight in KEYWORD_TRAIT_MAP.get(group_name, {}).items():
                    adjusted[trait] += weight * f

    # Clamp ke skala 1–5
    for k in ["O","C","E","A","N"]:  # jangan clamp EXTREME_ALERT
        adjusted[k] = round(min(5.0, max(1.0, adjusted[k])), 3)

    return adjusted

def determine_dominant_trait(scores, text):
    # Hitung E/A/N untuk konteks sosial
    social_hits = sum(1 for w in POSITIVE_SOCIAL+COLLABORATION if w in text.lower())
    emo_hits = sum(1 for w in EMO_POSITIVE if w in text.lower())

    # Jika banyak kata kolaborasi → dominan A
    if social_hits >= 1:
        return "A"
    if emo_hits >= 1:
        return "E"
    return max(scores, key=scores.get)

# ==========================
# HIGHLIGHT
# ==========================
def highlight_keywords_in_text(text: str, evidence: dict):
    tokens = re.findall(r'\w+|\W+', text)
    highlights = set()

    for items in evidence.values():
        for e in items:
            highlights.update([t.lower() for t in e["matched_tokens"]])

    result = ""
    for t in tokens:
        result += f"<mark>{t}</mark>" if t.lower() in highlights else t
    return result

# ==========================
# SUPER EXPLANATION (UPDATE EXTREME ALERT)
# ==========================
def extract_keywords(text, top_n=5):
    return [w for w,_ in Counter(re.findall(r'\w+', text.lower())).most_common(top_n)]

def generate_explanation_suggestion_super(text, adjusted, evidence):
    dominant = max(adjusted, key=adjusted.get)
    words = extract_keywords(text)
    snippet = ", ".join(words[:3])

    # Peringatan jika kalimat ekstrem
    if adjusted.get("EXTREME_ALERT"):
        explanation = (
            f"⚠ Kalimat ini mengandung indikasi emosional ekstrem / suicidal. "
            f"Kecenderungan trait {dominant} tetap terlihat, tetapi terdapat risiko tinggi. "
            f"Kata-kata seperti {snippet} menunjukkan hal tersebut."
        )
        suggestion = (
            f"Sangat disarankan untuk segera memperhatikan kondisi ini dan memberikan dukungan psikologis. "
            f"Mengamati kata-kata seperti {snippet} dapat membantu mencegah risiko lebih lanjut."
        )
    else:
        explanation = (
            f"Kalimat ini menunjukkan kecenderungan {dominant} karena kata-kata seperti {snippet} menandai pola tersebut."
        )
        suggestion = (
            f"Mengamati dan menindaklanjuti hal seperti {snippet} dapat membantu mengoptimalkan trait {dominant}."
        )

    return explanation, suggestion

def determine_dominant_contextual(adjusted, evidence):
    sorted_traits = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)
    top_trait, top_score = sorted_traits[0]

    # Hitung jumlah bukti sosial / positif
    social_hits = len(evidence.get("POSITIVE_SOCIAL", []))
    emo_hits = len(evidence.get("EMO_POSITIVE", []))

    if social_hits >= 2:
        return "E"
    if emo_hits >= 2:
        return "A"

    return top_trait

PERSONA_RULES = [
    # ================= EMOSI NEGATIF =================
    (
        "Cemas & Pikiran Tidak Tenang",
        lambda s: s["N"] >= 3.6,
        "mudah gugup, sulit merasa tenang, dan sering diliputi pikiran negatif"
    ),
    (
        "Overthinking Emosional",
        lambda s: s["N"] >= 3.5 and s["O"] <= 3.2,
        "pikiran sulit dikendalikan, sering cemas, dan kurang stabil secara emosi"
    ),
    (
        "Rentan Stres",
        lambda s: s["N"] >= 3.4 and s["C"] <= 3.0,
        "mudah tertekan oleh situasi dan membutuhkan manajemen stres yang baik"
    ),
    (
        "Empatik & Penjaga Harmoni",
        lambda s: s["A"] >= 3.7,
        "peduli, mudah memahami perasaan orang lain, dan mengutamakan keharmonisan"
    ),
    (
        "Pendamai Alami",
        lambda s: s["A"] >= 3.6 and s["N"] <= 3.2,
        "menghindari konflik, memilih kompromi, dan menciptakan suasana damai"
    ),
    (
        "Kolaborator Hangat",
        lambda s: s["A"] >= 3.5 and s["E"] >= 3.0,
        "mudah bekerja sama, suportif, dan menjaga hubungan interpersonal"
    ),
    (
        "Pendengar yang Baik",
        lambda s: s["A"] >= 3.4 and s["O"] <= 3.2,
        "lebih fokus pada perasaan manusia daripada perdebatan ide"
    ),
    (
        "Sosial Aktif",
        lambda s: s["E"] >= 3.6,
        "energik, aktif berinteraksi, dan merasa hidup saat bersama orang lain"
    ),
    (
        "Penggerak Diskusi",
        lambda s: s["E"] >= 3.5 and s["O"] >= 3.0,
        "suka berdiskusi, memancing ide, dan menjaga dinamika percakapan"
    ),
    (
        "Team Energizer",
        lambda s: s["E"] >= 3.4 and s["C"] >= 3.0,
        "menghidupkan suasana tim dan mendorong kolaborasi aktif"
    ),
    (
        "Ekstrovert Sosial",
        lambda s: s["E"] >= 3.8 and s["N"] <= 3.0,
        "percaya diri, mudah bergaul, dan nyaman di lingkungan sosial"
    ),
    (
        "Disiplin & Bertanggung Jawab",
        lambda s: s["C"] >= 3.8,
        "terstruktur, konsisten, fokus pada kualitas, dan dapat diandalkan"
    ),
    (
        "Perfeksionis Terstruktur",
        lambda s: s["C"] >= 3.6 and s["N"] <= 3.2,
        "menjaga standar tinggi, rapi, dan tidak mentoleransi pekerjaan asal-asalan"
    ),
    (
        "Manajer Tugas Andal",
        lambda s: s["C"] >= 3.5 and s["E"] >= 3.0,
        "mampu mengatur pekerjaan, waktu, dan tanggung jawab secara efektif"
    ),
    (
        "Pekerja Sistematis",
        lambda s: s["C"] >= 3.4 and s["O"] <= 3.2,
        "lebih nyaman dengan rencana jelas, alur kerja pasti, dan target terukur"
    ),
    (
        "Sensitif Emosional",
        lambda s: s["N"] >= 3.6 and s["N"] >= s["O"] + 0.2,
        "emosional, peka, dan mudah terpengaruh suasana"
    ),
    (
        "Kolaborator Intelektual",
        lambda s: s["A"] >= 3.6 and s["O"] >= 3.4,
        "terbuka terhadap ide, menyukai diskusi mendalam, dan nyaman berkolaborasi lintas perspektif"
    ),
    (
        "Pemikir Terbuka & Solutif",
        lambda s: s["A"] >= 3.5 and s["O"] >= 3.5 and s["C"] >= 3.0,
        "menggabungkan empati, kreativitas, dan logika untuk menemukan solusi bersama"
    ),
    (
        "Idealis Kolaboratif",
        lambda s: s["A"] >= 3.7 and s["O"] >= 3.6 and s["N"] <= 3.2,
        "berorientasi nilai, menyukai dialog intelektual, dan membangun inovasi secara kolektif"
    ),

    (
        "Tempramental",
        lambda s: s["N"] >= 4.0,
        "cepat marah, impulsif, dan reaktif terhadap frustrasi"
    ),
    (
        "Cemas & Overthinking",
        lambda s: s["N"] >= 3.5 and s["E"] <= 3.0,
        "mudah khawatir, berpikir berlebihan, dan gelisah"
    ),
    (
        "Sedih / Melankolis",
        lambda s: s["N"] >= 3.2 and s["O"] >= 3.0 and s["E"] <= 3.2,
        "sering merenung, mudah merasa kehilangan, dan introspektif"
    ),

    # ================= EMOSI POSITIF =================
    (
        "Romantis",
        lambda s: s["A"] >= 3.4 and s["A"] >= s["O"] + 0.2,
        "hangat, penuh afeksi, dan berorientasi hubungan"
    ),
    (
        "Ramah Sosial",
        lambda s: s["E"] >= 3.5 and s["A"] >= 3.2,
        "ceria, mudah bergaul, dan menyukai interaksi sosial"
    ),
    (
        "Empatik",
        lambda s: s["A"] >= 3.5 and s["N"] <= 3.2,
        "peduli, memahami perasaan orang lain, dan suportif"
    ),
    (
        "Kritik & Kritis",
        lambda s: s["O"] >= 3.7 and s["C"] >= 3.2,
        "analitis, kritis, dan memperhatikan detail"
    ),
    (
        "Visioner Kreatif",
        lambda s: s["O"] >= 3.7 and s["O"] >= s["A"] + 0.2,
        "imajinatif, reflektif, dan terbuka terhadap ide baru"
    ),
    (
        "Inovator",
        lambda s: s["O"] >= 3.5 and s["C"] >= 3.0 and s["E"] >= 3.0,
        "selalu mencari cara baru, kreatif, dan berpikir out-of-the-box"
    ),

    # ================= PENCAPAIAN & DISCIPLIN =================
    (
        "Perfeksionis",
        lambda s: s["C"] >= 3.6 and s["C"] >= s["N"] + 0.2,
        "terstruktur, disiplin, dan berorientasi pencapaian"
    ),
    (
        "Ambisius",
        lambda s: s["C"] >= 3.5 and s["O"] >= 3.5,
        "berorientasi tujuan, proaktif, dan berinisiatif"
    ),
    (
        "Gigih & Persisten",
        lambda s: s["C"] >= 3.4 and s["N"] <= 3.2,
        "konsisten, tidak mudah menyerah, dan berdedikasi"
    ),
    (
        "Pragmatis",
        lambda s: s["C"] >= 3.2 and s["E"] >= 3.2,
        "praktis, realistis, dan fokus pada hasil"
    ),

    # ================= KOLABORATOR =================
    (
        "Kolaborator",
        lambda s: s["A"] >= 3.5 and s["E"] >= 3.2 and s["C"] >= 3.0,
        "mampu bekerja sama, mendukung tim, dan membangun harmoni"
    ),
    (
        "Mediator",
        lambda s: s["A"] >= 3.3 and s["N"] <= 3.2 and s["E"] >= 3.0,
        "menjembatani konflik, tenang, dan diplomatis"
    ),
    (
        "Pemimpin Visioner",
        lambda s: s["O"] >= 3.6 and s["C"] >= 3.5 and s["E"] >= 3.2,
        "mengambil inisiatif, memimpin tim, dan strategis"
    ),

    # ================= KEPRIBADIAN SEIMBANG =================
    (
        "Seimbang",
        lambda s: 2.8 <= s["O"] <= 3.5 and 2.8 <= s["C"] <= 3.5 and 2.8 <= s["E"] <= 3.5 and 2.8 <= s["A"] <= 3.5 and 2.8 <= s["N"] <= 3.5,
        "adaptif, fleksibel, dan tidak ekstrem pada satu trait"
    )
]
def generate_global_conclusion(avg, dominant):
    O, C, E, A, N = avg["O"], avg["C"], avg["E"], avg["A"], avg["N"]

    # ================= KESIMPULAN =================
    conclusion = (
        f"Secara keseluruhan, hasil analisis menunjukkan bahwa trait kepribadian "
        f"yang paling dominan adalah {dominant}. Individu ini cenderung "
    )

    if dominant == "O":
        conclusion += "memiliki tingkat keterbukaan tinggi terhadap ide baru, reflektif, dan kreatif."
    elif dominant == "C":
        conclusion += "terstruktur, disiplin, konsisten, dan bertanggung jawab."
    elif dominant == "E":
        conclusion += "aktif secara sosial, komunikatif, dan energik."
    elif dominant == "A":
        conclusion += "kooperatif, empatik, dan menjaga keharmonisan sosial."
    elif dominant == "N":
        conclusion += "sensitif terhadap tekanan emosional dan mudah mengalami stres."

    # Insight tambahan dari Neuroticism
    if N < 0.35:
        conclusion += " Tingkat kestabilan emosi tergolong baik."
    elif N > 0.6:
        conclusion += " Namun terdapat kecenderungan emosi negatif yang cukup tinggi."

    # ================= SARAN =================
    suggestion = "Disarankan untuk "

    if dominant == "C":
        suggestion += (
            "memanfaatkan kemampuan perencanaan dan kedisiplinan dalam pekerjaan atau studi, "
            "namun tetap melatih fleksibilitas agar tidak terlalu kaku."
        )
    elif dominant == "O":
        suggestion += (
            "menyalurkan kreativitas ke aktivitas produktif seperti riset, inovasi, dan eksplorasi ide baru."
        )
    elif dominant == "E":
        suggestion += (
            "mengoptimalkan kemampuan komunikasi dan kepemimpinan dalam kerja tim, "
            "serta melatih kemampuan refleksi diri."
        )
    elif dominant == "A":
        suggestion += (
            "mempertahankan sikap empati sambil belajar bersikap lebih tegas dalam pengambilan keputusan."
        )
    elif dominant == "N":
        suggestion += (
            "melatih regulasi emosi melalui manajemen stres, mindfulness, atau journaling secara rutin."
        )

    return conclusion, suggestion