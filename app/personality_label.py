def map_personality_label(scores: dict):
    O, C, E, A, N = scores.values()

    if A > 3.5 and E > 3:
        return "Romantis — hangat dan mudah bergaul"
    if C > 3.5:
        return "Perfeksionis — terstruktur dan disiplin"
    if N > 3.5:
        return "Sensitif — emosional dan reflektif"
    if O > 3.5:
        return "Kreatif — imajinatif dan terbuka"
    if E > 3.5:
        return "Ekstrovert — aktif dan komunikatif"

    return "Seimbang — fleksibel dalam berbagai situasi"
