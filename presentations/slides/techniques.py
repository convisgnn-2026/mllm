from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree
rgb = lambda h: RGBColor.from_string(h)
ORANGE, BLUE, GREEN, GRAY, INK, INK2, NAVY, RED, LIGHT = "F4B183", "DDEBF7", "E2F0D9", "E7E6E6", "1F1F1F", "3C4A5A", "1E4D7B", "C00000", "F2F4F7"
COMP, MEM = "F8CBAD", "BDD7EE"   # 연산 바운드 / 메모리 바운드
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

def box(s, x, y, w, h, fill=None, line="7F7F7F", shape=MSO_SHAPE.RECTANGLE, text=None, size=12, bold=False, color=INK, align=PP_ALIGN.CENTER, lw=1.0, name="shape", dash=False):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h)); sh.name = name
    if fill: sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    else: sh.fill.background()
    if line: sh.line.color.rgb = rgb(line); sh.line.width = Pt(lw)
    else: sh.line.fill.background()
    if dash:
        from pptx.enum.dml import MSO_LINE
        sh.line.dash_style = MSO_LINE.DASH
    sh.shadow.inherit = False
    if text is not None:
        tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, Inches(0.04))
        for i, t in enumerate(text if isinstance(text, list) else [text]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
            r = p.add_run(); r.text = t; r.font.size = Pt(size); r.font.bold = bold if i == 0 else False; r.font.color.rgb = rgb(color); r.font.name = "Calibri"
    return sh

def text(s, x, y, w, h, paras, size=13, color=INK, bold=False, italic=False, bullets=False, space=6, anchor=MSO_ANCHOR.TOP, align=None, name="text"):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tb.name = name
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, 0)
    for i, p_ in enumerate(paras if isinstance(paras, list) else [paras]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(space)
        if align: p.alignment = align
        for j, (seg, o) in enumerate(p_ if isinstance(p_, list) else [(p_, {})]):
            r = p.add_run(); r.text = ("•  " if bullets and j == 0 else "") + seg
            r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", bold); r.font.italic = o.get("italic", italic); r.font.color.rgb = rgb(o.get("color", color)); r.font.name = "Calibri"
    return tb

def arrow(s, x1, y1, x2, y2, color="7F7F7F", w=1.5, name="arrow"):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)); c.name = name
    c.line.color.rgb = rgb(color); c.line.width = Pt(w)
    t = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd")); t.set("type", "triangle"); t.set("w", "med"); t.set("len", "med"); return c

def title(s, t): text(s, 0.5, 0.3, 12.4, 0.7, [[(t, {"bold": True, "size": 28})]], name="title")
def cite(s, t): text(s, 0.5, 6.95, 12.4, 0.35, [[(t, {"size": 10, "color": "7F7F7F"})]], name="cite")

def pipeline(s, y, stages, gpu_groups, cut_after, transfer_label, before=True):
    """단계 상자 한 줄. gpu_groups: 각 단계가 놓인 GPU 라벨(분리 후). cut_after: 가위 위치(단계 인덱스 리스트)."""
    x0, w, gap = 0.7, 1.7, 0.35
    for i, (nm, kind, sub) in enumerate(stages):
        x = x0 + i * (w + gap)
        box(s, x, y, w, 0.75, fill=COMP if kind == "C" else MEM, line="7F7F7F", text=[nm, sub], size=11, bold=True, name=f"stage {nm}")
        if i < len(stages) - 1: arrow(s, x + w, y + 0.37, x + w + gap, y + 0.37, name=f"flow {i}")
    return x0, w, gap

def legend(s, x, y):
    text(s, x, y, 4.0, 0.3, [[("■ 연산 바운드   ", {"size": 10, "color": "C55A11"}), ("■ 메모리 바운드", {"size": 10, "color": "2F5597"})]], name="legend")

def right_notes(s, rows):
    for (head, body), y in zip(rows, [1.15, 2.35, 3.95, 5.0]):
        text(s, 8.3, y, 4.6, 0.3, [[(head, {"bold": True, "size": 13, "color": NAVY})]], name=f"note {head}")
        text(s, 8.3, y + 0.32, 4.6, 1.1, body, size=12, color=INK2, bullets=True, space=3, name=f"note body {head}")

# ========== ⑤ LLM PD 분할 ==========
s = prs.slides.add_slide(prs.slide_layouts[6]); title(s, "이기종 기법 ① LLM — Prefill / Decode 분할")
text(s, 0.7, 1.15, 7.2, 0.3, [[("기존: 한 GPU가 두 단계를 번갈아 처리 (혼합 배칭)", {"bold": True, "size": 13})]], name="before head")
box(s, 0.7, 1.5, 7.2, 1.1, fill=None, line="BFBFBF", dash=True, name="gpu before")
text(s, 0.85, 1.55, 1.2, 0.3, [[("GPU 1장", {"size": 10, "color": "7F7F7F"})]], name="gpu before label")
for i, (nm, kind) in enumerate([("Prefill", "C"), ("Decode", "M"), ("Prefill", "C"), ("Decode", "M"), ("Decode", "M")]):
    box(s, 1.0 + i * 1.35, 1.85, 1.2, 0.55, fill=COMP if kind == "C" else MEM, line="7F7F7F", text=nm, size=11, name=f"mix {i}")
text(s, 0.7, 2.7, 7.2, 0.3, [[("긴 프롬프트의 Prefill이 끼어들면 그 스텝의 모든 Decode가 느려짐 (TBT 튐) · Decode만 돌 때는 연산기가 놂", {"size": 11, "color": RED})]], name="before caption")
text(s, 0.7, 3.2, 7.2, 0.3, [[("Splitwise: 단계마다 전용 기계", {"bold": True, "size": 13})]], name="after head")
box(s, 0.7, 3.55, 3.3, 1.6, fill=None, line="C55A11", name="prompt machine")
text(s, 0.85, 3.6, 3.0, 0.3, [[("Prompt machine — 최신·빠른 GPU", {"size": 10, "color": "C55A11", "bold": True})]], name="pm label")
box(s, 0.95, 4.0, 2.8, 0.75, fill=COMP, line="7F7F7F", text=["Prefill", "연산 바운드 · 배치 키우면 TTFT 악화"], size=10, bold=True, name="prefill stage")
box(s, 4.6, 3.55, 3.3, 1.6, fill=None, line="2F5597", name="token machine")
text(s, 4.75, 3.6, 3.0, 0.3, [[("Token machine — 싼·구형·전력 제한 GPU", {"size": 10, "color": "2F5597", "bold": True})]], name="tm label")
box(s, 4.85, 4.0, 2.8, 0.75, fill=MEM, line="7F7F7F", text=["Decode", "메모리 바운드 · 배치 64까지 TBT 2배뿐"], size=10, bold=True, name="decode stage")
arrow(s, 4.0, 4.37, 4.6, 4.37, color=NAVY, w=2.25, name="kv xfer")
text(s, 3.5, 5.2, 1.7, 0.5, [[("KV 캐시 전송", {"size": 9, "color": NAVY, "bold": True})], [("레이어별로 Prefill과 겹쳐 숨김", {"size": 8, "color": NAVY})]], align=PP_ALIGN.CENTER, name="kv label")
legend(s, 0.7, 5.6)
text(s, 0.7, 5.95, 7.2, 0.95, [[("우리 박스(4090 + 3090)에서 확인한 것: ", {"bold": True, "size": 11})], [("입력이 길고 출력이 짧은 요청(code)에서 4090 Prefill + 3090 Decode가 통하고, 출력이 길면 Decode 쪽 KV 메모리가 먼저 찬다.", {"size": 11, "color": INK2})]], space=2, name="ours")
right_notes(s, [("무엇을 나누나", ["Prefill(프롬프트 처리)과 Decode(토큰 생성)를 다른 기계에", "요청은 Prompt 기계 → KV 전송 → Token 기계 순서"]),
                ("왜 이득인가", ["Prefill은 연산 집약, Decode는 메모리 대역폭에 묶여 최신 GPU의 연산·전력을 못 씀", "섞어 돌리면 서로 간섭 → 나누면 각 단계에 맞는 배치·하드웨어 선택 가능"]),
                ("대가", ["요청마다 KV 캐시(8B 모델 2K 프롬프트 ≈ 270 MB) 전송", "Token 기계에 KV가 몰림 → 출력이 길면 메모리 용량이 먼저 한계"]),
                ("보고된 효과", ["같은 비용에서 처리량 1.4배(비용 20% 절감), 같은 전력·비용에서 2.35배", "Decode를 A100(구형)에 둬도 처리량 유지"])])
cite(s, "Patel et al., \"Splitwise: Efficient Generative LLM Inference Using Phase Splitting\", ISCA 2024")
s.notes_slide.notes_text_frame.text = "Splitwise: prompt/token phase 특성 차이, 전용 기계, KV 전송 레이어별 오버랩, iso-cost/iso-power 비교."

# ========== ⑥ DiT 단계 분리 ==========
s = prs.slides.add_slide(prs.slide_layouts[6]); title(s, "이기종 기법 ② DiT — Text Encoder / Denoise / VAE 분리")
text(s, 0.7, 1.15, 7.2, 0.3, [[("기존: 한 GPU 그룹이 세 단계를 같은 병렬도로 통째로 처리", {"bold": True, "size": 13})]], name="before head")
box(s, 0.7, 1.5, 7.2, 1.15, fill=None, line="BFBFBF", dash=True, name="group before")
text(s, 0.85, 1.55, 2.5, 0.3, [[("GPU 그룹 (같은 병렬도 DoP)", {"size": 10, "color": "7F7F7F"})]], name="gb label")
for i, (nm, kind, sub) in enumerate([("Text Encoder", "M", "T5·CLIP, 가중치 큼"), ("Denoise (DiT)", "C", "수십 스텝 반복, E2E 대부분"), ("VAE Decode", "M", "병렬화 이득 없음")]):
    box(s, 1.0 + i * 2.35, 1.85, 2.1, 0.7, fill=COMP if kind == "C" else MEM, line="7F7F7F", text=[nm, sub], size=10, bold=True, name=f"mono {i}")
    if i < 2: arrow(s, 3.1 + i * 2.35, 2.2, 3.35 + i * 2.35, 2.2, name=f"mono flow {i}")
text(s, 0.7, 2.75, 7.2, 0.3, [[("GPU를 늘려도 VAE 시간은 안 줄어 전체에서 VAE 비중만 커짐 · 단계마다 필요한 GPU 수가 다른데 하나로 묶임", {"size": 11, "color": RED})]], name="before caption")
text(s, 0.7, 3.25, 7.2, 0.3, [[("DDiT / DisagFusion: 단계별 독립 GPU 그룹, 그룹마다 GPU 수·병렬도를 따로", {"bold": True, "size": 13})]], name="after head")
groups = [("Encoder 그룹", "M", "Text Encoder", "작게 · 가중치만 상주", "2F5597"), ("Denoise 그룹", "C", "DiT 백본", "크게 · 시퀀스 병렬", "C55A11"), ("VAE 그룹", "M", "VAE Decode", "작게 · 메모리 큰 카드", "2F5597")]
for i, (g, kind, nm, sub, col) in enumerate(groups):
    x = 0.7 + i * 2.45
    box(s, x, 3.6, 2.25, 1.55, fill=None, line=col, name=f"group {i}")
    text(s, x + 0.1, 3.65, 2.1, 0.3, [[(g, {"size": 10, "color": col, "bold": True})]], name=f"group label {i}")
    box(s, x + 0.15, 4.05, 1.95, 0.75, fill=COMP if kind == "C" else MEM, line="7F7F7F", text=[nm, sub], size=10, bold=True, name=f"gstage {i}")
    if i < 2: arrow(s, x + 2.25, 4.42, x + 2.45, 4.42, color=NAVY, w=2.25, name=f"latent {i}")
text(s, 2.5, 5.2, 1.4, 0.3, [[("임베딩 전송", {"size": 9, "color": NAVY, "bold": True})]], align=PP_ALIGN.CENTER, name="xfer1")
text(s, 4.95, 5.2, 1.4, 0.3, [[("Latent 전송(작음)", {"size": 9, "color": NAVY, "bold": True})]], align=PP_ALIGN.CENTER, name="xfer2")
legend(s, 0.7, 5.6)
text(s, 0.7, 5.95, 7.2, 0.95, [[("우리 측정(이전 세션): ", {"bold": True, "size": 11})], [("Denoise가 E2E 지연의 약 90%, VAE가 메모리의 약 50%를 차지 → 시간은 Denoise에, 메모리는 VAE에 몰려 있어 같은 카드에 두면 둘 다 낭비.", {"size": 11, "color": INK2})]], space=2, name="ours")
right_notes(s, [("무엇을 나누나", ["Text Encoder / DiT Denoise / VAE 를 독립 GPU 그룹에", "그룹마다 인스턴스 수와 병렬도(DoP)를 따로 정함"]),
                ("왜 이득인가", ["Denoise는 GPU를 늘릴수록 빨라지지만 VAE는 그렇지 않음 → 같은 병렬도로 묶으면 VAE 쪽 GPU가 놂", "Encoder·VAE는 가중치·활성 메모리가 커서 메모리 큰 카드가, Denoise는 연산 빠른 카드가 맞음"]),
                ("대가", ["단계 사이 임베딩·latent 전송 (latent는 압축 공간이라 작음)", "단계별 큐가 생겨 스케줄링이 필요"]),
                ("보고된 효과", ["DDiT: p99 지연 최대 1.44배, 평균 30% 개선 (OpenSora, T5 + DiT + VAE)", "DisagFusion: 소비자 GPU(24 GB)에서도 대형 모델 서빙, 단계별 탄력 스케줄링"])])
cite(s, "Huang et al., \"DDiT: Dynamic Resource Allocation for Diffusion Transformer Model Serving\", arXiv 2506.13497 (2025) · Zha et al., \"DisagFusion\", arXiv 2605.25550 (2026)")
s.notes_slide.notes_text_frame.text = "DDiT: DiT/VAE 이기종 병렬도, 독립 GPU 그룹. DisagFusion: encoder/transformer/decoder 분리, 소비자 GPU. 90%/50%는 우리 측정."

# ========== ⑦ VLM EPD 분할 ==========
s = prs.slides.add_slide(prs.slide_layouts[6]); title(s, "이기종 기법 ③ VLM — Encode / Prefill / Decode 분할")
text(s, 0.7, 1.15, 7.2, 0.3, [[("기존: 인코더와 LLM이 같은 GPU에 함께 상주", {"bold": True, "size": 13})]], name="before head")
box(s, 0.7, 1.5, 7.2, 1.15, fill=None, line="BFBFBF", dash=True, name="gpu before")
text(s, 0.85, 1.55, 2.5, 0.3, [[("GPU 1장 (인코더 + LLM 가중치)", {"size": 10, "color": "7F7F7F"})]], name="gb label")
for i, (nm, kind, sub) in enumerate([("Encode", "C", "이미지→토큰, 고해상도일수록 무거움"), ("Prefill", "C", "텍스트 + 이미지 토큰"), ("Decode", "M", "토큰 생성")]):
    box(s, 1.0 + i * 2.35, 1.85, 2.1, 0.7, fill=COMP if kind == "C" else MEM, line="7F7F7F", text=[nm, sub], size=10, bold=True, name=f"mono {i}")
    if i < 2: arrow(s, 3.1 + i * 2.35, 2.2, 3.35 + i * 2.35, 2.2, name=f"mono flow {i}")
text(s, 0.7, 2.75, 7.2, 0.3, [[("인코딩이 Prefill을 막아 TTFT 증가 · 두 모델이 메모리를 나눠 써서 배치와 이미지 수가 제한됨", {"size": 11, "color": RED})]], name="before caption")
text(s, 0.7, 3.25, 7.2, 0.3, [[("EPD Disaggregation: 세 단계를 전용 자원에", {"bold": True, "size": 13})]], name="after head")
groups = [("Encode 자원", "C", "멀티모달 인코더", "인코더 가중치만 · 큰 배치", "C55A11"), ("Prefill 자원", "C", "LLM Prefill", "LLM 가중치 · 연산", "C55A11"), ("Decode 자원", "M", "LLM Decode", "LLM 가중치 · KV 메모리", "2F5597")]
for i, (g, kind, nm, sub, col) in enumerate(groups):
    x = 0.7 + i * 2.45
    box(s, x, 3.6, 2.25, 1.55, fill=None, line=col, name=f"group {i}")
    text(s, x + 0.1, 3.65, 2.1, 0.3, [[(g, {"size": 10, "color": col, "bold": True})]], name=f"group label {i}")
    box(s, x + 0.15, 4.05, 1.95, 0.75, fill=COMP if kind == "C" else MEM, line="7F7F7F", text=[nm, sub], size=10, bold=True, name=f"gstage {i}")
    if i < 2: arrow(s, x + 2.25, 4.42, x + 2.45, 4.42, color=NAVY, w=2.25, name=f"xfer {i}")
text(s, 2.4, 5.2, 1.6, 0.3, [[("멀티모달 토큰 전송(캐시)", {"size": 9, "color": NAVY, "bold": True})]], align=PP_ALIGN.CENTER, name="xfer1")
text(s, 4.95, 5.2, 1.4, 0.3, [[("KV 캐시 전송", {"size": 9, "color": NAVY, "bold": True})]], align=PP_ALIGN.CENTER, name="xfer2")
legend(s, 0.7, 5.6)
text(s, 0.7, 5.95, 7.2, 0.95, [[("LLM PD 분할과의 관계: ", {"bold": True, "size": 11})], [("PD 분할 앞에 Encode 단계가 하나 더 붙은 꼴. 인코더는 LLM 가중치가 필요 없어 작은·구형 카드에 따로 둘 수 있다.", {"size": 11, "color": INK2})]], space=2, name="ours")
right_notes(s, [("무엇을 나누나", ["Encode(이미지·영상 → 토큰) / Prefill / Decode 를 전용 자원에", "단계마다 배칭·병렬화·스케줄링을 따로 설계"]),
                ("왜 이득인가", ["인코딩은 연산 집약이고 토큰 수를 크게 늘려 Prefill을 지연시킴", "인코더와 LLM을 한 GPU에 두면 메모리를 나눠 써서 배치·이미지 수가 제한됨"]),
                ("대가", ["멀티모달 토큰과 KV 캐시 전송 (토큰은 캐시해 재사용 가능)", "단계별 자원 배분을 워크로드에 맞춰 정해야 함"]),
                ("보고된 효과", ["메모리 15배 절감, 배치 22배, 요청당 이미지 10배", "SLO 만족률 최대 90~100% 개선, TTFT 최대 71% 감소"])])
cite(s, "Singh et al., \"Efficiently Serving Large Multimodal Models Using EPD Disaggregation\", ICML 2025")
s.notes_slide.notes_text_frame.text = "EPD: encode-prefill 간섭 제거, 메모리 분리, 멀티모달 토큰 캐시, 자원 배분 최적화."

# ========== ⑧ 우리가 하려는 것 ==========
s = prs.slides.add_slide(prs.slide_layouts[6]); title(s, "우리가 하려는 것 — 비어 있는 자원에 '맞는 단계'를 넣는다")
def gpu_card(s, x, y, w, h, label, kind_label, fills, name):
    box(s, x, y, w, h, fill=None, line="7F7F7F", name=name)
    text(s, x + 0.1, y + 0.05, w - 0.2, 0.3, [[(label, {"bold": True, "size": 12})], [(kind_label, {"size": 9, "color": "7F7F7F"})]], space=0, name=f"{name} label")
    yy = y + 0.75
    for (txt, fill, h_) in fills:
        box(s, x + 0.15, yy, w - 0.3, h_, fill=fill, line="7F7F7F" if fill not in (None, "FFFFFF") else "BFBFBF", text=txt, size=10, bold=True, name=f"{name} {txt[:8]}", dash=(fill is None))
        yy += h_ + 0.08
# 왼쪽: 지금
text(s, 0.7, 1.1, 5.8, 0.35, [[("지금: 요청 단위로 통째로 배정", {"bold": True, "size": 14})]], name="now head")
gpu_card(s, 0.7, 1.55, 2.7, 3.6, "GPU A (B300)", "요청 1: LLM, Decode 중", [("LLM Decode (메모리)", MEM, 0.6), ("연산기 대부분 유휴", None, 1.9)], "nowA")
gpu_card(s, 3.7, 1.55, 2.7, 3.6, "GPU B (PRO 6000)", "비어 있음", [("(대기 중인 DiT 요청은 GPU 하나가 통째로 빌 때까지 기다림)", None, 2.6)], "nowB")
text(s, 0.7, 5.3, 5.8, 0.6, [[("한 장은 메모리만 쓰며 연산기가 놀고, 한 장은 통째로 비어 있는데도 다음 요청은 기다린다", {"size": 11, "color": RED})]], name="now caption")
# 가운데 화살표
box(s, 6.6, 3.1, 0.6, 0.6, fill="4472C4", line=None, shape=MSO_SHAPE.RIGHT_ARROW, name="big arrow")
# 오른쪽: 하려는 것
text(s, 7.4, 1.1, 5.5, 0.35, [[("우리: 분할 기법으로 단계 단위로 채운다", {"bold": True, "size": 14})]], name="goal head")
gpu_card(s, 7.4, 1.55, 2.7, 3.6, "GPU A (B300)", "요청 1 Decode + 요청 2 Denoise", [("LLM Decode (메모리)", MEM, 0.6), ("DiT Denoise (연산)", COMP, 1.9)], "goalA")
gpu_card(s, 10.2, 1.55, 2.7, 3.6, "GPU B (PRO 6000)", "요청 2 VAE + 요청 3 Encode", [("VAE Decode (메모리 큼)", MEM, 1.2), ("VLM Encode (연산)", COMP, 1.3)], "goalB")
text(s, 7.4, 5.3, 5.5, 0.6, [[("요청을 단계로 쪼개 특성이 보완되는 곳에 넣으면 같은 두 장으로 세 요청을 동시에 처리", {"size": 11, "color": "385723"})]], name="goal caption")
box(s, 0.7, 6.0, 12.2, 0.8, fill=NAVY, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="solver bar")
text(s, 0.9, 6.08, 11.8, 0.65, [[("Solver가 정하는 것: ", {"bold": True, "color": "FFFFFF", "size": 13}), ("언제, 어느 요청의 어느 단계를, 어느 카드에 넣어야 SLO를 지키면서 비용 효율이 최대인가 — 분할이 늘 이득은 아니므로(전송·간섭·메모리) 모델로 미리 판단한다", {"color": "FFFFFF", "size": 12})]], anchor=MSO_ANCHOR.MIDDLE, name="solver text")
s.notes_slide.notes_text_frame.text = "목표 그림: 요청 단위 배정(왼쪽) vs 단계 단위 배정(오른쪽). Solver가 언제·무엇을·어디에."
prs.save("techniques.pptx"); print("saved")
