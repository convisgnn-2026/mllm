from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
rgb = lambda h: RGBColor.from_string(h)
ORANGE, BLUE, GREEN, GRAY, INK, INK2, NAVY, RED, LIGHT = "F4B183", "DDEBF7", "E2F0D9", "E7E6E6", "1F1F1F", "3C4A5A", "1E4D7B", "C00000", "F2F4F7"
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

def box(s, x, y, w, h, fill=None, line="7F7F7F", shape=MSO_SHAPE.RECTANGLE, text=None, size=14, bold=False, color=INK, align=PP_ALIGN.CENTER, lw=1.0, name="shape"):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h)); sh.name = name
    if fill: sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    else: sh.fill.background()
    if line: sh.line.color.rgb = rgb(line); sh.line.width = Pt(lw)
    else: sh.line.fill.background()
    sh.shadow.inherit = False
    if text is not None:
        tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, Inches(0.04))
        lines = text if isinstance(text, list) else [text]
        for i, t in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
            r = p.add_run(); r.text = t; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = rgb(color); r.font.name = "Calibri"
    return sh

def text(s, x, y, w, h, paras, size=14, color=INK, bold=False, italic=False, bullets=False, space=6, anchor=MSO_ANCHOR.TOP, name="text"):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tb.name = name
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, 0)
    for i, p_ in enumerate(paras if isinstance(paras, list) else [paras]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(space)
        segs = p_ if isinstance(p_, list) else [(p_, {})]
        for j, (seg, o) in enumerate(segs):
            r = p.add_run(); r.text = ("•  " if bullets and j == 0 else "") + seg
            r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", bold); r.font.italic = o.get("italic", italic); r.font.color.rgb = rgb(o.get("color", color)); r.font.name = "Calibri"
    return tb

def arrow(s, x1, y1, x2, y2, color=RED, w=2.25, name="arrow"):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)); c.name = name
    c.line.color.rgb = rgb(color); c.line.width = Pt(w)
    ln = c.line._get_or_add_ln(); from pptx.oxml.ns import qn; from lxml import etree
    t = etree.SubElement(ln, qn("a:tailEnd")); t.set("type", "triangle"); t.set("w", "med"); t.set("len", "med"); return c

def figure(s, router_text=("Router", "Round Robin"), router_fill=ORANGE, req_items=("LLM", "VLM", "Image DiT", "Video DiT", "…"), pool_labels=True, y0=0.0, show_output=True):
    # 요청 상자
    box(s, 0.6, 1.9 + y0, 2.6, 2.5, fill=GRAY, line=None, name="requests")
    text(s, 0.75, 2.0 + y0, 2.4, 0.4, [[("Inference Requests", {"bold": True, "size": 15})]], name="requests head")
    text(s, 0.9, 2.5 + y0, 2.2, 1.8, list(req_items), size=13, bullets=True, space=3, name="requests list")
    box(s, 3.4, 2.75 + y0, 0.8, 0.6, fill="4472C4", line=None, shape=MSO_SHAPE.RIGHT_ARROW, name="arrow in")
    # 라우터
    box(s, 4.4, 2.2 + y0, 3.0, 1.5, fill=router_fill, line="7F7F7F", name="router")
    text(s, 4.4, 2.25 + y0, 3.0, 0.45, [[(router_text[0], {"bold": True, "size": 15})]], name="router head").text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    arrow_line = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(4.4), Inches(2.7 + y0), Inches(7.4), Inches(2.7 + y0)); arrow_line.line.color.rgb = rgb("7F7F7F"); arrow_line.line.width = Pt(1)
    text(s, 4.4, 2.85 + y0, 3.0, 0.8, [[(router_text[1], {"bold": True, "size": 15})]], anchor=MSO_ANCHOR.MIDDLE, name="router body").text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    box(s, 5.6, 3.85 + y0, 0.6, 0.9, fill="4472C4", line=None, shape=MSO_SHAPE.DOWN_ARROW, name="arrow down")
    # 풀
    box(s, 1.9, 4.95 + y0, 9.2, 1.55, fill=ORANGE, line="7F7F7F", name="pool")
    nodes = []
    for i in range(4):
        x = 2.2 + i * 2.1
        a = box(s, x, 5.1 + y0, 1.7, 0.4, fill=BLUE, line="7F7F7F", text="Pro 6000 × 4" if pool_labels else "", size=12, name=f"pro{i}")
        b = box(s, x - 0.08, 5.75 + y0, 1.86, 0.6, fill=GREEN, line="7F7F7F", text="B300 × 8" if pool_labels else "", size=13, name=f"b300_{i}")
        nodes.append((x, a, b))
    if show_output:
        box(s, 11.3, 5.3 + y0, 1.0, 0.85, fill="4472C4", line=None, shape=MSO_SHAPE.RIGHT_ARROW, name="arrow out")
        text(s, 12.3, 5.5 + y0, 0.9, 0.45, [[("output", {"bold": True, "size": 15})]], name="output")
    return nodes

def title(s, t): text(s, 0.5, 0.3, 12.4, 0.7, [[(t, {"bold": True, "size": 30})]], name="title")
def side(s, paras, y=1.0, h=3.6): text(s, 8.1, y, 4.8, h, paras, size=14, color=INK2, space=8, name="side note")

# ---------- 3-1 지금의 플랫폼 ----------
s = prs.slides.add_slide(prs.slide_layouts[6]); title(s, "진행 중 프로젝트 ① 지금의 플랫폼")
nodes = figure(s)
for i, (x, a, b) in enumerate(nodes):
    arrow(s, 6.0, 3.5, x + 0.85, 5.1, name=f"rr{i}")
    text(s, x + 0.3, 4.55, 0.5, 0.35, [[(str(i + 1), {"bold": True, "size": 16, "color": RED})]], name=f"rrnum{i}")
side(s, [[("요청 종류에 상관없이, ", {}), ("GPU가 비어 있으면 그 GPU에 넣는다.", {"bold": True})],
         "LLM·VLM·이미지·영상 생성 요청이 한 라우터로 들어오고, 라운드로빈으로 다음 노드에 배정된다.",
         "AI 서빙은 LLM에서 멀티모달(이미지·영상)까지 넓어졌고, 회사는 자체 플랫폼을 고도화해 이를 한 풀에서 서비스하려 한다.",
         [("지금은 이 단순한 규칙만으로 운영된다.", {"italic": True})]], y=1.0, h=3.9)
s.notes_slide.notes_text_frame.text = "현재 플랫폼: 요청 종류 무관 라운드로빈. 멀티모달 요청이 한 풀로 들어옴."

# ---------- 3-2 요청마다 맞는 자원이 다르다 ----------
s = prs.slides.add_slide(prs.slide_layouts[6]); title(s, "진행 중 프로젝트 ② 요청마다 무거운 단계가 다르다")
box(s, 0.6, 1.9, 2.6, 2.5, fill=GRAY, line=None, name="requests")
text(s, 0.75, 2.0, 2.4, 0.4, [[("Inference Requests", {"bold": True, "size": 15})]], name="requests head")
rows = [("LLM", [("Prefill", "C"), ("Decode", "M")]), ("VLM", [("Encoder", "C"), ("Prefill", "C"), ("Decode", "M")]), ("DiT", [("Text Enc", "C"), ("Denoise", "C"), ("VAE", "M")])]
for i, (nm, stages) in enumerate(rows):
    y = 2.5 + i * 0.6
    text(s, 0.8, y, 0.7, 0.4, [[(nm, {"bold": True, "size": 13})]], anchor=MSO_ANCHOR.MIDDLE, name=f"req {nm}")
    for j, (st, kind) in enumerate(stages):
        box(s, 1.5 + j * 0.58, y + 0.02, 0.55, 0.36, fill="F8CBAD" if kind == "C" else "BDD7EE", line=None, text=st, size=8, name=f"{nm} {st}")
text(s, 0.75, 4.5, 2.5, 0.3, [[("■ 연산 바운드  ", {"size": 10, "color": "C55A11"}), ("■ 메모리 바운드", {"size": 10, "color": "2F5597"})]], name="legend")
box(s, 3.4, 2.75, 0.8, 0.6, fill="4472C4", line=None, shape=MSO_SHAPE.RIGHT_ARROW, name="arrow in")
box(s, 4.4, 2.2, 3.0, 1.5, fill=ORANGE, line="7F7F7F", name="router")
text(s, 4.4, 2.25, 3.0, 0.45, [[("Router", {"bold": True, "size": 15})]], name="router head").text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
text(s, 4.4, 2.85, 3.0, 0.8, [[("Round Robin", {"bold": True, "size": 15})], [("단계·특성 정보를 쓰지 않음", {"size": 11, "color": RED})]], anchor=MSO_ANCHOR.MIDDLE, name="router body")
for p in s.shapes[-1].text_frame.paragraphs: p.alignment = PP_ALIGN.CENTER
box(s, 5.6, 3.85, 0.6, 0.9, fill="4472C4", line=None, shape=MSO_SHAPE.DOWN_ARROW, name="arrow down")
box(s, 1.9, 4.95, 9.2, 1.55, fill=ORANGE, line="7F7F7F", name="pool")
for i in range(4):
    x = 2.2 + i * 2.1
    box(s, x, 5.1, 1.7, 0.4, fill=BLUE, line="7F7F7F", text="Pro 6000 × 4", size=12, name=f"pro{i}")
    box(s, x - 0.08, 5.75, 1.86, 0.6, fill=GREEN, line="7F7F7F", text="B300 × 8", size=13, name=f"b300_{i}")
text(s, 2.0, 6.6, 9.0, 0.35, [[("같은 요청이라도 어느 카드, 어느 단계에 놓이느냐에 따라 처리량·지연이 달라진다", {"size": 12, "italic": True, "color": INK2})]], name="pool caption")
side(s, [[("각 모델은 자원 특성이 다르다.", {"bold": True})],
         "LLM은 Prefill(연산)과 Decode(메모리), DiT는 Denoise(연산, E2E의 90%)와 VAE(메모리의 50%), VLM은 그 앞에 Encoder가 하나 더 있다.",
         "단계를 특성에 맞는 디바이스에 나눠 놓으면 처리량이 오른다는 연구가 모달리티마다 있다 (PD 분할, VAE 분리, EPD 분할).",
         [("지금 라우터는 이 정보를 전혀 쓰지 않는다.", {"italic": True})]], y=1.0, h=3.9)
s.notes_slide.notes_text_frame.text = "모달리티별 단계와 연산/메모리 특성. 라운드로빈은 이를 무시."

# ---------- 3-3 자원 쪽 제약 ----------
s = prs.slides.add_slide(prs.slide_layouts[6]); title(s, "진행 중 프로젝트 ③ 자원은 고정이고, 세대가 섞여 있다")
box(s, 0.6, 1.5, 7.2, 4.9, fill=ORANGE, line="7F7F7F", name="pool big")
text(s, 0.8, 1.6, 6.8, 0.4, [[("고정 온프레미스 GPU 풀 (늘릴 수 없음)", {"bold": True, "size": 14})]], name="pool head")
for i in range(4):
    x = 0.85 + i * 1.75
    box(s, x, 2.15, 1.55, 1.3, fill=BLUE, line="7F7F7F", text=["Pro 6000 × 4", "96 GB · PCIe", "노드 간 100 GbE"], size=10, name=f"pro{i}")
    box(s, x, 3.65, 1.55, 1.6, fill=GREEN, line="7F7F7F", text=["B300 × 8", "HBM3e · NVLink", "노드 간 IB"], size=10, name=f"b300_{i}")
    if i < 3:
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + 1.55), Inches(4.45), Inches(x + 1.75), Inches(4.45)); c.line.color.rgb = rgb("385723"); c.line.width = Pt(3); c.name = f"ib{i}"
text(s, 0.85, 5.4, 6.8, 0.9, [[("IB는 B300끼리만 이어지고, PRO 6000 노드와는 100 GbE로만 연결된다.", {"size": 11, "color": INK2})],
                              [("세대가 다르면 성능·메모리·지원 소프트웨어가 다르다 (예: RTX 30/40 계열은 CUDA 13 미지원).", {"size": 11, "color": INK2})]], name="pool caption")
side(s, [[("자원은 고정이다.", {"bold": True})],
         "운영자는 급변하는 AI 생태계에 대응하려 더 나은 자원을 확보하지만, 제조사도 새 장비를 빠르게 내놓아 세대가 섞인 풀이 된다.",
         "기존 장비는 아키텍처와 지원 소프트웨어가 달라, 최신 연구의 기법을 그대로 적용할 수 없는 경우가 생긴다.",
         [("그래서 운영자는 '이 제약 안에서 어떻게 배치해야 최적인가'를 매번 고민하게 된다.", {"italic": True})]], y=1.0, h=4.2)
s.notes_slide.notes_text_frame.text = "자원 제약: 고정, 이기종 세대, 링크 비대칭(IB vs 100GbE), 소프트웨어 호환."

# ---------- 3-4 목표 ----------
s = prs.slides.add_slide(prs.slide_layouts[6]); title(s, "진행 중 프로젝트 ④ 목표: 같은 풀에서 SLO를 지키며 비용 효율 최대")
box(s, 0.6, 1.9, 2.3, 2.5, fill=GRAY, line=None, name="requests")
text(s, 0.75, 2.0, 2.1, 0.4, [[("Inference Requests", {"bold": True, "size": 15})]], name="requests head")
text(s, 0.9, 2.5, 1.9, 1.8, ["LLM", "VLM", "Image DiT", "Video DiT", "…"], size=13, bullets=True, space=3, name="requests list")
box(s, 3.05, 2.75, 0.7, 0.6, fill="4472C4", line=None, shape=MSO_SHAPE.RIGHT_ARROW, name="arrow in")
box(s, 3.9, 2.2, 3.0, 1.5, fill="FFE699", line="7F7F7F", name="router")
text(s, 3.9, 2.25, 3.0, 0.45, [[("Router", {"bold": True, "size": 15})]], name="router head").text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
text(s, 3.9, 2.85, 3.0, 0.8, [[("Solver 가 배치 결정", {"bold": True, "size": 15})]], anchor=MSO_ANCHOR.MIDDLE, name="router body").text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
box(s, 1.0, 4.95, 6.9, 1.55, fill=ORANGE, line="7F7F7F", name="pool")
xs = []
for i in range(4):
    x = 1.2 + i * 1.65; xs.append(x)
    box(s, x, 5.1, 1.45, 0.4, fill=BLUE, line="7F7F7F", text="Pro 6000 × 4", size=11, name=f"pro{i}")
    box(s, x - 0.05, 5.75, 1.55, 0.6, fill=GREEN, line="7F7F7F", text="B300 × 8", size=12, name=f"b300_{i}")
arrow(s, 4.6, 3.7, xs[0] + 0.7, 5.1, color="C55A11", name="to pro0"); arrow(s, 5.0, 3.7, xs[1] + 0.7, 5.1, color="C55A11", name="to pro1")
arrow(s, 5.6, 3.7, xs[2] + 0.75, 5.75, color="385723", name="to b300_2"); arrow(s, 6.2, 3.7, xs[3] + 0.75, 5.75, color="385723", name="to b300_3")
text(s, 1.0, 6.6, 6.9, 0.35, [[("요청의 종류·단계·SLO를 보고, 맞는 카드와 분할 방식을 고른다", {"size": 12, "italic": True, "color": INK2})]], name="pool caption")
box(s, 8.3, 1.3, 4.5, 1.5, fill=LIGHT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="operator card")
text(s, 8.5, 1.4, 4.1, 1.35, [[("운영자", {"bold": True, "color": NAVY})], "고정 자원에서 최소 비용으로 최대한 많은 사용자를 서비스 → 비용 효율 최대", "동시에 사용자 SLO를 지킬 의무"], size=12.5, color=INK2, space=4, name="operator")
box(s, 8.3, 3.0, 4.5, 1.5, fill=LIGHT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="user card")
text(s, 8.5, 3.1, 4.1, 1.35, [[("사용자", {"bold": True, "color": NAVY})], "몰리는 시간에 무거운 요청이 오면 SLO가 깨짐", "같은 조건인데 대우가 달라지면 공정성 불만"], size=12.5, color=INK2, space=4, name="user")
box(s, 8.3, 4.7, 4.5, 1.5, fill=NAVY, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="goal card")
text(s, 8.5, 4.8, 4.1, 1.3, [[("Solver", {"bold": True, "color": "FFFFFF", "size": 14})], [("요청마다 SLO를 만족하는 가장 비용 효율적인 자원·분할 방식을 정한다", {"color": "FFFFFF"})]], size=12.5, space=4, name="goal")
s.notes_slide.notes_text_frame.text = "운영자: 비용 효율 최대 + SLO. 사용자: SLO, fairness. 라우터를 Solver가 이끈다."
prs.save("ongoing.pptx"); print("saved")
