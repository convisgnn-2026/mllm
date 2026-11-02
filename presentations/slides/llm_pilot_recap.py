from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

NAVY, GREEN, GRAY, LIGHT, INK, INK2, PALE = "1E4D7B", "2E8B6E", "7A7A7A", "F2F4F7", "1F1F1F", "3C4A5A", "D9E4EE"
rgb = lambda h: RGBColor.from_string(h)
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
s = prs.slides.add_slide(prs.slide_layouts[6])

def box(x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=""):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h)); sh.name = name or "shape"
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: sh.adjustments[0] = 0.06
    if fill: sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    else: sh.fill.background()
    if line: sh.line.color.rgb = rgb(line); sh.line.width = Pt(1.25)
    else: sh.line.fill.background()
    sh.shadow.inherit = False; return sh

def text(x, y, w, h, runs, size=14, color=INK, bold=False, italic=False, bullets=False, anchor=MSO_ANCHOR.TOP, space=6, name=""):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tb.name = name or "text"
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, 0)
    paras = runs if isinstance(runs, list) else [runs]
    for i, p_ in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(space)
        segs = p_ if isinstance(p_, list) else [(p_, {})]
        for seg, opt in segs:
            r = p.add_run(); r.text = ("•  " if bullets else "") + seg if seg is segs[0][0] else seg
            r.font.size = Pt(opt.get("size", size)); r.font.bold = opt.get("bold", bold); r.font.italic = opt.get("italic", italic)
            r.font.color.rgb = rgb(opt.get("color", color)); r.font.name = "Calibri"
    return tb

text(0.6, 0.35, 12.1, 0.8, "LLM-Pilot Recap", size=32, bold=True, name="title")
text(0.6, 1.25, 12.1, 0.6, [[("클라우드에서 LLM 추론 서빙을 할 때, ", {}), ("SLO를 만족하면서 가장 비용 효율적인 인스턴스·오프로딩 구성", {"bold": True, "color": NAVY}), ("을 자동으로 골라 주는 Solver  (CCGrid 2026)", {})]], size=16, color=INK2, name="one-liner")

box(0.6, 2.05, 4.0, 4.3, fill=LIGHT, name="problem card")
text(0.85, 2.2, 3.5, 0.4, "풀려던 문제", size=18, bold=True, color=NAVY, name="problem head")
text(0.85, 2.7, 3.55, 3.5, ["클라우드 인스턴스는 GPU·CPU·메모리·네트워크가 묶음으로 팔리고 시간 단위로 과금",
                            "오프로딩 전략(없음 / KVO / AO)과 인스턴스 종류의 조합이 너무 많음",
                            "비싼 GPU가 늘 답은 아님 — 워크로드와 SLO에 따라 최적 조합이 달라짐",
                            "하나씩 실측해 보기에는 비용·시간이 큼"], size=14, bullets=True, space=10, name="problem body")

steps = [("① 입력", "워크로드(입력·출력 길이, 배치), 모델,\nSLO(TTFT·TBT·TPS), 예산"),
         ("② 해석적 성능 모델", "연산량 F · 메모리 M · 전송량 D로 단계별 지연 예측,\n소량 실측으로 선형 보정"),
         ("③ Solver", "SLO·예산을 어기는 조합을 걸러내고\n남은 것 중 비용 효율(CE = 토큰/$) 최대를 선택")]
for i, (h, b) in enumerate(steps):
    y = 2.05 + i * 1.47
    box(4.95, y, 4.5, 1.25, fill="FFFFFF", line="9BB7D4", name=f"step {i+1}")
    text(5.15, y + 0.12, 4.1, 0.35, h, size=15, bold=True, color=NAVY, name=f"step {i+1} head")
    text(5.15, y + 0.5, 4.15, 0.7, b, size=12.5, color=INK2, space=2, name=f"step {i+1} body")
    if i < 2: box(7.05, y + 1.27, 0.3, 0.2, fill="9BB7D4", shape=MSO_SHAPE.DOWN_ARROW, name=f"arrow {i+1}")

box(9.8, 2.05, 2.9, 4.3, fill=NAVY, name="result card")
text(10.0, 2.2, 2.5, 0.4, "결과 (AWS 실측)", size=15, bold=True, color="FFFFFF", name="result head")
text(10.0, 2.65, 2.5, 0.9, "2.31×", size=48, bold=True, color="FFFFFF", name="stat1")
text(10.0, 3.6, 2.5, 0.5, "기존 최선 대비 비용 효율(토큰/$)", size=12, color=PALE, name="stat1 label")
text(10.0, 4.35, 2.5, 0.6, "실행 가능", size=28, bold=True, color="FFFFFF", name="stat2")
text(10.0, 4.95, 2.5, 1.2, "실시간 SLO에서 기존 방법이 전부 불가능 판정인 예산 구간에서도 구성을 찾아냄", size=12, color=PALE, name="stat2 label")

text(0.6, 6.55, 12.1, 0.75, "핵심 아이디어: 빠른 자원이 아니라 '워크로드와 SLO에 맞는' 자원을 모델로 고른다  →  이번 연구는 이 틀을 고정 온프레미스 이기종 GPU 풀과 멀티모달로 확장", size=13, italic=True, color=INK2, name="takeaway")
s.notes_slide.notes_text_frame.text = "LLM-Pilot(CCGrid 2026): 클라우드 VM 인스턴스와 오프로딩 전략(KVO, AO)의 조합을 해석적 성능 모델(F/M/D + 선형 보정)로 예측하고, SLO·예산 feasibility 필터 후 CE(tokens/$) 최대 구성을 선택. AWS 실측에서 CE 최대 2.31배, 기존 방법이 infeasible인 구간에서도 feasible 구성 발견."
prs.save("llm_pilot_recap.pptx"); print("saved")
