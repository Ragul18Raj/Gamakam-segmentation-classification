"""Write the results slide and test-case statuses into the review deck.
Every number on the slide is read from results*/summary.json produced by score.py."""
import copy, json, sys
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION

SRC, OUT = sys.argv[1], sys.argv[2]
clean = json.load(open("results/summary.json"))
hard = json.load(open("results_stress/summary.json"))
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
NAVY, TEAL, ORANGE, GREY = "1F2A30", "2A9D8F", "D9622B", "5B6B73"

prs = Presentation(SRC)


def set_text(shape, text):
    """Replace a text box's text, keeping the first run's formatting."""
    txb = shape.text_frame._txBody
    ps = txb.findall(f"{A}p")
    p0 = ps[0]
    for p in ps[1:]:
        txb.remove(p)
    runs = p0.findall(f"{A}r")
    for r in runs[1:]:
        p0.remove(r)
    runs[0].find(f"{A}t").text = text


def set_cell(tc, text, bold=None, color=None, fill=None, align=None):
    p = tc.find(f"{A}txBody").find(f"{A}p")
    runs = p.findall(f"{A}r")
    for r in runs[1:]:
        p.remove(r)
    r = runs[0]
    r.find(f"{A}t").text = text
    rpr = r.find(f"{A}rPr")
    if bold is not None:
        if bold:
            rpr.set("b", "1")
        elif "b" in rpr.attrib:
            del rpr.attrib["b"]
    if color:
        rpr.find(f"{A}solidFill").find(f"{A}srgbClr").set("val", color)
    if fill:
        tc.find(f"{A}tcPr").find(f"{A}solidFill").find(f"{A}srgbClr").set("val", fill)
    if align:
        ppr = p.find(f"{A}pPr")
        ppr.set("algn", align)


def f2(x):
    return f"{x:.2f}"


def pct(x):
    return f"{100 * x:.0f}%"


# ------------------------------------------------------------------ slide 14 (new results slide)
s = prs.slides[13]
sh = {x.name: x for x in s.shapes}
set_text(sh["Text 1"], "Results: pipeline check on simulated vocal contours")

card = sh["Shape 3"]
card.width = Inches(7.45)
set_text(sh["Text 4"], "Measured: 126 simulated clips, 7-fold leave-one-song-out")
sh["Text 4"].width = Inches(7.0)

rows = [
    ("Measure", "Clean sim", "Harder sim", "Target (real)"),
    (f"Stage-1 accuracy (chance {f2(clean['stage1']['chance'])})", f2(clean["stage1"]["acc"]), f2(hard["stage1"]["acc"]), "≥ 0.50"),
    (f"Frame macro-F1, CNN (all-none {f2(clean['all_none']['frame_macro_f1'])})", f2(clean["cnn"]["frame_macro_f1"]), f2(hard["cnn"]["frame_macro_f1"]), "≥ 0.60"),
    ("Event F1, CNN", f2(clean["cnn"]["event_f1_macro"]), f2(hard["cnn"]["event_f1_macro"]), "≥ 0.60"),
    ("Event F1, rule-based + 3-NN", f2(clean["rule_knn"]["event_f1_macro"]), f2(hard["rule_knn"]["event_f1_macro"]), "baseline"),
    ("Count error per type per clip, CNN", f2(clean["cnn"]["count_mae_per_type_per_clip"]), f2(hard["cnn"]["count_mae_per_type_per_clip"]), "≤ 0.5"),
    ("Clips with all counts exact, CNN", pct(clean["cnn"]["clips_all_counts_exact"]), pct(hard["cnn"]["clips_all_counts_exact"]), "≥ 40%"),
]
gf = sh["Table 0"]
tbl = gf._element.find(f".//{A}tbl")
grid = tbl.find(f"{A}tblGrid")
cols = grid.findall(f"{A}gridCol")
widths = [3.3, 1.15, 1.15, 1.35]
for i in range(len(widths) - len(cols)):
    gc = copy.deepcopy(cols[-1])
    ext = gc.find(f".//{{http://schemas.microsoft.com/office/drawing/2014/main}}colId")
    if ext is not None:
        ext.set("val", str(20002 + i))
    grid.append(gc)
for gc, w in zip(grid.findall(f"{A}gridCol"), widths):
    gc.set("w", str(int(w * 914400)))
trs = tbl.findall(f"{A}tr")
hdr_tpl, body_tpl = trs[0], trs[1]
for tr in trs:
    tbl.remove(tr)


def make_row(tpl, n):
    tr = copy.deepcopy(tpl)
    tcs = tr.findall(f"{A}tc")
    while len(tr.findall(f"{A}tc")) < n:
        tr.findall(f"{A}tc")[-1].addnext(copy.deepcopy(tcs[-1]))
    return tr


for ri, row in enumerate(rows):
    tr = make_row(hdr_tpl if ri == 0 else body_tpl, 4)
    rid = tr.find(f".//{{http://schemas.microsoft.com/office/drawing/2014/main}}rowId")
    if rid is not None:
        rid.set("val", str(10000 + ri))
    tr.set("h", str(int(0.40 * 914400)))
    for ci, (tc, txt) in enumerate(zip(tr.findall(f"{A}tc"), row)):
        if ri == 0:
            set_cell(tc, txt, align=None if ci == 0 else "ctr")
        else:
            set_cell(tc, txt, align=None if ci == 0 else "ctr",
                     color=(TEAL if ci == 3 else None), bold=(ci == 3),
                     fill=("EEF5F4" if ci == 0 else "FFFFFF"))
    tbl.append(tr)
gf.width = Inches(sum(widths))
gf.height = Inches(0.40 * len(rows))
tb = s.shapes.add_textbox(Inches(0.85), Inches(5.0), Inches(6.95), Inches(0.7))
tb.name = "Table note"
tf = tb.text_frame
tf.word_wrap = True
tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
r = tf.paragraphs[0].add_run()
r.text = ("Here the simple rule-based baseline is slightly ahead of the CNN, so on real singing the CNN "
          "has to earn its place by beating it. Targets were set before recording. "
          "Chance = share of the largest gamaka class.")
r.font.size, r.font.italic, r.font.name = Pt(12), True, "Calibri"
r.font.color.rgb = RGBColor.from_string(GREY)

# right card: chart of event F1 by tempo
rc = sh["Shape 5"]
rc.left, rc.width = Inches(8.25), Inches(4.45)
set_text(sh["Text 6"], "Event F1 by tempo, CNN")
sh["Text 6"].left, sh["Text 6"].width = Inches(8.5), Inches(4.0)
for name in ("Text 7", "Shape 8", "Text 9", "Text 10"):
    el = sh[name]._element
    el.getparent().remove(el)

cd = CategoryChartData()
cd.categories = ["Slow", "Medium", "Fast"]
cd.add_series("Clean sim", [round(clean["cnn"]["event_f1_by_tempo"][k], 2) for k in ("slow", "medium", "fast")])
cd.add_series("Harder sim", [round(hard["cnn"]["event_f1_by_tempo"][k], 2) for k in ("slow", "medium", "fast")])
gfc = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(8.45), Inches(2.0), Inches(4.05), Inches(3.75), cd)
gfc.name = "Chart tempo"
ch = gfc.chart
ch.font.size = Pt(12)
ch.font.name = "Calibri"
ch.font.color.rgb = RGBColor.from_string(NAVY)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 0, 1
va.major_unit = 0.25
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RGBColor.from_string("D5DDE0")
va.format.line.fill.background()
va.tick_labels.font.size = Pt(11)
va.tick_labels.font.color.rgb = RGBColor.from_string(GREY)
ca = ch.category_axis
ca.tick_labels.font.size = Pt(12)
ca.format.line.color.rgb = RGBColor.from_string("D5DDE0")
plot = ch.plots[0]
plot.gap_width = 70
plot.overlap = -10
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked = "0.00", False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(11)
for ser, col in zip(plot.series, (TEAL, ORANGE)):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = RGBColor.from_string(col)

set_text(sh["Text 12"], "Simulated, not recorded: synthetic pitch curves with known G1–G3 were run through the full pipeline. "
         "This checks the code and the scoring end to end; real singing is expected to score lower. "
         "Vocal recording in November; measured results by 30 November.")

notes = (
    "These numbers come from simulated data, not from recordings. I generated 126 synthetic vocal-style pitch curves "
    "following the recording plan (7 songs x 3 gamakas x 3 tempos x 2 octaves), with known G1, G2 and G3 events, "
    "look-alike ornaments labelled none, CREPE-like noise, tonic error and annotator boundary jitter. "
    "The harder condition makes the look-alike ornaments overlap the targets and widens the boundary jitter. "
    "The full pipeline ran with 7-fold leave-one-song-out: features, the dilated CNN (85,972 parameters), "
    "post-processing, and event scoring with a 200 ms onset tolerance. "
    "The point is that the pipeline and evaluation work end to end before the singer records. "
    "The right-hand column gives the targets for the real vocal data, set before recording."
)
ns = s.notes_slide
if ns.notes_text_frame is None:      # this deck's notes master has no body placeholder: borrow one
    donor = prs.slides[12].notes_slide
    for ph in donor.placeholders:
        if ph.placeholder_format.type == 2:
            ns.shapes._spTree.append(copy.deepcopy(ph._element))
ns.notes_text_frame.text = notes

# ------------------------------------------------------------------ last slide page number
last = prs.slides[14]
for x in last.shapes:
    if x.name == "Text 2" and x.text_frame.text.strip() == "14":
        set_text(x, "15")

# ------------------------------------------------------------------ slide 9 test-case statuses
t9 = [x for x in prs.slides[8].shapes if x.has_table][0].table
new_status = {"T1": "Pending (needs recordings)", "T2": "Pending (needs recordings)",
              "T4": "Pass (simulated data)", "T5": "Pass (simulated data)", "T6": "Pass (simulated data)",
              "T7": "Pass (simulated data)", "T8": "Pass (simulated data)", "T9": "Pass (simulated data)"}
for r in t9.rows:
    k = r.cells[0].text.strip()
    if k in new_status:
        set_cell(r.cells[4]._tc, new_status[k])

prs.save(OUT)
print("saved", OUT)
