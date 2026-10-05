"""Animated, interactive ECharts rendered in an iframe (no Streamlit component package needed)."""
import json
import re

import numpy as np
import streamlit as st

from ui.theme import AWAY, CLASS_COLORS, FAINT, GRID, MUTED, SERIES, SURFACE, TEXT, TOWARD

ECHARTS = "https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"


class _Opt:
    """Collects raw JS functions (formatters) while an option dict is built."""

    def __init__(self):
        self.fns: list[str] = []

    def js(self, fn: str) -> str:
        self.fns.append(fn)
        return f"@@JS{len(self.fns) - 1}@@"

    def dump(self, option: dict) -> str:
        s = json.dumps(option, default=_np, allow_nan=False).replace("</", "<\\/")  # data can't close the <script>
        return re.sub(r'"@@JS(\d+)@@"', lambda m: self.fns[int(m.group(1))], s)


def _np(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _f(v):
    """JSON-safe float (NaN -> null)."""
    v = float(v)
    return None if np.isnan(v) else v


BASE = {
    "backgroundColor": "transparent",
    "textStyle": {"fontFamily": "Inter, system-ui, sans-serif", "color": MUTED},
    "animationDuration": 1100, "animationEasing": "cubicOut", "animationDurationUpdate": 600,
}
TOOLTIP = {"backgroundColor": "#1f1e26", "borderColor": "#3a3942", "textStyle": {"color": TEXT, "fontSize": 12},
           "extraCssText": "border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.4);"}


def render(o: _Opt, option: dict, height: int = 340) -> None:
    opt = {**BASE, **option, "tooltip": {**TOOLTIP, **option.get("tooltip", {})}}
    st.iframe(f"""<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:transparent}}</style></head><body>
<div id="c" style="width:100%;height:{height - 8}px"></div>
<script src="{ECHARTS}"></script>
<script>
const opt = {o.dump(opt)};
if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) opt.animation = false;
const ch = echarts.init(document.getElementById('c'), null, {{renderer: 'canvas'}});
ch.setOption(opt);
window.addEventListener('resize', () => ch.resize());
</script></body></html>""", height=height)


def _axis(**kw) -> dict:
    base = {"axisLine": {"lineStyle": {"color": GRID}}, "axisTick": {"show": False},
            "axisLabel": {"color": MUTED, "fontSize": 11}, "splitLine": {"lineStyle": {"color": GRID, "type": "dashed"}},
            "nameTextStyle": {"color": FAINT, "fontSize": 11}}
    base.update(kw)
    return base


def _legend(**kw) -> dict:
    return {"textStyle": {"color": MUTED, "fontSize": 11}, "icon": "roundRect", "itemWidth": 10, "itemHeight": 10, **kw}


# ------------------------------------------------------------------ one post

def gauge(value: float, title: str, color: str, max_value: float = 100, suffix: str = "%", height: int = 250) -> None:
    o = _Opt()
    render(o, {"series": [{
        "type": "gauge", "startAngle": 210, "endAngle": -30, "min": 0, "max": max_value,
        "progress": {"show": True, "width": 16, "roundCap": True, "itemStyle": {"color": color}},
        "axisLine": {"roundCap": True, "lineStyle": {"width": 16, "color": [[1, GRID]]}},
        "pointer": {"show": False}, "axisTick": {"show": False}, "splitLine": {"show": False},
        "axisLabel": {"show": False}, "anchor": {"show": False},
        "title": {"offsetCenter": [0, "34%"], "color": FAINT, "fontSize": 12},
        "detail": {"valueAnimation": True, "offsetCenter": [0, "-4%"], "fontSize": 34, "fontWeight": 800, "color": TEXT,
                   "formatter": o.js(f"function(v){{return v.toFixed(1)+'{suffix}'}}")},
        "data": [{"value": round(float(value), 2), "name": title}],
    }]}, height)


def class_donut(proba: dict, height: int = 250) -> None:
    o = _Opt()
    render(o, {
        "tooltip": {"trigger": "item",
                    "formatter": o.js("function(p){return p.marker+p.name+': <b>'+(p.value*100).toFixed(1)+'%</b>'}")},
        "legend": _legend(bottom=0),
        "series": [{"type": "pie", "radius": ["52%", "78%"], "center": ["50%", "44%"], "padAngle": 2,
                    "itemStyle": {"borderRadius": 6, "borderColor": SURFACE, "borderWidth": 2},
                    "label": {"show": False, "position": "center"},
                    "emphasis": {"scaleSize": 6, "label": {
                        "show": True, "fontSize": 17, "fontWeight": 800, "color": TEXT,
                        "formatter": o.js("function(p){return p.name+'\\n'+(p.value*100).toFixed(0)+'%'}")}},
                    "animationType": "scale",
                    "data": [{"name": k, "value": float(v), "itemStyle": {"color": CLASS_COLORS[k]}} for k, v in proba.items()]}],
    }, height)


def shap_bars(names: list[str], values: list[float], height: int = 330) -> None:
    o = _Opt()
    render(o, {
        "grid": {"left": 24, "right": 24, "top": 10, "bottom": 30, "containLabel": True},
        "tooltip": {"trigger": "item", "formatter": o.js(
            "function(p){return p.name+'<br/>'+(p.value>0?'pushes toward':'pulls away from')+' Viral: <b>'+p.value.toFixed(3)+'</b>'}")},
        "xAxis": _axis(type="value", name="SHAP contribution to Viral log-odds", nameLocation="middle", nameGap=22),
        "yAxis": _axis(type="category", data=names, splitLine={"show": False}),
        "series": [{"type": "bar", "barWidth": "56%",
                    "data": [{"value": float(v), "itemStyle": {
                        "color": TOWARD if v > 0 else AWAY, "borderRadius": [0, 4, 4, 0] if v > 0 else [4, 0, 0, 4]}}
                        for v in values],
                    "animationDelay": o.js("function(i){return i*70}")}],
    }, height)


def radar(factors: list[str], series: dict, height: int = 360) -> None:
    o = _Opt()
    render(o, {
        "legend": _legend(bottom=0),
        "tooltip": {"trigger": "item"},
        "radar": {"indicator": [{"name": f.replace("_", " "), "max": 10} for f in factors], "radius": "64%",
                  "axisName": {"color": MUTED, "fontSize": 11}, "splitLine": {"lineStyle": {"color": GRID}},
                  "splitArea": {"areaStyle": {"color": ["transparent"]}}, "axisLine": {"lineStyle": {"color": GRID}}},
        "series": [{"type": "radar", "symbolSize": 7, "data": [
            {"name": n, "value": [round(float(x), 2) for x in v], "lineStyle": {"width": 2, "color": SERIES[i]},
             "itemStyle": {"color": SERIES[i]}, "areaStyle": {"color": SERIES[i], "opacity": 0.2 if i == 0 else 0.06}}
            for i, (n, v) in enumerate(series.items())]}],
    }, height)


def heatmap(rows: list[str], cols: list[str], values, vmin: float, vmax: float, height: int = 380,
            colors=("#16263a", "#3987e5", "#cfe3fb"), decimals: int = 2) -> None:
    o = _Opt()
    data = [[j, i, None if values[i][j] is None or np.isnan(values[i][j]) else round(float(values[i][j]), 3)]
            for i in range(len(rows)) for j in range(len(cols))]
    render(o, {
        "grid": {"left": 8, "right": 16, "top": 8, "bottom": 54, "containLabel": True},
        "tooltip": {"position": "top", "formatter": o.js(
            f"function(p){{var R={json.dumps(rows)},C={json.dumps(cols)};"
            f"return R[p.value[1]]+' · '+C[p.value[0]]+': <b>'+(p.value[2]===null?'n/a':p.value[2].toFixed({decimals}))+'</b>'}}")},
        "xAxis": _axis(type="category", data=cols, splitLine={"show": False},
                       axisLabel={"color": MUTED, "fontSize": 10, "interval": 0}),
        "yAxis": _axis(type="category", data=rows, splitLine={"show": False}, inverse=True),
        "visualMap": {"min": vmin, "max": vmax, "calculable": True, "orient": "horizontal", "left": "center", "bottom": 0,
                      "itemHeight": 140, "inRange": {"color": list(colors)}, "textStyle": {"color": MUTED}},
        "series": [{"type": "heatmap", "name": "", "data": data,
                    "label": {"show": True, "color": TEXT, "fontSize": 10, "formatter": o.js(
                        f"function(p){{return p.value[2]===null?'':p.value[2].toFixed({decimals})}}")},
                    "itemStyle": {"borderColor": SURFACE, "borderWidth": 2, "borderRadius": 4},
                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,.5)"}}}],
    }, height)


def clock(slot_labels: list[str], values: list[float], best: int, height: int = 360) -> None:
    o = _Opt()
    render(o, {
        "tooltip": {"trigger": "item", "formatter": o.js(
            "function(p){return p.name+'<br/>expected reward: <b>'+p.value.toFixed(3)+'</b>'}")},
        "angleAxis": {"type": "category", "data": slot_labels, "startAngle": 90,
                      "axisLabel": {"color": MUTED, "fontSize": 10}, "axisLine": {"lineStyle": {"color": GRID}}},
        "radiusAxis": {"axisLabel": {"show": False}, "splitLine": {"lineStyle": {"color": GRID}}, "axisLine": {"show": False}},
        "polar": {"radius": ["14%", "76%"]},
        "series": [{"type": "bar", "coordinateSystem": "polar", "roundCap": True,
                    "data": [{"name": slot_labels[i], "value": float(v),
                              "itemStyle": {"color": CLASS_COLORS["Viral"] if i == best else "#2f4a6b"}}
                             for i, v in enumerate(values)],
                    "animationDelay": o.js("function(i){return i*90}")}],
    }, height)


def hbars(names: list[str], values: list[float], color: str = SERIES[0], height: int = 300, pct: bool = False,
          xname: str = "", decimals: int = 3) -> None:
    o = _Opt()
    fmt = "(v*100).toFixed(1)+'%'" if pct else f"v.toFixed({decimals})"
    render(o, {
        "grid": {"left": 24, "right": 52, "top": 8, "bottom": 30 if xname else 8, "containLabel": True},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"},
                    "valueFormatter": o.js(f"function(v){{return {fmt}}}")},
        "xAxis": _axis(type="value", name=xname, nameLocation="middle", nameGap=22),
        "yAxis": _axis(type="category", data=names, splitLine={"show": False}),
        "series": [{"type": "bar", "data": [_f(v) for v in values], "barWidth": "56%",
                    "itemStyle": {"color": color, "borderRadius": [0, 4, 4, 0]},
                    "label": {"show": True, "position": "right", "color": MUTED, "fontSize": 10,
                              "formatter": o.js(f"function(p){{var v=p.value;return {fmt}}}")},
                    "animationDelay": o.js("function(i){return i*60}")}],
    }, height)


# ------------------------------------------------------------------ report charts

def grouped_bars_err(categories: list[str], groups: dict, height: int = 340, yname: str = "", ymax=None,
                     colors: list[str] | None = None) -> None:
    """groups: name -> (means, stds). Bars with whisker error bars (± 1 std)."""
    o = _Opt()
    n, series = len(groups), []
    for i, (name, (means, stds)) in enumerate(groups.items()):
        pal = colors or SERIES
        series.append({"name": name, "type": "bar", "data": [_f(m) for m in means], "barGap": "18%",
                       "itemStyle": {"color": pal[i] if i < len(pal) else FAINT, "borderRadius": [4, 4, 0, 0]},
                       "animationDelay": o.js(f"function(k){{return k*80+{i * 140}}}")})
        if all(s is None or np.isnan(s) for s in stds):
            continue
        series.append({"name": name, "type": "custom", "z": 10, "tooltip": {"show": False}, "silent": True,
                       "renderItem": o.js(f"""function(params, api) {{
                           if (api.value(1) === api.value(2)) return;
                           var x = api.value(0), lo = api.coord([x, api.value(1)]), hi = api.coord([x, api.value(2)]);
                           var band = api.size([1, 0])[0], bw = band * 0.7 / {n}, cx = lo[0] + ({i} - ({n} - 1) / 2) * bw * 1.18;
                           var s = {{stroke: '{MUTED}', lineWidth: 1.2}};
                           return {{type: 'group', children: [
                             {{type: 'line', shape: {{x1: cx, y1: lo[1], x2: cx, y2: hi[1]}}, style: s}},
                             {{type: 'line', shape: {{x1: cx - 4, y1: lo[1], x2: cx + 4, y2: lo[1]}}, style: s}},
                             {{type: 'line', shape: {{x1: cx - 4, y1: hi[1], x2: cx + 4, y2: hi[1]}}, style: s}}]}};
                         }}"""),
                       "data": [[k, _f(m - (0 if np.isnan(s) else s)), _f(m + (0 if np.isnan(s) else s))]
                                for k, (m, s) in enumerate(zip(means, stds))]})
    render(o, {
        "legend": _legend(top=0, data=list(groups)),
        "grid": {"left": 8, "right": 16, "top": 36, "bottom": 8, "containLabel": True},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"},
                    "valueFormatter": o.js("function(v){return v==null?'':v.toFixed(3)}")},
        "xAxis": _axis(type="category", data=categories, splitLine={"show": False},
                       axisLabel={"color": MUTED, "fontSize": 11, "interval": 0}),
        "yAxis": _axis(type="value", name=yname, max=ymax),
        "series": series,
    }, height)


def lines(x, series: dict, height: int = 340, xname: str = "", yname: str = "", ref: tuple | None = None,
          area: bool = False, smooth: bool = False, ymax=None, ymin=None) -> None:
    """x: shared x values, or dict name -> x values."""
    o = _Opt()
    ss = []
    for i, (name, ys) in enumerate(series.items()):
        xs = x[name] if isinstance(x, dict) else x
        color = SERIES[i] if i < len(SERIES) else FAINT  # never cycle hues: extra series fall back to neutral
        s = {"name": name, "type": "line", "data": [[_f(a), _f(b)] for a, b in zip(xs, ys)],
             "showSymbol": False, "smooth": smooth, "connectNulls": False,
             "lineStyle": {"width": 2, "color": color}, "itemStyle": {"color": color}, "emphasis": {"focus": "series"}}
        if area:
            s["areaStyle"] = {"opacity": 0.1}
        ss.append(s)
    if ref and ss:
        ss[0]["markLine"] = {"symbol": "none", "silent": True, "lineStyle": {"color": FAINT, "type": "dashed"},
                             "label": {"color": FAINT, "formatter": ref[0], "position": "insideEndTop"},
                             "data": [{"yAxis": float(ref[1])}]}
    render(o, {
        "legend": _legend(top=0, type="scroll"),
        "grid": {"left": 8, "right": 20, "top": 36, "bottom": 30, "containLabel": True},
        "tooltip": {"trigger": "axis", "valueFormatter": o.js("function(v){return v==null?'':v.toFixed(3)}")},
        "xAxis": _axis(type="value", name=xname, nameLocation="middle", nameGap=24, splitLine={"show": False},
                       min="dataMin", max="dataMax"),
        "yAxis": _axis(type="value", name=yname, max=ymax, min=ymin, scale=ymin is not None),
        "dataZoom": [{"type": "inside"}],
        "series": ss,
    }, height)


def scatter_groups(groups: dict, height: int = 420, xname: str = "", yname: str = "") -> None:
    o = _Opt()
    render(o, {
        "legend": _legend(top=0),
        "grid": {"left": 8, "right": 16, "top": 36, "bottom": 30, "containLabel": True},
        "tooltip": {"trigger": "item", "formatter": o.js("function(p){return p.marker+p.seriesName}")},
        "xAxis": _axis(type="value", name=xname, nameLocation="middle", nameGap=22, splitLine={"show": False}),
        "yAxis": _axis(type="value", name=yname, splitLine={"show": False}),
        "dataZoom": [{"type": "inside", "xAxisIndex": 0}, {"type": "inside", "yAxisIndex": 0}],
        "series": [{"name": n, "type": "scatter", "symbolSize": 6, "data": [[_f(a), _f(b)] for a, b in pts],
                    "itemStyle": {"color": SERIES[i] if i < len(SERIES) else FAINT, "opacity": 0.7, "borderColor": SURFACE, "borderWidth": 0.5},
                    "emphasis": {"focus": "series", "itemStyle": {"opacity": 1}},
                    "animationDelay": o.js("function(k){return k*0.5}")}
                   for i, (n, pts) in enumerate(groups.items())],
    }, height)


def stacked_share(categories: list[str], shares: dict, height: int = 300) -> None:
    o = _Opt()
    render(o, {
        "legend": _legend(top=0),
        "grid": {"left": 8, "right": 16, "top": 36, "bottom": 8, "containLabel": True},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"},
                    "valueFormatter": o.js("function(v){return (v*100).toFixed(1)+'%'}")},
        "xAxis": _axis(type="value", max=1, axisLabel={"color": MUTED, "formatter": o.js("function(v){return Math.round(v*100)+'%'}")}),
        "yAxis": _axis(type="category", data=categories, splitLine={"show": False}),
        "series": [{"name": k, "type": "bar", "stack": "s", "data": [_f(x) for x in v], "barWidth": "58%",
                    "itemStyle": {"color": CLASS_COLORS.get(k, SERIES[i]), "borderColor": SURFACE, "borderWidth": 2}}
                   for i, (k, v) in enumerate(shares.items())],
    }, height)


def sunburst(data: list, height: int = 430) -> None:
    o = _Opt()
    render(o, {
        "tooltip": {"trigger": "item", "formatter": "{b}: {c} posts"},
        "series": [{"type": "sunburst", "data": data, "radius": ["10%", "94%"], "sort": None,
                    "itemStyle": {"borderColor": SURFACE, "borderWidth": 2, "borderRadius": 4},
                    "label": {"color": TEXT, "fontSize": 11, "minAngle": 10},
                    "emphasis": {"focus": "ancestor"},
                    "levels": [{}, {"r0": "10%", "r": "38%"}, {"r0": "38%", "r": "66%", "label": {"rotate": "tangential"}},
                               {"r0": "66%", "r": "92%", "label": {"rotate": "tangential", "fontSize": 10}}]}],
    }, height)


def histogram(groups: dict, bins: np.ndarray, height: int = 320, xname: str = "") -> None:
    centers = (bins[:-1] + bins[1:]) / 2
    series = {}
    for name, vals in groups.items():
        h, _ = np.histogram(vals, bins=bins)
        series[name] = (h / max(1, h.sum())).tolist()
    lines(centers, series, height, xname=xname, yname="share of posts", area=True, smooth=True)


def calendar_heat(weekdays: list[str], hours: list[int], values, height: int = 300) -> None:
    heatmap(weekdays, [f"{h:02d}" for h in hours], values, 0, float(np.nanmax(values)) if np.size(values) else 1,
            height, colors=("#1a1622", "#8134af", "#f58529"), decimals=2)


def forest(names: list[str], est: list[float], lo: list[float], hi: list[float], height: int = 300,
           xname: str = "", pct: bool = True) -> None:
    """Point estimates with 95% intervals around a zero line; green where the interval excludes zero."""
    o = _Opt()
    fmt = "(v*100).toFixed(1)+' pts'" if pct else "v.toFixed(3)"
    data = [[_f(e), i, _f(l), _f(h)] for i, (e, l, h) in enumerate(zip(est, lo, hi))]
    sig = [(l is not None and not np.isnan(l) and l > 0) or (h is not None and not np.isnan(h) and h < 0)
           for l, h in zip(lo, hi)]
    render(o, {
        "grid": {"left": 24, "right": 30, "top": 10, "bottom": 34, "containLabel": True},
        "tooltip": {"trigger": "item", "formatter": o.js(
            f"function(p){{var v=p.value[0],l=p.value[2],h=p.value[3];var f=function(v){{return {fmt}}};"
            "return p.name+'<br/>lift <b>'+f(v)+'</b><br/>95% CI '+(l==null?'n/a':f(l)+' to '+f(h))}")},
        "xAxis": _axis(type="value", name=xname, nameLocation="middle", nameGap=24,
                       axisLabel={"color": MUTED, "formatter": o.js(f"function(v){{return {fmt}}}")}),
        "yAxis": _axis(type="category", data=names, splitLine={"show": False}),
        "series": [
            {"type": "custom", "silent": True, "z": 2, "renderItem": o.js(f"""function(params, api) {{
                if (api.value(2) === null || isNaN(api.value(2))) return;
                var y = api.value(1), a = api.coord([api.value(2), y]), b = api.coord([api.value(3), y]);
                var s = {{stroke: '{MUTED}', lineWidth: 2}};
                return {{type: 'group', children: [
                  {{type: 'line', shape: {{x1: a[0], y1: a[1], x2: b[0], y2: b[1]}}, style: s}},
                  {{type: 'line', shape: {{x1: a[0], y1: a[1] - 5, x2: a[0], y2: a[1] + 5}}, style: s}},
                  {{type: 'line', shape: {{x1: b[0], y1: b[1] - 5, x2: b[0], y2: b[1] + 5}}, style: s}}]}};
              }}"""), "data": data, "encode": {"x": [0, 2, 3], "y": 1}},
            {"type": "scatter", "z": 3, "symbolSize": 12, "data": [
                {"name": n, "value": d, "itemStyle": {"color": CLASS_COLORS["Viral"] if s_ else SERIES[0],
                                                       "borderColor": SURFACE, "borderWidth": 2}}
                for n, d, s_ in zip(names, data, sig)],
             "markLine": {"symbol": "none", "silent": True, "lineStyle": {"color": FAINT, "type": "dashed"},
                          "label": {"show": False}, "data": [{"xAxis": 0}]}},
        ],
    }, height)


def flow(nodes: list[dict], edges: list[tuple[str, str]], categories: list[str], height: int = 620) -> None:
    """Animated architecture diagram. nodes: {id, x, y, cat, desc}; particles flow along each edge."""
    o = _Opt()
    pos = {n["id"]: (n["x"], n["y"]) for n in nodes}
    palette = SERIES + ["#e66767"]
    graph_nodes = [{"name": n["id"], "value": [n["x"], n["y"]], "category": categories.index(n["cat"]),
                    "symbolSize": n.get("size", 18), "desc": n.get("desc", ""),
                    "label": {"show": True, "position": n.get("label_pos", "bottom"), "color": TEXT, "fontSize": 11,
                              "fontWeight": 600}} for n in nodes]
    links = [{"source": a, "target": b} for a, b in edges]
    lines = [{"coords": [list(pos[a]), list(pos[b])]} for a, b in edges]
    render(o, {
        "tooltip": {"trigger": "item", "formatter": o.js(
            "function(p){if(p.dataType==='node'){return '<b>'+p.name+'</b><br/><span style=\"color:#c3c2b7\">'"
            "+(p.data.desc||'')+'</span>'} return ''}"), "extraCssText": "max-width:320px;white-space:normal;"},
        "legend": _legend(top=0, data=categories),
        "xAxis": {"show": False, "min": 0, "max": 100}, "yAxis": {"show": False, "min": 0, "max": 100},
        "grid": {"left": 10, "right": 10, "top": 40, "bottom": 30},
        "series": [
            {"type": "graph", "coordinateSystem": "cartesian2d", "data": graph_nodes, "links": links,
             "categories": [{"name": c, "itemStyle": {"color": palette[i % len(palette)]}} for i, c in enumerate(categories)],
             "edgeSymbol": ["none", "arrow"], "edgeSymbolSize": 7,
             "lineStyle": {"color": "#4a4954", "width": 1.6, "curveness": 0.12, "opacity": 0.9},
             "itemStyle": {"borderColor": SURFACE, "borderWidth": 3, "shadowBlur": 16, "shadowColor": "rgba(221,42,123,.35)"},
             "emphasis": {"focus": "adjacency", "lineStyle": {"width": 3, "color": "#dd2a7b"}},
             "animationDuration": 1400, "animationEasingUpdate": "quinticInOut"},
            {"type": "lines", "coordinateSystem": "cartesian2d", "data": lines, "polyline": False, "silent": True, "zlevel": 2,
             "lineStyle": {"width": 0, "curveness": 0.12},
             "effect": {"show": True, "period": 3.2, "trailLength": 0.35, "symbol": "circle", "symbolSize": 4,
                        "color": "#f58529"}},
        ],
    }, height)
