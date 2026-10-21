"""Static charts, self-contained HTML and a three-page Chinese research report."""
import base64
import html
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

COLORS = {"Main": "#156f80", "Benchmark": "#e68a37", "Equal": "#6e70a4", "Conservative": "#839642"}

def charts(out, results, monthly, linked):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.facecolor": "white",
                         "svg.hashsalt": "portfolio-project-v1"})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name + ".png"), dpi=160, bbox_inches="tight")
        fig.savefig(out / (name + ".svg"), bbox_inches="tight", metadata={"Date":None})
        plt.close(fig)
    for column, name, title in [("nav", "wealth", "Growth of $1 | monthly rebalancing | zero cost"),
                                 ("drawdown", "drawdown", "Drawdown from prior peak | zero cost")]:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        for portfolio, result in results.items():
            d = result["daily"]
            values = d[column]
            # Start chart at initial capital / zero drawdown anchor.
            start = result["anchor"]
            values = pd.concat([pd.Series([1.0 if column == "nav" else 0.0], index=[start]), values])
            ax.plot(values.index, values, label=portfolio, color=COLORS[portfolio], linewidth=1.4)
        ax.set_title(title, loc="left", weight="bold")
        if column == "drawdown": ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(ncol=4, frameon=False, fontsize=9)
        save(fig, name)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    weights = results["Main"]["weights_start"]
    ax.stackplot(weights.index, *[weights[t] for t in weights], labels=weights.columns,
                 colors=plt.get_cmap("tab20")(np.arange(len(weights.columns))))
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_ylim(0, 1)
    ax.set_title("Main portfolio | beginning-of-session weights", loc="left", weight="bold")
    ax.legend(ncol=8, loc="upper center", bbox_to_anchor=(.5, -.15), frameon=False, fontsize=8)
    save(fig, "weights")
    main = monthly["Main"]
    mat = pd.DataFrame({"year": main.index.year, "month": main.index.month, "value": main.values}).pivot(index="year", columns="month", values="value")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    bound = max(.01, np.nanmax(np.abs(mat.to_numpy())))
    ax.imshow(mat, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(mat.columns)), [str(m) for m in mat.columns])
    ax.set_yticks(range(len(mat.index)), [str(y) for y in mat.index]); ax.grid(False)
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if np.isfinite(mat.iloc[y, x]): ax.text(x, y, f"{mat.iloc[y,x]:.1%}", ha="center", va="center", fontsize=8)
    ax.set_title("Main portfolio | compounded monthly returns", loc="left", weight="bold")
    save(fig, "monthly")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = linked.sort_values()
    ax.barh(x.index, x.values, color=["#156f80" if v >= 0 else "#b45d59" for v in x])
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.axvline(0, color="#555", linewidth=.6)
    ax.set_title("Main portfolio | linked cumulative wealth contributions", loc="left", weight="bold")
    ax.set_xlabel("Percentage points of total return; not benchmark attribution")
    save(fig, "contributions")

def findings(metrics, annual, linked, sensitivity, active_linked):
    m, b = metrics.loc["Main"], metrics.loc["Benchmark"]
    worst_year = annual.Main.idxmin()
    top = linked.idxmax()
    cost10 = sensitivity.loc[(sensitivity.portfolio == "Main") & (sensitivity.cost_bps == 10)].iloc[0]
    weakest = active_linked.idxmin()
    return [
        f"主组合累计收益{m.total_return:.2%}，基准{b.total_return:.2%}，差额{m.cumulative_excess_pp*100:.2f}个百分点；相对财富收益{m.relative_wealth_return:.2%}。",
        f"主组合年化波动率{m.annual_volatility:.2%}，基准{b.annual_volatility:.2%}；最大回撤分别为{m.max_drawdown:.2%}与{b.max_drawdown:.2%}。是否降低风险须同时查看两项指标。",
        f"主组合最差日历年为{worst_year}，收益{annual.loc[worst_year,'Main']:.2%}，同期基准{annual.loc[worst_year,'Benchmark']:.2%}。这说明跨资产配置仍可能在同一年受损。",
        f"{top}的累计财富贡献最大，为{linked[top]*100:.2f}个百分点。该数值采用逐日净值链接，各资产贡献之和等于总收益；不代表相对基准的独立归因。",
        f"10个基点成本情景下，主组合累计收益{cost10.total_return:.2%}，较零成本降低{(m.total_return-cost10.total_return)*100:.2f}个百分点；这里按买入加卖出金额比例收费，包含初始建仓。",
        f"累计收益差额的财富贡献对账中，{weakest}的主组合贡献减基准贡献为{active_linked[weakest]*100:.2f}个百分点，是最大负差额项。这是两个组合贡献的算术对账，不是Brinson归因或因果证明。"
    ]

def html_dashboard(out, cfg, metrics, annual, monthly, notes):
    def img(name):
        raw = base64.b64encode((out / (name + ".png")).read_bytes()).decode()
        return f'<img alt="{name}" src="data:image/png;base64,{raw}">'
    table = metrics[["total_return", "annual_return", "annual_volatility", "max_drawdown", "sharpe", "tracking_error", "information_ratio"]].copy()
    for col in table:
        table[col] = table[col].map(lambda v: "n.a." if pd.isna(v) else (f"{v:.2f}" if col in ["sharpe", "information_ratio"] else f"{v:.2%}"))
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ETF投资组合分析</title>
<style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#eef3f5;color:#213441;margin:0}}main{{max-width:1050px;margin:auto;padding:32px}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}h1{{font-size:32px}}h2{{font-size:21px}}p,li{{line-height:1.8}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #ddd}}th:first-child{{text-align:left}}.scroll{{overflow:auto}}small{{color:#546a76}}</style>
<main><h1>ETF投资组合绩效与风险分析</h1><p>{cfg['start']} 至 {cfg['end']} · USD · 月度再平衡 · Independent Project</p>
<section><h2>研究设计</h2><p>检验预设60%股票／40%债券配置，相对60% VTI／40% BND基准的收益与风险。等权组合为50%股票／50%债券，保守组合为40%股票／60%债券；不同风险预算会影响比较。</p><p>调整收盘价作为总回报代理；无风险利率为零的简化假设。以下主图和指标均为零成本，成本情景见报告及CSV。配置在首次取数前确定，但研究本身为回顾性分析，未证明未来有效。</p></section>
<section><h2>实际发现</h2><ol>{''.join('<li>'+html.escape(n)+'</li>' for n in notes)}</ol></section>
<section><h2>指标对比</h2><div class="scroll">{table.to_html(border=0)}</div><small>年化采用252个交易日；标准差ddof=1；零分母为n.a.。</small></section>
{''.join('<section>'+img(name)+'</section>' for name in ['wealth','drawdown','monthly','weights','contributions'])}
<section><h2>月度收益表 · 所有配置</h2><div class="scroll">{monthly.map(lambda v:f'{v:.2%}').to_html(border=0)}</div></section>
<section><h2>局限</h2><p>调整价是供应商回溯更新的总回报代理；未独立重建分红与拆股。美元计价不等于没有海外资产汇率风险。ETF覆盖存在重叠，VB是对VTI小盘暴露的倾斜。未模拟税收、滑点和成交限制，成本是情景假设。基准选择及样本区间影响结果，回顾性结果不代表样本外表现。</p><p>来源：<a href="https://finance.yahoo.com/">Yahoo Finance</a>；原始响应、下载时点及SHA-256见data/snapshot/manifest.json。数据许可未转让，当前文件供本地个人研究。</p></section></main></html>'''
    (out / "dashboard.html").write_text(body, encoding="utf-8")

def pdf_report(out, cfg, metrics, annual, notes, sensitivity, metadata):
    fontfile = Path("C:/Windows/Fonts/simsun.ttc")
    if fontfile.exists():
        pdfmetrics.registerFont(TTFont("ChineseEmbedded",str(fontfile),subfontIndex=0))
        font = "ChineseEmbedded"
    else:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        font = "STSong-Light"
    c = canvas.Canvas(str(out / "portfolio_report.pdf"), pagesize=(595.28, 841.89), invariant=1)
    c.setTitle("投资组合绩效与风险分析 | Independent Project")
    style = ParagraphStyle("body", fontName=font, fontSize=9.5, leading=15, textColor=HexColor("#263e4c"), wordWrap="CJK")
    def header(page, subtitle):
        c.setFillColor(HexColor("#156f80")); c.rect(0, 810, 595.28, 32, fill=1, stroke=0)
        c.setFillColor(HexColor("#ffffff")); c.setFont(font, 11); c.drawString(38, 821, "ETF投资组合绩效与风险分析 | Independent Project" + (" | 离线短样例" if cfg.get("sample") else ""))
        c.setFillColor(HexColor("#213441")); c.setFont(font, 19); c.drawString(38, 776, subtitle)
        c.setFont(font, 8); c.drawString(38, 26, f"{cfg['start']} / {cfg['end']} | USD | 月度再平衡 | 第{page}页 / 共3页")
    def para(text, y, size=None):
        s = style if size is None else ParagraphStyle("small", parent=style, fontSize=size, leading=size+5)
        p = Paragraph(text, s); _, h = p.wrap(519, 750); p.drawOn(c, 38, y-h); return y-h-8
    def title(text, y):
        y -= 7
        c.setFont(font, 12); c.setFillColor(HexColor("#156f80")); c.drawString(38, y, text); return y-17
    def image(name, y, height):
        c.drawImage(ImageReader(str(out/(name+".png"))), 38, y-height, width=519, height=height, preserveAspectRatio=True, anchor="c"); return y-height-12
    header(1, "研究设计与核心结果")
    y = para("研究问题：预设的跨地区、跨债券类型ETF组合，相对美国股债基准的表现如何？本研究不挑选历史最优权重，重点是可验证的组合计算与结果解释。", 749)
    y = para(f"数据区间：{cfg['anchor']}为初始价格锚点，收益覆盖{cfg['start']}至{cfg['end']}，共{int(metrics.loc['Main','observations'])}个日收益。来源为Yahoo Finance调整收盘价，包含供应商的分红与拆股调整，作为总回报代理。下载于{metadata['retrieved_utc'][:10]}（UTC）。", y)
    y = title("配置与比较", y)
    y = para("主组合：VTI 30%、VB 10%、VEA 15%、VWO 5%、BND 20%、IEF 10%、TIP 5%、BNDX 5%。股票60%，债券40%。<br/>基准：VTI 60%、BND 40%；等权：八只各12.5%；保守：20%、5%、10%、5%、30%、15%、10%、5%（同上述顺序）。各配置均采用相同日期与月度再平衡。", y)
    y = image("wealth", y, 192)
    y = title("零成本指标", y)
    headers = ["组合", "累计收益", "年化收益", "年化波动", "最大回撤", "Sharpe"]
    xs = [38, 121, 207, 292, 377, 472]
    c.setFont(font, 9)
    for x, text in zip(xs, headers): c.drawString(x, y, text)
    y -= 20
    for name, row in metrics.iterrows():
        vals = [name, f"{row.total_return:.2%}", f"{row.annual_return:.2%}", f"{row.annual_volatility:.2%}", f"{row.max_drawdown:.2%}", f"{row.sharpe:.2f}"]
        for x, text in zip(xs, vals): c.drawString(x, y, text)
        y -= 19
    y = para("年化收益按净值复合增长、252个交易日折算；波动率使用日收益样本标准差乘以√252。Sharpe采用日超额收益，无风险利率设为零（简化假设）。", y-4)
    para(notes[0], y)
    c.showPage()
    header(2, "风险、配置与收益贡献")
    y = image("drawdown", 749, 182)
    y = para(notes[1], y)
    y = para(notes[2], y)
    y = image("contributions", y, 175)
    y = para(notes[3], y)
    y = para(notes[5], y, 9)
    y = title("相对风险与配置解释", y)
    m = metrics.loc["Main"]
    y = para(f"主组合tracking error为{m.tracking_error:.2%}，information ratio为{m.information_ratio:.2f}。Tracking error是组合减基准的日收益差标准差年化；information ratio是平均日主动收益年化除以tracking error。", y)
    y = para("主组合减少美国股票权重，加入国际股票、小盘倾斜与其他债券类型，因而相对基准的差异可能来自这些配置暴露。这里只提供配置比较与绝对收益贡献，没有实现Brinson归因，不能将上述差异解释为已经证实的因果关系。等权和保守配置的股债比例不同，也影响风险比较。", y)
    para("累计财富贡献使用Σ(前一日净值 × 当日资产贡献)，仅应用于零成本模型；加总等于累计总收益。每日贡献使用前一期权重，不直接加总多日贡献百分比。", y)
    c.showPage()
    header(3, "方法、成本与验证")
    y = para("组合模型：r(i,t)=P(i,t)/P(i,t-1)-1；r(p,t)=Σw(i,t-1)r(i,t)。非再平衡日，权重按w(i,t)=w(i,t-1)(1+r(i,t))/(1+r(p,t))漂移。月末收盘形成目标配置，下一交易日收益使用目标权重；不在当日提前使用新权重。", 749)
    y = para("费用假设：费率乘以买入与卖出合计金额占交易前净值的比例Σ|目标权重-漂移权重|，不除以2；初始投资的买入比例为100%。费用在下一日收益之前扣除，净收益=(1-费用比例)(1+毛收益)-1。按比例减小全部资产，不单独建现金或成交数量账本。末日不进行无后续收益的再平衡，也不收取清仓费。", y)
    y = title("成本敏感性：主组合", y)
    c.setFont(font, 9)
    xs = [38, 146, 262, 388]
    for x, text in zip(xs, ["费率（基点）", "累计收益", "年化收益", "最大回撤"]): c.drawString(x, y, text)
    y -= 20
    for _, row in sensitivity[sensitivity.portfolio.eq("Main")].iterrows():
        for x, text in zip(xs, [str(int(row.cost_bps)), f"{row.total_return:.2%}", f"{row.annual_return:.2%}", f"{row.max_drawdown:.2%}"]): c.drawString(x, y, text)
        y -= 19
    y = para(notes[4], y-5)
    y = image("monthly", y, 148)
    y = title("验证与局限", y)
    y = para("验证记录见verification.json：小型组合手算、独立持仓单位模型核对、SQLite LAG价格收益核对、净值与收益及贡献一致性；另有自动化测试覆盖月度再平衡、未来数据扰动、成本和零分母等边界。浮点绝对误差容差为1e-10。Excel核对工作簿以公式引擎重算，未声称在桌面Excel中执行。", y, 9)
    y = para("局限：①调整价可能被供应商回溯修订，尚未独立重建分红与拆股；②不模拟税收、滑点、成交约束，费率是假设；③ETF存在暴露重叠，美元计价仍可能包含海外汇率风险；④配置虽在首次取数前固定，但研究是回顾性的，未验证样本外效果。", y, 9)
    para("来源与复现：Yahoo Finance历史价格；调整定义见in.help.yahoo.com/kb/adjusted-close-sln28256.html。数据仅用于本地个人研究，不主张再分发许可。原始快照、文件校验值、配置和依赖版本随项目保留。离线入口及输出说明见README。", y, 8)
    c.save()
