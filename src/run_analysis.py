import argparse
import hashlib
import json
import platform
from pathlib import Path
import pandas as pd
from .quality import validate
from .portfolio import simulate, metrics, monthly_returns, linked_contributions
from .verification import verify, SQL
from .presentation import charts, findings, html_dashboard, pdf_report

ROOT = Path(__file__).resolve().parents[1]

def run(config):
    path = ROOT / config
    cfg = json.loads(path.read_text(encoding="utf-8"))
    out = ROOT / cfg["output"]
    out.mkdir(parents=True,exist_ok=True)
    pricefile = ROOT/cfg["prices"]
    frame = pd.read_csv(pricefile)
    calendar = pd.read_csv(ROOT/cfg["calendar"]).date
    prices, issues = validate(frame,calendar,list(cfg["securities"]),cfg["anchor"],cfg["end"])
    issues.to_csv(out/"input_quality_issues.csv",index=False)
    if prices is None: raise ValueError("Input quality failed; see input_quality_issues.csv")
    if len(prices) < 3: raise ValueError("Insufficient observations")
    if (prices.index[1] < pd.Timestamp(cfg["start"])) or (prices.index[-1] > pd.Timestamp(cfg["end"])):
        raise ValueError("Price range incompatible with analysis period")
    results, sensitivity, all_results = {}, [], {}
    for bps in cfg["cost_bps"]:
        runs = {name:simulate(prices,w,bps) for name,w in cfg["portfolios"].items()}
        all_results[bps] = runs
        for name,result in runs.items():
            met = metrics(result["daily"],runs["Benchmark"]["daily"],cfg["annual_days"],cfg["risk_free_annual"])
            sensitivity.append({"portfolio":name,"cost_bps":bps,**met})
            result["daily"].rename_axis("date").to_csv(out/f"daily_{name}_{bps}bps.csv",float_format="%.12g")
        if bps == 0: results = runs
    if not results: raise ValueError("Cost scenarios must include zero")
    for name,result in results.items():
        result["anchor"] = prices.index[0]
        for part in ["weights_start","weights_end","contributions"]:
            result[part].rename_axis("date").to_csv(out/f"{part}_{name}.csv",float_format="%.12g")
        result["rebalances"].to_csv(out/f"rebalances_{name}.csv",index=False,float_format="%.12g")
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(out/"cost_sensitivity.csv",index=False,float_format="%.12g",na_rep="n.a.")
    metric = sensitivity[sensitivity.cost_bps.eq(0)].set_index("portfolio").drop(columns="cost_bps")
    metric.to_csv(out/"metrics.csv",float_format="%.12g",na_rep="n.a.")
    monthly = pd.DataFrame({name:monthly_returns(res["daily"]) for name,res in results.items()})
    monthly.rename_axis("month").to_csv(out/"monthly_returns.csv",float_format="%.12g")
    annual = pd.DataFrame({name:(1+res["daily"].net_return).groupby(res["daily"].index.year).prod()-1 for name,res in results.items()})
    annual.rename_axis("year").to_csv(out/"annual_returns.csv",float_format="%.12g")
    linked = linked_contributions(results["Main"])
    if abs(linked.sum()-metric.loc["Main","total_return"]) > cfg["tolerance"]: raise AssertionError("Linked contribution mismatch")
    linked.rename("wealth_contribution").rename_axis("ticker").to_csv(out/"linked_contributions.csv",float_format="%.12g")
    bench_linked = linked_contributions(results["Benchmark"])
    active_linked = linked-bench_linked
    if abs(active_linked.sum()-metric.loc["Main","cumulative_excess_pp"]) > cfg["tolerance"]:
        raise AssertionError("Active wealth reconciliation mismatch")
    pd.DataFrame({"main":linked,"benchmark":bench_linked,"difference":active_linked}).rename_axis("ticker").to_csv(out/"active_wealth_reconciliation.csv",float_format="%.12g")
    checks = verify(prices,results,cfg,all_results)
    (out/"verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    manifestfile = pricefile.parent/"manifest.json"
    metadata = json.loads(manifestfile.read_text(encoding="utf-8"))
    for name in ["prices.csv","calendar.csv"]:
        if hashlib.sha256((pricefile.parent/name).read_bytes()).hexdigest()!=metadata[name+"_sha256"]:
            raise ValueError("Snapshot checksum mismatch: "+name)
    provenance = {"python":platform.python_version(), "config_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                  "prices_sha256":hashlib.sha256(pricefile.read_bytes()).hexdigest(),
                  "calendar_sha256":hashlib.sha256((ROOT/cfg["calendar"]).read_bytes()).hexdigest(),
                  "observations":len(prices)-1,"anchor":cfg["anchor"],"start":cfg["start"],"end":cfg["end"],
                  "random_seed":"No randomized computation", "quality_errors":int(issues.severity.eq("error").sum()),
                  "quality_warnings":int(issues.severity.eq("warning").sum()),"research_type":"Retrospective simulation"}
    (out/"run_manifest.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    charts(out,results,monthly,linked)
    notes = findings(metric,annual,linked,sensitivity,active_linked)
    (out/"findings.json").write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding="utf-8")
    html_dashboard(out,cfg,metric,annual,monthly,notes)
    pdf_report(out,cfg,metric,annual,notes,sensitivity,metadata)
    print(metric.to_string())
    print(f"PASS: {len(prices)-1} sessions; outputs: {out}")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="config/full.json");a=p.parse_args();run(a.config)
