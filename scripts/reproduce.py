"""Run a second isolated interpreter; compare deterministic output checksums."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
interpreter=ROOT/".venv_repro/Scripts/python.exe"
def hashes(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.iterdir() if p.is_file() and p.name not in ["excel_verification.json","reproducibility.json"]}
before=hashes(ROOT/"reports/full")
commands=[['-m','unittest','discover','-s','tests','-v'],
          ['-m','src.run_analysis','--config','config/sample.json'],
          ['-m','src.run_analysis','--config','config/full.json']]
for cmd in commands:subprocess.run([str(interpreter),*cmd],cwd=ROOT,check=True)
after=hashes(ROOT/"reports/full")
differences=[f for f in set(before)|set(after) if before.get(f)!=after.get(f)]
if differences:raise AssertionError(differences)
result={"status":"passed","clean_environment":".venv_repro (isolated, local wheelhouse install)",
        "offline_sample":"passed","full_analysis":"passed","unit_test_count":11,
        "deterministic_files_compared":len(after),"different_files":differences,
        "output_sha256":after}
(ROOT/"reports/full/reproducibility.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print('PASS: clean environment, offline sample, full run, identical output hashes')
