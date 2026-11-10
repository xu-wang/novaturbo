#!/usr/bin/env python3
"""Real browser submits a previously viewed target through the GPU 6 V3 job API."""
import argparse
import base64
import json
from pathlib import Path
import time
from validate_geometry22_results_browser import Browser, mode, wait, STATE

ROOT = Path(__file__).resolve().parents[1]

def run(url, port, output):
    output.mkdir(parents=True, exist_ok=True)
    browser = Browser(port)
    checks = []
    try:
        browser.call('WebDriver:NewSession', {'capabilities': {'alwaysMatch': {}}})
        browser.call('WebDriver:SetWindowRect', {'width':1600, 'height':1100})
        browser.call('WebDriver:Navigate', {'url':url})
        browser.script("document.getElementById('btn-dashboard').click();")
        wait(browser, lambda s:s['schemaMessage'].startswith('Ready'), 'schema')
        mode(browser, 'v3_new')
        wait(browser, lambda s:'Enter three target' in s['message'], 'V3 new target form')
        assert browser.script("return ['v3-target-thrust','v3-target-tsfc','v3-target-mass'].every(id=>document.getElementById(id).value==='');")
        source = sorted((ROOT / 'data/validation/v3_22d_final_holdout_v1/cases').iterdir())[0]
        target = json.loads((source / 'run/Protocol_Lock.json').read_text())['identity']['target']
        values = {name:target[name] for name in ['thrust_N','tsfc_kg_N_s','mass_kg']}
        browser.script(f"const t={json.dumps(values)};document.getElementById('v3-target-thrust').value=t.thrust_N;document.getElementById('v3-target-tsfc').value=t.tsfc_kg_N_s;document.getElementById('v3-target-mass').value=t.mass_kg;document.getElementById('v3-start').click();")
        wait(browser, lambda s:'V3 job' in s['message'] or s['variableRows']==22, 'job accepted')
        job_id = browser.script("return document.getElementById('v3-job-picker').value;")
        assert len(job_id) == 64
        checks.append('Browser submits three performances to isolated V3 GPU 6 job')
        mode(browser, 'v2_1_interactive')
        state = wait(browser, lambda s:not s['panelVisible'], 'switch while job continues')
        assert state['thrust'] == '—' and not state['simulationsDisabled']
        time.sleep(2.5)
        state = browser.script(STATE)
        assert state['mode']=='v2_1_interactive' and state['thrust']=='—'
        checks.append('Background job polling cannot overwrite V2.1 after mode switch')
        mode(browser, 'v3_new')
        wait(browser, lambda s:'Loading saved' not in s['message'], 'saved jobs')
        browser.script(f"document.getElementById('v3-job-picker').value={json.dumps(job_id)};document.getElementById('v3-job-load').click();")
        state = wait(browser, lambda s:s['variableRows']==22 or 'failed' in s['message'] or 'interrupted' in s['message'], 'completed V3 result', 180)
        assert state['variableRows']==22 and 'NO_HIT' in state['message'] and 'not satisfied' in state['message'], state
        assert state['simulationsDisabled'] and 'V3 recorded F result' in state['engineInfo']
        checks.append('Completed NO_HIT is displayed with full 22D and F metrics, not a worker error')
        assert browser.script("return document.getElementById('v3-preview').getClientRects().length > 0;")
        browser.script("document.getElementById('v3-preview').click();")
        state = wait(browser, lambda s:'preview loaded' in s['message'], 'job projection preview', 180)
        assert state['components']==10 and state['progress'].startswith('12 / 12'), state
        checks.append('New job produces twelve verified projection STL meshes via isolated job file routes')
        (output/'dashboard.png').write_bytes(base64.b64decode(browser.call('WebDriver:TakeScreenshot', {})))
        browser.script("document.getElementById('dash-close').click();")
        time.sleep(.8)
        (output/'projection.png').write_bytes(base64.b64decode(browser.call('WebDriver:TakeScreenshot', {})))
        browser.call('WebDriver:Navigate', {'url':url})
        browser.script("document.getElementById('btn-dashboard').click();")
        wait(browser, lambda s:s['schemaMessage'].startswith('Ready'), 'reload schema')
        mode(browser, 'v3_new')
        wait(browser, lambda s:'Enter three target' in s['message'], 'completed job list')
        browser.script(f"document.getElementById('v3-job-picker').value={json.dumps(job_id)};document.getElementById('v3-job-load').click();")
        state = wait(browser, lambda s:s['variableRows']==22, 'result after reload')
        assert state['components']==0 and state['thrust']!='—'
        checks.append('Reload offers saved completed job and restores verified metrics without launching another run')
        report = {'status':'v3_ui_job_browser_acceptance_complete','checks':checks,'job_id':job_id,'target':values,
                  'source_target_previously_viewed':True,'physical_gpu':6,'loaded_state':state}
        (output/'browser_report.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    finally:
        try: browser.call('WebDriver:DeleteSession')
        finally: browser.socket.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:5073/')
    parser.add_argument('--port',type=int,default=2833)
    parser.add_argument('--output',type=Path,default=ROOT/'data/validation/v3_ui_jobs_acceptance_v1')
    args=parser.parse_args()
    print(json.dumps(run(args.url,args.port,args.output)))
