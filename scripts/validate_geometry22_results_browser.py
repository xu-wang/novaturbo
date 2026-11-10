#!/usr/bin/env python3
"""Validate read-only V3 UI in an isolated Firefox Marionette session."""
import argparse
import base64
import json
from pathlib import Path
import socket
import time


class Browser:
    def __init__(self, port):
        self.socket = socket.create_connection(('127.0.0.1', port), 10)
        self.socket.settimeout(40)
        self.number = 0
        self.read()

    def read(self):
        length = bytearray()
        while (part := self.socket.recv(1)) != b':':
            if not part:
                raise ConnectionError('Firefox connection closed')
            length.extend(part)
        size = int(length)
        payload = bytearray()
        while len(payload) < size:
            part = self.socket.recv(size - len(payload))
            if not part:
                raise ConnectionError('Firefox frame truncated')
            payload.extend(part)
        return json.loads(payload)

    def call(self, name, params=None):
        self.number += 1
        data = json.dumps([0, self.number, name, params or {}]).encode()
        self.socket.sendall(str(len(data)).encode() + b':' + data)
        kind, number, error, result = self.read()
        if kind != 1 or number != self.number or error:
            raise RuntimeError(f'{name}: {error or result}')
        return result.get('value')

    def script(self, code):
        return self.call('WebDriver:ExecuteScript', {'script': code, 'args': []})


STATE = """return {
 simulationsDisabled: ['thermal','flow','stress','flame'].every(n=>document.getElementById('btn-sim-'+n).disabled),
 engineInfo: document.getElementById('engine-info').textContent,
 mode: document.getElementById('geometry19-model-mode').value,
 panelVisible: !document.getElementById('geometry22-panel').hidden,
 options: document.getElementById('v3-result-picker').options.length,
 loadDisabled: document.getElementById('v3-load').disabled,
 previewDisabled: document.getElementById('v3-preview').disabled,
 solveDisabled: document.getElementById('geometry19-solve').disabled,
 qualityDisabled: document.getElementById('geometry19-quality').disabled,
 message: document.getElementById('v3-result-status').textContent,
 schemaMessage: document.getElementById('geometry19-message').textContent,
 thrust: document.getElementById('g19-thrust').textContent,
 variableRows: document.querySelectorAll('#v3-variables tr').length,
 physicalOnlyRows: [...document.querySelectorAll('#v3-variables tr')].filter(r => r.textContent.includes('no STL change')).length,
 components: document.querySelectorAll('#component-list .comp-item').length,
 progress: document.getElementById('load-progress').textContent,
 vertices: document.getElementById('stat-verts').textContent,
 viewerStatus: document.getElementById('status-left').textContent
};"""


def wait(browser, predicate, description, timeout=90):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = browser.script(STATE)
        if predicate(last):
            return last
        time.sleep(.5)
    raise AssertionError(f'Timed out: {description}; {last}')


def mode(browser, name):
    browser.script(f"const s=document.getElementById('geometry19-model-mode');s.value={json.dumps(name)};s.dispatchEvent(new Event('change',{{bubbles:true}}));")


def run(url, port, output):
    output.mkdir(parents=True, exist_ok=True)
    browser = Browser(port)
    checks = []
    try:
        browser.call('WebDriver:NewSession', {'capabilities': {'alwaysMatch': {}}})
        browser.call('WebDriver:SetWindowRect', {'width': 1600, 'height': 1100})
        browser.call('WebDriver:Navigate', {'url': url})
        browser.script("document.getElementById('btn-dashboard').click();")
        wait(browser, lambda s: s['schemaMessage'].startswith('Ready'), 'schema initialization')
        mode(browser, 'v3_results')
        state = wait(browser, lambda s: s['panelVisible'] and s['options'] > 1, 'V3 result listing')
        assert state['loadDisabled'] and state['previewDisabled'] and state['solveDisabled'] and state['qualityDisabled'], state
        assert state['components'] == 0 and state['thrust'] == '—' and state['simulationsDisabled'], state
        checks.append('V3 switch clears prior physics/mesh and lists results without starting a solver')
        browser.script("const p=document.getElementById('v3-result-picker');p.value=[...p.options].find(o=>o.value&&!o.disabled).value;p.dispatchEvent(new Event('change'));document.getElementById('v3-load').click();")
        state = wait(browser, lambda s: s['variableRows'] == 22, 'verified 22D result')
        assert state['physicalOnlyRows'] == 3 and not state['previewDisabled'], state
        assert 'not satisfied' in state['message'] and 'NO_HIT' in state['message'], state
        assert state['thrust'] != '—' and state['components'] == 0 and 'V3 recorded F result' in state['engineInfo'], state
        checks.append('NO_HIT design shows F metrics, all 22 variables and three non-mesh physical parameters')
        browser.script("document.getElementById('v3-preview').click();")
        state = wait(browser, lambda s: s['components'] > 0 and s['vertices'] != '0' and 'V3 · 19D projection' in s['viewerStatus'], 'real STL preview', 120)
        assert state['components'] == 10 and state['progress'].startswith('12 / 12'), state
        checks.append('Twelve verified STL meshes load, including two toggleable lattice meshes, under V3 projection identity')
        loaded_state = state
        (output / 'v3_dashboard.png').write_bytes(base64.b64decode(browser.call('WebDriver:TakeScreenshot', {})))
        browser.script("document.getElementById('dash-close').click();")
        time.sleep(.8)
        (output / 'v3_projection.png').write_bytes(base64.b64decode(browser.call('WebDriver:TakeScreenshot', {})))
        browser.script("document.getElementById('btn-dashboard').click();")
        mode(browser, 'v2_1_interactive')
        state = wait(browser, lambda s: not s['panelVisible'] and not s['qualityDisabled'], 'V2.1 mode restored')
        assert not state['simulationsDisabled'] and not state['solveDisabled'] and state['components'] == 0 and state['thrust'] == '—', state
        checks.append('Switch back restores V2.1 controls and clears the V3 mesh and metrics')
        mode(browser, 'v3_results')
        wait(browser, lambda s: s['panelVisible'] and s['options'] > 1, 'V3 listing for delayed response')
        browser.script("""window.__v3OriginalFetch=window.fetch;
window.fetch=(url, options)=>/^\/api\/geometry22\/results\/[a-f0-9]{64}$/.test(url)
 ? new Promise(resolve=>{window.__v3Resolve=resolve;}) : window.__v3OriginalFetch(url,options);
const p=document.getElementById('v3-result-picker');p.value=[...p.options].find(o=>o.value&&!o.disabled).value;
p.dispatchEvent(new Event('change'));document.getElementById('v3-load').click();""")
        mode(browser, 'v2_2_new')
        browser.script("window.__v3Resolve({ok:true,json:async()=>({})});window.fetch=window.__v3OriginalFetch;")
        state = wait(browser, lambda s: not s['panelVisible'], 'V2.2 switch')
        assert state['components'] == 0 and state['thrust'] == '—' and state['solveDisabled'], state
        checks.append('Late V3 result cannot overwrite V2.2 metrics or re-enable legacy solving')
        browser.call('WebDriver:Navigate', {'url': url})
        browser.script("document.getElementById('btn-dashboard').click();")
        wait(browser, lambda s: s['schemaMessage'].startswith('Ready'), 'schema after reload')
        mode(browser, 'v3_results')
        state = wait(browser, lambda s: s['options'] > 1 and s['panelVisible'], 'V3 listing after reload')
        assert state['components'] == 0 and state['thrust'] == '—' and state['variableRows'] == 0, state
        checks.append('Reload offers verified results but does not automatically show an old design')
        report = {'status': 'v3_read_only_results_browser_acceptance_complete', 'checks': checks, 'loaded_state': loaded_state,
                  'no_training_or_inverse_design_launched': True, 'mesh_scope': '19d_geometry_projection_only'}
        (output / 'browser_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        return report
    finally:
        try:
            browser.call('WebDriver:DeleteSession')
        finally:
            browser.socket.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:5073/')
    parser.add_argument('--port', type=int, default=2833)
    parser.add_argument('--output', type=Path, default=Path('data/validation/v3_readonly_visualization_v1'))
    args = parser.parse_args()
    print(json.dumps(run(args.url, args.port, args.output), ensure_ascii=False))
