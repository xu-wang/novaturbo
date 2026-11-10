#!/usr/bin/env python3
"""Exercise the editable 22D V3 single-design form in Firefox Marionette."""
import argparse
import json
import time

from validate_geometry22_results_browser import Browser, mode


STATE = """return {
  mode: document.getElementById('geometry19-model-mode').value,
  count: document.querySelectorAll('#v3-direct-inputs input').length,
  output: !document.getElementById('v3-direct-output').hidden,
  message: document.getElementById('v3-result-status').textContent,
  meta: document.getElementById('v3-direct-meta').textContent,
  metrics: document.getElementById('v3-direct-metrics').textContent,
  physics: document.getElementById('v3-direct-physics').textContent,
  constraints: document.querySelectorAll('#v3-direct-constraints tr').length,
  thrust: document.getElementById('g19-thrust').textContent,
  tsfc: document.getElementById('g19-tsfc').textContent,
  mass: document.getElementById('g19-mass').textContent,
  solver: document.getElementById('g19-solver').textContent,
  feasibility: document.getElementById('g19-constraints').textContent,
  mesh: document.getElementById('g19-mesh').textContent,
  engineInfo: document.getElementById('engine-info').textContent,
  provenance: document.getElementById('geometry19-provenance').textContent,
  schemaMessage: document.getElementById('geometry19-message').textContent,
  buttonDisabled: document.getElementById('v3-direct-run').disabled,
  clearance: document.querySelector('#v3-direct-inputs input[data-variable=compressor_tip_clearance_ratio]')?.value,
  panel: !document.getElementById('geometry22-panel').hidden
};"""


def wait(browser, predicate, label, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = browser.script(STATE)
        if predicate(state):
            return state
        time.sleep(.25)
    raise AssertionError(f'{label} timed out: {state}')


def run(url, port):
    browser = Browser(port)
    checks = []
    try:
        browser.call('WebDriver:NewSession', {'capabilities': {'alwaysMatch': {}}})
        browser.call('WebDriver:Navigate', {'url': url})
        browser.script("document.getElementById('btn-dashboard').click();")
        wait(browser, lambda s: s['schemaMessage'].startswith('Ready'), 'Geometry19 dashboard initialized')
        mode(browser, 'v3_manual')
        state = wait(browser, lambda s: s['count'] == 22 and s['panel'], '22D form')
        assert browser.script("const names=[...document.querySelectorAll('#v3-direct-inputs input')].map(x=>x.dataset.variable);return ['diffuser_area_ratio','combustor_primary_air_fraction','compressor_tip_clearance_ratio'].every(n=>names.includes(n));"), state
        checks.append('All 22 schema fields render as editable inputs')
        browser.script("document.getElementById('v3-direct-run').click();")
        state = wait(browser, lambda s: s['output'] and s['constraints'] > 0, 'reference V3 evaluation')
        assert 'Thrust' in state['metrics'] and 'Converged' in state['metrics'] and state['thrust'] != '—', state
        assert 'Compressor total efficiency' in state['physics'] and 'geometry22_map_enhanced_v3_development_v1' in state['meta'], state
        old_meta = state['meta']
        old_efficiency = browser.script("return Number([...document.querySelectorAll('#v3-direct-physics tr')].find(r=>r.textContent.includes('Compressor total efficiency')).cells[1].textContent);")
        wait(browser, lambda s: not s['buttonDisabled'], 'reference evaluation settled')
        checks.append('Reference 22D design returns solver, three performances, V3 metrics and constraints')
        browser.script("const x=document.querySelector('#v3-direct-inputs input[data-variable=compressor_tip_clearance_ratio]');x.value='0.004';x.dispatchEvent(new Event('input',{bubbles:true}));")
        state = browser.script(STATE)
        assert not state['output'] and not state['meta'] and not state['metrics'] and not state['physics'], state
        assert all(state[name] == '—' for name in ('thrust', 'tsfc', 'mass', 'solver', 'feasibility')), state
        assert state['mesh'] == 'Not generated' and not state['provenance'] and old_meta not in state['engineInfo'], state
        assert state['clearance'] == '0.004', state
        checks.append('Editing a coordinate clears all old physics and mesh status before reevaluation')
        browser.script("document.getElementById('v3-direct-run').click();")
        state = wait(browser, lambda s: s['output'] and s['meta'] != old_meta, 'edited V3 evaluation')
        new_efficiency = browser.script("return Number([...document.querySelectorAll('#v3-direct-physics tr')].find(r=>r.textContent.includes('Compressor total efficiency')).cells[1].textContent);")
        assert new_efficiency < old_efficiency, (old_efficiency, new_efficiency)
        checks.append('Editing tip clearance changes the 22D identity and physical efficiency')
        browser.script("document.getElementById('v3-direct-reset').click();")
        state = browser.script(STATE)
        assert not state['output'] and not state['meta'] and not state['metrics'] and not state['physics'], state
        assert all(state[name] == '—' for name in ('thrust', 'tsfc', 'mass', 'solver', 'feasibility')), state
        assert state['mesh'] == 'Not generated' and not state['provenance'], state
        assert browser.script("return Number(document.querySelector('#v3-direct-inputs input[data-variable=compressor_tip_clearance_ratio]').value);") != .004
        checks.append('Reset clears the edited result and restores reference inputs')
        browser.script("const x=document.querySelector('#v3-direct-inputs input[data-variable=compressor_tip_clearance_ratio]');x.value='0.01';x.dispatchEvent(new Event('input',{bubbles:true}));document.getElementById('v3-direct-run').click();")
        state = wait(browser, lambda s: 'within' in s['message'], 'range rejection')
        assert not state['output'], state
        checks.append('Out-of-range coordinate is rejected before physical evaluation')
        mode(browser, 'v2_1_interactive')
        state = wait(browser, lambda s: not s['panel'] and s['thrust'] == '—', 'V2.1 restore')
        checks.append('Switching models clears the V3 result and restores V2.1')
        mode(browser, 'v3_results')
        wait(browser, lambda s: browser.script("return document.getElementById('v3-result-picker').options.length > 1;"), 'archived V3 list')
        browser.script("const p=document.getElementById('v3-result-picker');p.value=[...p.options].find(o=>o.value&&!o.disabled).value;p.dispatchEvent(new Event('change'));document.getElementById('v3-load').click();")
        wait(browser, lambda s: browser.script("return !document.getElementById('v3-edit-selected').hidden && document.querySelectorAll('#v3-variables tr').length===22;"), 'archived 22D result')
        recorded = browser.script("return Number(document.querySelector('#v3-variables tr td:nth-child(2)').textContent);")
        browser.script("document.getElementById('v3-edit-selected').click();")
        state = wait(browser, lambda s: s['mode'] == 'v3_manual' and s['count'] == 22, 'load result into editor')
        input_value = browser.script("return Number(document.querySelector('#v3-direct-inputs input').value);")
        assert abs(input_value - recorded) <= max(1e-6, abs(input_value) * 1e-5), (input_value, recorded)
        checks.append('A verified V3 result can seed the editable 22D form')
        return {'status': 'PASS', 'checks': checks}
    finally:
        try:
            browser.call('WebDriver:DeleteSession')
        finally:
            browser.socket.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:5097/')
    parser.add_argument('--port', type=int, default=2837)
    args = parser.parse_args()
    print(json.dumps(run(args.url, args.port), indent=2))
