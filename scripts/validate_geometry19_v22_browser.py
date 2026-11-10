#!/usr/bin/env python3
"""Exercise the V2.2 Viewer workflow in a Firefox Marionette session.

Start the UI server and Firefox with Marionette before running this script.
The browser uses the real GPU pipeline through the UI, so allow several minutes.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import socket
import time


class Marionette:
    def __init__(self, host: str, port: int):
        self.socket = socket.create_connection((host, port), 10)
        self.number = 0
        greeting = self._read()
        if greeting.get('marionetteProtocol') != 3:
            raise RuntimeError(f'Unexpected Firefox greeting: {greeting}')

    def _read(self):
        length = bytearray()
        while (part := self.socket.recv(1)) != b':':
            if not part:
                raise ConnectionError('Firefox Marionette closed the connection')
            length.extend(part)
        expected = int(length)
        payload = bytearray()
        while len(payload) < expected:
            part = self.socket.recv(expected - len(payload))
            if not part:
                raise ConnectionError('Firefox Marionette closed the frame')
            payload.extend(part)
        return json.loads(payload)

    def call(self, name: str, params: dict | None = None):
        self.number += 1
        body = json.dumps([0, self.number, name, params or {}]).encode()
        self.socket.sendall(str(len(body)).encode() + b':' + body)
        kind, number, error, result = self._read()
        if kind != 1 or number != self.number or error:
            raise RuntimeError(f'{name}: {error or result}')
        return result.get('value')

    def script(self, source: str):
        return self.call('WebDriver:ExecuteScript', {'script': source, 'args': []})

    def close(self):
        try:
            self.call('WebDriver:DeleteSession')
        finally:
            self.socket.close()


STATE = """return {
  mode: document.getElementById('geometry19-model-mode').value,
  subtitle: document.getElementById('geometry19-dashboard-subtitle').textContent,
  message: document.getElementById('geometry19-message').textContent,
  targetsVisible: getComputedStyle(document.getElementById('geometry19-v22-targets')).display !== 'none',
  legacyVisible: getComputedStyle(document.querySelector('.kpi-row')).display !== 'none',
  groupsVisible: getComputedStyle(document.getElementById('geometry19-groups')).display !== 'none',
  jsonVisible: getComputedStyle(document.querySelector('.geometry19-json-panel')).display !== 'none',
  resultsBeforeControls: !!(document.querySelector('.geometry19-results').compareDocumentPosition(
    document.getElementById('geometry19-groups')) & Node.DOCUMENT_POSITION_FOLLOWING),
  solveDisabled: document.getElementById('geometry19-solve').disabled,
  generateDisabled: document.getElementById('geometry19-v22-generate').disabled,
  thrust: document.getElementById('g19-thrust').textContent,
  tsfc: document.getElementById('g19-tsfc').textContent,
  mass: document.getElementById('g19-mass').textContent,
  verts: document.getElementById('stat-verts').textContent,
  componentCount: document.querySelectorAll('#component-list .comp-item').length,
  viewerStatus: document.getElementById('status-left').textContent
};"""


def wait_for(browser: Marionette, predicate, description: str, timeout=180):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = browser.script(STATE)
        if predicate(last):
            return last
        time.sleep(1)
    raise AssertionError(f'Timed out waiting for {description}: {last}')


def select_mode(browser: Marionette, mode: str):
    browser.script("""const input = document.getElementById('geometry19-model-mode');
input.value = '%s'; input.dispatchEvent(new Event('change', {bubbles: true}));""" % mode)


def assert_blank(state):
    assert (state['thrust'], state['tsfc'], state['mass']) == ('—', '—', '—'), state
    assert state['componentCount'] == 0 and state['verts'] == '0', state
    if state['mode'] == 'v2_2_new':
        assert not state['groupsVisible'] and not state['jsonVisible'], state


def run(url: str, screenshot: Path | None, marionette_port: int):
    browser = Marionette('127.0.0.1', marionette_port)
    checks = []
    try:
        browser.call('WebDriver:NewSession', {'capabilities': {'alwaysMatch': {}}})
        browser.call('WebDriver:Navigate', {'url': url})
        wait_for(browser, lambda s: s['componentCount'] > 0,
                 'the original Viewer model to load', 45)
        browser.script("document.getElementById('btn-dashboard').click();")
        state = wait_for(browser, lambda s: s['message'].startswith('就绪 ·'),
                         'Geometry19 schema to load', 45)
        assert not state['targetsVisible'], state
        assert state['legacyVisible'], state
        checks.append('Viewer and Geometry19 dashboard opened')

        select_mode(browser, 'v2_2_new')
        state = browser.script(STATE)
        assert_blank(state)
        assert state['targetsVisible'] and state['solveDisabled'] and not state['generateDisabled'], state
        assert 'V2.2 开发版物理计算' in state['subtitle'], state
        assert not state['legacyVisible'], state
        checks.append('V2.2 switch cleared old metrics and 3D components')

        browser.script("document.getElementById('geometry19-v22-generate').click();")
        state = browser.script(STATE)
        assert '有限正数' in state['message'], state
        assert_blank(state)
        checks.append('Empty targets were rejected without showing a result')

        browser.script("""document.getElementById('g19-target-thrust').value = '60.188872113156265';
document.getElementById('g19-target-tsfc').value = '0.00006720163166164489';
document.getElementById('g19-target-mass').value = '1.7666065876437729';
document.getElementById('geometry19-v22-generate').click();""")
        state = wait_for(browser, lambda s: s['generateDisabled'], 'V2.2 job to start', 20)
        assert_blank(state)
        checks.append('Generation began with the display blank')

        state = wait_for(browser, lambda s: s['message'].startswith('新的 V2.2 设计'),
                         'new V2.2 design to finish', 240)
        assert state['thrust'] != '—' and state['tsfc'] != '—' and state['mass'] != '—', state
        state = wait_for(browser, lambda s: s['componentCount'] > 0 and s['verts'] != '0',
                         'new STL model to load', 90)
        assert 'Geometry19 ' in state['viewerStatus'] and '已加载' in state['viewerStatus'], state
        assert state['groupsVisible'] and state['jsonVisible'], state
        assert state['resultsBeforeControls'], state
        checks.append('New F-verified metrics and STL components appeared')
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            screenshot.write_bytes(base64.b64decode(browser.call('WebDriver:TakeScreenshot', {})))

        select_mode(browser, 'v2_1_interactive')
        assert_blank(browser.script(STATE))
        browser.script("""window.__geometry19TestFetch = window.fetch;
window.fetch = (url, options) => url === '/api/geometry19/solve'
  ? new Promise(resolve => { window.__geometry19ResolveSolve = resolve; })
  : window.__geometry19TestFetch(url, options);
document.getElementById('geometry19-solve').click();""")
        wait_for(browser, lambda s: s['solveDisabled'], 'V2.1 request to start', 10)
        select_mode(browser, 'v2_2_new')
        browser.script("""window.__geometry19ResolveSolve({ok: true, json: async () => ({})});
window.fetch = window.__geometry19TestFetch;""")
        time.sleep(1)
        state = browser.script(STATE)
        assert_blank(state)
        assert state['solveDisabled'], state
        checks.append('Late V2.1 response did not repopulate V2.2 or enable V2.1 controls')

        browser.call('WebDriver:Navigate', {'url': url})
        browser.script("document.getElementById('btn-dashboard').click();")
        wait_for(browser, lambda s: s['message'].startswith('就绪 ·'),
                 'dashboard after reload', 45)
        select_mode(browser, 'v2_2_new')
        time.sleep(2)
        assert_blank(browser.script(STATE))
        checks.append('Reload and mode switch did not load a historical result')
        return checks
    finally:
        browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:5000/')
    parser.add_argument('--marionette-port', type=int, default=2828)
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    for check in run(args.url, args.screenshot, args.marionette_port):
        print('PASS:', check)
