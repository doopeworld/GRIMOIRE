#!/usr/bin/env python3
"""upload_release.py TAG ASSET COMMIT_SHA NOTES.md [NAME]

Create GitHub release TAG on doopeworld/GRIMOIRE (pre-release, at COMMIT_SHA) if it does
not exist, then upload ASSET to it.  The token comes from env GH_TOKEN -- pass it on stdin
over ssh (`read -r GH_TOKEN`), never in argv or a file.  First used for v1.1 (2026-10-03).
"""
import json, os, sys, urllib.request, urllib.error
REPO = 'doopeworld/GRIMOIRE'
TAG, ASSET, SHA, NOTES = sys.argv[1:5]
NAME = sys.argv[5] if len(sys.argv) > 5 else f'GRIMOIRE {TAG}'
tok = os.environ['GH_TOKEN'].strip()
H = {'Authorization': 'token ' + tok, 'Accept': 'application/vnd.github+json', 'User-Agent': 'grimoire-release'}
BODY = open(NOTES).read()

def call(method, url, data=None, headers=None):
    h = dict(H); h.update(headers or {})
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return r.status, json.loads(r.read().decode() or '{}')
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or '{}')

st, rel = call('GET', f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}')
if st == 404:
    st, rel = call('POST', f'https://api.github.com/repos/{REPO}/releases',
                   json.dumps({'tag_name': TAG, 'target_commitish': SHA, 'name': NAME,
                               'body': BODY, 'prerelease': True}).encode(), {'Content-Type': 'application/json'})
    print('create release:', st, rel.get('html_url') or rel.get('message'))
    if st != 201: sys.exit(1)
elif st != 200:
    print('release lookup failed:', st, rel.get('message')); sys.exit(1)
else:
    print('release exists:', st, rel.get('html_url'))
name = os.path.basename(ASSET)
for a in rel.get('assets', []):
    if a['name'] == name:
        print('asset already there:', a['browser_download_url']); sys.exit(0)
size = os.path.getsize(ASSET)
print(f'uploading {name} ({size/1e6:.1f} MB) ...', flush=True)
up = rel['upload_url'].split('{')[0] + '?name=' + name
with open(ASSET, 'rb') as f:
    req = urllib.request.Request(up, data=f, method='POST',
                                 headers=dict(H, **{'Content-Type': 'application/gzip', 'Content-Length': str(size)}))
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            d = json.loads(r.read().decode())
            print('uploaded:', r.status, d.get('browser_download_url'), d.get('size'))
    except urllib.error.HTTPError as e:
        print('upload failed:', e.code, e.read()[:300]); sys.exit(1)
