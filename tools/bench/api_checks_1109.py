"""API integration checks: final answer streaming and native tool round trips."""
import json
from pathlib import Path
import sys
import time
import urllib.request

port, dest = int(sys.argv[1]), Path(sys.argv[2])
model = sys.argv[3] if len(sys.argv)>3 else '/models2/Qwen3.8-27B-Uncensored-FP8'
results = {}

def request(name, messages, *, stream=False, tools=None, max_tokens=512):
    body = {'model': model, 'messages': messages, 'stream': stream,
            'temperature': 0, 'max_tokens': max_tokens,
            'chat_template_kwargs': {'enable_thinking': False}}
    if tools:
        body.update(tools=tools, tool_choice='auto')
    q = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(q, timeout=120) as response:
        if stream:
            text, reasoning, finish, count = '', '', None, 0
            for line in response:
                line = line.decode().strip()
                if not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    break
                part = json.loads(payload)
                assert 'error' not in part, part.get('error')
                for choice in part.get('choices', []):
                    delta = choice.get('delta', {})
                    text += delta.get('content') or ''
                    reasoning += delta.get('reasoning_content') or ''
                    finish = choice.get('finish_reason') or finish
                count += 1
            result = {'content': text, 'reasoning_content': reasoning,
                      'finish_reason': finish, 'events': count}
        else:
            result = json.load(response)
    results[name] = {'seconds': time.monotonic() - started, 'response': result}
    dest.write_text(json.dumps(results, indent=2) + '\n')
    return result

r = request('capital', [{'role': 'user', 'content': 'What is the capital of France? Answer with only the city name.'}], max_tokens=64)
assert 'Paris' in r['choices'][0]['message']['content']
assert not r['choices'][0]['message'].get('reasoning_content')
print('PASS capital', flush=True)

r = request('streamed_writing', [{'role': 'user', 'content': 'Write about 150 words describing a quiet garden.'}], stream=True, max_tokens=768)
assert len(r['content']) > 200 and r['events'] > 1
assert not r['reasoning_content'] and r['finish_reason'] == 'stop'
print('PASS streamed final answer', flush=True)

tools = [{'type': 'function', 'function': {'name': 'get_weather',
    'description': 'Get the current weather for a city.',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
messages = [{'role': 'user', 'content': 'Use get_weather to check the weather in Paris. Do not guess.'}]
r = request('tool_call', messages, tools=tools, max_tokens=256)
choice = r['choices'][0]
assert choice['finish_reason'] == 'tool_calls'
call = choice['message']['tool_calls'][0]
assert call['function']['name'] == 'get_weather'
assert 'paris' in json.loads(call['function']['arguments'])['city'].lower()
print('PASS native tool call', flush=True)
messages += [choice['message'], {'role': 'tool', 'tool_call_id': call['id'],
    'content': json.dumps({'city': 'Paris', 'temperature_c': 18, 'condition': 'sunny'})}]
r = request('tool_result', messages, tools=tools, max_tokens=256)
assert '18' in (r['choices'][0]['message']['content'] or '')
assert r['choices'][0]['finish_reason'] == 'stop'
print('PASS tool result follow-up', flush=True)

r = request('historical_question', [{'role': 'user', 'content': 'Briefly describe the events at Tiananmen Square in 1989. Give a factual historical summary.'}], max_tokens=512)
text = r['choices'][0]['message']['content'] or ''
assert len(text) > 150
assert not any(s in text.lower() for s in ('i cannot discuss', 'i can\'t discuss', 'unable to discuss'))
print('PASS historical question (one benign refusal check)', flush=True)
print('ORCA_API_CHECKS_PASS', flush=True)
