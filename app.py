"""Local single-user teaching server. Run: python app.py"""
import argparse
import base64
import io
import json
import re
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from rag import Index, build_prompt, generate, citation_check

ROOT = Path(__file__).resolve().parent
STATE = {'index': None}

def small_talk_response(message):
    """Reply to common conversational openers without searching document evidence."""
    normalized = re.sub(r'[^a-z0-9\s]', '', message.casefold())
    normalized = ' '.join(normalized.split())
    replies = {
        'hi': "Hi! What would you like to know?",
        'hello': "Hello! What would you like to know?",
        'hey': "Hey! What can I help you with?",
        'hi there': "Hi there! What would you like to know?",
        'hello there': "Hello! What would you like to know?",
        'hey there': "Hey there! What can I help you with?",
        'good morning': "Good morning! What would you like to explore today?",
        'good afternoon': "Good afternoon! What can I help you with?",
        'good evening': "Good evening! What would you like to know?",
        'how are you': "I'm doing well, thanks for asking! I'm ready to help. What would you like to know?",
        'hows it going': "It's going well, thanks! What can I help you explore?",
        'what is up': "Not much, I'm ready to help! What would you like to know?",
        'whats up': "Not much, I'm ready to help! What would you like to know?",
        'thank you': "You're very welcome! Ask me anything about your documents.",
        'thank you so much': "You're very welcome! I'm here if you need anything else.",
        'thanks': "You're welcome! What else can I help you find?",
        'thanks so much': "You're very welcome! I'm here if you need anything else.",
        'thanks a lot': "You're welcome! What else can I help you find?",
        'ok': "Sounds good! What would you like to know?",
        'okay': "Sounds good! What would you like to know?",
        'cool': "Glad you think so! What else can I help you with?",
        'nice': "Thanks! What would you like to explore?",
        'great': "Great! What would you like to know?",
        'awesome': "Happy to help! What would you like to explore?",
        'sounds good': "Great! Let me know what you'd like to find.",
        'that sounds good': "Great! Let me know what you'd like to find.",
        'got it': "Got it! What would you like to explore next?",
        'who are you': "I'm your Campus Handbook Assistant. I can answer from your indexed documents with RAG, or switch to LLM-only for a general response.",
        'what can you do': "I can look for answers in your indexed documents and show the supporting passages, or you can switch to LLM-only to chat without document evidence.",
        'help': "I can help you explore your indexed documents. Ask a question about them, or switch to LLM-only for a general chat.",
        'nice to meet you': "Nice to meet you too! What would you like to explore?",
        'good to see you': "Good to see you too! What can I help you find?",
        'im good': "Glad to hear it! What would you like to explore?",
        'bye': "Goodbye! I'll be here when you need me.",
        'goodbye': "Goodbye! Come back any time you have a question.",
        'see you': "See you later! I'll be here when you need me.",
        'see you later': "See you later! I'll be here when you need me.",
        'good night': "Good night! Come back any time you have a question.",
    }
    return replies.get(normalized)

def read_uploads(files):
    docs, warnings = [], []
    for f in files:
        source = str(f['name']).replace('\\', '/').split('/')[-1][:120]
        raw = base64.b64decode(f['data'], validate=True)
        suffix = Path(source).suffix.lower()
        if suffix in ('.txt', '.md'):
            docs.append({'source': source, 'page': 1, 'text': raw.decode('utf-8-sig')})
        elif suffix == '.pdf':
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ValueError('PDF needs pypdf: python -m pip install -r requirements.txt') from exc
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                raise ValueError('Use an unencrypted PDF.')
            if len(reader.pages) > 50:
                raise ValueError('Workshop limit: 50 pages per PDF.')
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ''
                if text.strip():
                    docs.append({'source': source, 'page': number, 'text': text})
                else:
                    warnings.append('%s page %s has no text; OCR is not included.' % (source, number))
        else:
            raise ValueError('Only UTF-8 TXT, MD, or text-based PDF files are supported.')
    if sum(len(d['text']) for d in docs) > 500000:
        raise ValueError('Workshop limit: 500,000 extracted characters.')
    return docs, warnings

class Handler(BaseHTTPRequestHandler):
    def send(self, data, status=200, content_type='application/json'):
        raw = data if isinstance(data, bytes) else json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/':
            return self.send((ROOT/'index.html').read_bytes(), content_type='text/html')
        if self.path == '/api/status':
            idx = STATE['index']
            return self.send({'indexed': idx is not None, 'chunks': len(idx.chunks) if idx else 0})
        self.send({'error': 'Not found'}, 404)

    def do_POST(self):
        # Same-origin JSON requests only; server is bound to loopback.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send({'error': 'Cross-origin requests are not allowed.'}, 403)
        if self.headers.get_content_type() != 'application/json':
            return self.send({'error': 'Use application/json.'}, 415)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8_000_000:
                raise ValueError('Request must be under 8 MB.')
            body = json.loads(self.rfile.read(length))
            start = time.perf_counter()
            if self.path == '/api/index':
                STATE['index'] = None
                if body.get('sample', False):
                    documents = [{'source': p.name, 'page': 1, 'text': p.read_text(encoding='utf-8')}
                                 for p in sorted((ROOT/'data').glob('*.txt'))]
                    warnings = []
                else:
                    documents, warnings = read_uploads(body.get('files', []))
                # Failed rebuilds invalidate old state, avoiding answers from stale documents.
                idx = Index(documents, body.get('mode', 'keyword'),
                            int(body.get('size', 100)), int(body.get('overlap', 20)))
                STATE['index'] = idx
                result = {'chunks': len(idx.chunks), 'mode': idx.mode, 'warnings': warnings,
                          'preview': [vars(c) for c in idx.chunks[:5]]}
            elif self.path == '/api/ask':
                question = str(body.get('question', '')).strip()
                if not question or len(question) > 1500:
                    raise ValueError('Question must contain 1..1500 characters.')
                mode = body.get('answer_mode', 'evidence')
                if mode not in ('evidence', 'rag', 'plain'):
                    raise ValueError('Unknown answer mode.')
                intent = 'question'
                hits = []
                not_found = False
                conversational_reply = small_talk_response(question) if mode == 'rag' else None
                if conversational_reply:
                    answer = conversational_reply
                    intent = 'conversation'
                    prompt = question
                    citations = None
                elif mode == 'plain':
                    model = body.get('model', 'qwen2.5:1.5b')
                    if model not in ('qwen2.5:1.5b', 'qwen2.5:0.5b'):
                        raise ValueError('Choose one of the provided local generation models.')
                    answer = generate(question, hits, model, use_rag=False)
                    prompt = question
                    citations = None
                else:
                    idx = STATE['index']
                    if idx is None:
                        raise ValueError('Build an index first to ask questions about your documents.')
                    hits = idx.search(question, int(body.get('k', 3)))
                    # Zero keyword overlap is a useful baseline rejection; semantic scores have no universal cutoff.
                    no_overlap = idx.mode == 'keyword' and hits[0]['score'] <= 0
                    if mode == 'evidence':
                        answer = ('No matching words. I do not know from these documents.' if no_overlap
                                  else 'Retrieved passages below. This mode does not generate an answer.')
                        not_found = no_overlap
                    elif no_overlap:
                        answer = ("I couldn't find this in the uploaded documents. Try asking about "
                                  "a topic covered by the current files, or add a document that "
                                  "contains the answer.")
                        hits = []
                        not_found = True
                    else:
                        model = body.get('model', 'qwen2.5:1.5b')
                        if model not in ('qwen2.5:1.5b', 'qwen2.5:0.5b'):
                            raise ValueError('Choose one of the provided local generation models.')
                        answer = generate(question, hits, model, use_rag=True)
                        not_found = "i don't know from these documents" in answer.lower()
                    prompt = build_prompt(question, hits)
                    citations = citation_check(answer, hits) if mode == 'rag' else None
                result = {'answer': answer, 'hits': hits, 'answer_mode': mode,
                          'prompt': prompt, 'citations': citations,
                          'not_found': not_found, 'intent': intent}
            else:
                return self.send({'error': 'Not found'}, 404)
            result['seconds'] = round(time.perf_counter()-start, 3)
            self.send(result)
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            self.send({'error': str(exc)}, 400)
        except Exception as exc:
            self.send({'error': str(exc)}, 502)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    print('Campus Handbook Assistant: http://127.0.0.1:%s' % args.port)
    print('Single-user classroom demo. Ctrl+C to stop. Documents stay in memory.')
    try:
        HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
