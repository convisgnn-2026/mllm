"""CPU integration checks for scheduling, scoring and drain semantics."""
import asyncio
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import httpx

MODULE=Path(__file__).resolve().parents[1]/'controlled_mixed.py'
spec=importlib.util.spec_from_file_location('controlled',MODULE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ReplayTests(unittest.TestCase):
    def test_common_clock_errors_warmup_and_drain(self):
        async def handler(req):
            if req.url.path=='/health':
                return httpx.Response(404 if req.url.host=='llm' else 200,json={'ok':True})
            if req.url.path=='/healthcheck':return httpx.Response(200,json={'ok':True})
            payload=json.loads(req.content)
            if req.url.path=='/generate':
                if payload['seed']==9:await asyncio.sleep(10)
                return httpx.Response(500,json={'error':'forced failure'})
            text='data: '+json.dumps({'choices':[{'text':'a'}]})+'\n\n'
            text+='data: '+json.dumps({'usage':{'completion_tokens':2}})+'\n\ndata: [DONE]\n\n'
            return httpx.Response(200,text=text)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);plan=p/'plan.json'
            plan.write_text(json.dumps(dict(model='mock',seed=0,warmup_s=.02,duration_s=.08,requests=[
                dict(kind='llm',idx=0,at_s=0.,measured=False,prompt='warm',in_len=8,out_len=2),
                dict(kind='llm',idx=1,at_s=.03,measured=True,prompt='hello',in_len=8,out_len=2),
                dict(kind='dit',idx=0,at_s=.03,measured=True,prompt='image',image_seed=1),
                dict(kind='dit',idx=1,at_s=.04,measured=True,prompt='timeout',image_seed=9)])))
            args=SimpleNamespace(plan=plan,out=p/'out',only='both',llm_url='http://llm',dit_url='http://dit',drain_timeout=.02)
            with redirect_stdout(io.StringIO()):report,records=asyncio.run(m.replay(args,httpx.MockTransport(handler)))
            self.assertEqual(report['requests'],4)
            self.assertEqual(report['measured']['llm']['n'],1)
            self.assertEqual(report['measured']['dit']['n'],2)
            self.assertEqual(report['measured']['dit']['errors'],2)
            self.assertEqual(report['measured']['dit']['attainment']['A'],0.)
            self.assertEqual(report['measured']['llm']['attainment']['A'],1.)
            self.assertTrue(any(r['err']=='drain_timeout' for r in records))
            simultaneous=[r for r in records if r['planned_s']==.03]
            self.assertLess(abs(simultaneous[0]['send_s']-simultaneous[1]['send_s']),.03)
            self.assertTrue((p/'out/backlog.csv').exists())
            with self.assertRaises(ValueError):asyncio.run(m.replay(args,httpx.MockTransport(handler)))

    def test_prepare_reproducible_fixed_horizon(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);trace=p/'trace.csv';trace.write_text('ContextTokens,GeneratedTokens\n128,32\n512,64\n')
            args=SimpleNamespace(trace=trace,model='mock',seed=0,duration=10.,warmup=2.,llm_rate=2.,dit_rate=.3,out=p/'a.json')
            with redirect_stdout(io.StringIO()):m.prepare(args);args.out=p/'b.json';m.prepare(args)
            self.assertEqual((p/'a.json').read_bytes(),(p/'b.json').read_bytes())
            plan=json.loads((p/'a.json').read_text())
            self.assertTrue(all(r['at_s']<12 and r['measured']==(r['at_s']>=2) for r in plan['requests']))

if __name__=='__main__':unittest.main()
