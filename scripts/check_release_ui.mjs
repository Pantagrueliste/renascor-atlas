// Usage: CHROME=/path/to/chrome node scripts/check_release_ui.mjs URL [--mobile]
// Requires Node 22+ (built-in WebSocket); no npm dependencies.
import {spawn} from 'node:child_process';
import {mkdtempSync, readFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
const url = process.argv[2];
if (!url || !process.env.CHROME) throw new Error('Provide a URL and CHROME executable path.');
const profile = mkdtempSync(join(tmpdir(), 'atlas-smoke-'));
const browser = spawn(process.env.CHROME, ['--headless=new','--remote-debugging-port=0',`--user-data-dir=${profile}`,'--use-angle=swiftshader','--enable-unsafe-swiftshader','about:blank'], {stdio:'ignore'});
const sleep = ms => new Promise(r=>setTimeout(r,ms));
let ws;
try {
  let port;
  for (let i=0;i<80;i++) {
    try { port = readFileSync(join(profile,'DevToolsActivePort'),'utf8').split('\n')[0]; break; } catch { await sleep(250); }
  }
  if (!port) throw new Error('Chrome did not start.');
  const tabs = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  ws = new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  let id=0; const pending=new Map(); const errors=[];
  ws.onmessage = event => {
    const message=JSON.parse(event.data);
    if(message.id) { pending.get(message.id)?.(message); pending.delete(message.id); }
    if(message.method==='Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
  };
  const send = (method,params={}) => new Promise((resolve,reject)=>{
    const i=++id;
    const timer=setTimeout(()=>{pending.delete(i);reject(new Error(`Timed out: ${method}`));},30000);
    pending.set(i,m=>{clearTimeout(timer);m.error?reject(new Error(m.error.message)):resolve(m.result);});
    ws.send(JSON.stringify({id:i,method,params}));
  });
  await send('Runtime.enable');
  await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:'light'}]});
  await send('Emulation.setDeviceMetricsOverride',{width:process.argv.includes('--mobile')?390:1300,height:1000,deviceScaleFactor:1,mobile:process.argv.includes('--mobile')});
  await send('Page.navigate',{url});
  await send('Runtime.evaluate',{expression:'new Promise(resolve => { const timer=setInterval(()=>{if(typeof ready!=="undefined" && ready && map.loaded()){clearInterval(timer);resolve(true);}},100); })',awaitPromise:true});
  const expression=readFileSync(fileURLToPath(new URL('../web/smoke-checks.js',import.meta.url)),'utf8');
  const result=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
  if(result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  if(errors.length) throw new Error(errors.join('\n'));
  console.log(JSON.stringify(result.result.value,null,2));
} finally {
  ws?.close();
  if(browser.exitCode === null) { const exited=new Promise(r=>browser.once('exit',r)); browser.kill(); await exited; }
  rmSync(profile,{recursive:true,force:true});
}
