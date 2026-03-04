let session='sess_'+Math.random().toString(36).slice(2,8), featurePath='';
const log=(id,msg)=>{const el=document.getElementById(id);el.textContent += (typeof msg==='string'?msg:JSON.stringify(msg,null,2))+'\n'; el.scrollTop=el.scrollHeight;}

document.getElementById('scanBtn').onclick=async()=>{
  const sets=JSON.parse(document.getElementById('datasetsInput').value); document.getElementById('scanOut').textContent='';
  for(const s of sets){const r=await fetch('/api/training/scan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_path:s.path})}); log('scanOut',await r.json());}
};

function startSSE(url,payload,outId,onDone){const es=new EventSourcePolyfill(url,payload);es.onmessage=(e)=>{const d=JSON.parse(e.data);log(outId,d);if(d.phase==='done'){onDone&&onDone(d);es.close();}};es.onerror=()=>es.close();}
class EventSourcePolyfill{constructor(url,payload){this.es=null;fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}).then(resp=>{const reader=resp.body.getReader();const dec=new TextDecoder();let buf='';const pump=()=>reader.read().then(({done,value})=>{if(done) return;buf+=dec.decode(value,{stream:true});buf.split('\n\n').forEach(chunk=>{if(chunk.startsWith('data: ')){this.onmessage&&this.onmessage({data:chunk.slice(6)})}});buf='';return pump();});return pump();});}}

document.getElementById('extractBtn').onclick=()=>{const datasets=JSON.parse(document.getElementById('datasetsInput').value);const cfg=JSON.parse(document.getElementById('trainCfg').value);document.getElementById('streamLog').textContent='';startSSE('/api/training/extract',{session_id:session,datasets,config:cfg},'streamLog',(d)=>featurePath=d.feature_path)};
document.getElementById('trainBtn').onclick=()=>{const cfg=JSON.parse(document.getElementById('trainCfg').value);startSSE('/api/training/train',{session_id:session,feature_path:featurePath||`data/features/${session}.npz`,config:cfg},'streamLog')};
document.getElementById('evalBtn').onclick=async()=>{const r=await fetch('/api/training/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:session,use_masks:true})});document.getElementById('evalOut').textContent=JSON.stringify(await r.json(),null,2)};
