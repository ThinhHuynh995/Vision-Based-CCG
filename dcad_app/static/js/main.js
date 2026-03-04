const steps = ["Tải media","Tăng cường","Tiền xử lý 1","Tiền xử lý 2","Phân đoạn","Density map","Optical flow","DCAD+CCG","Lưu kết quả"];
const dz=document.getElementById('dropzone'), fi=document.getElementById('fileInput'), run=document.getElementById('runBtn');
let currentFile=null, timer=null, prog=0;
function renderSteps(active=-1, done=-1){const el=document.getElementById('pipelineSteps'); if(!el) return; el.innerHTML=''; steps.forEach((s,i)=>{const st=i<=done?'✓':(i===active?'⟳':'○'); el.innerHTML+=`<li>${st} ${s}</li>`})}
renderSteps();
if(dz){dz.onclick=()=>fi.click(); ['dragover','dragleave','drop'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();if(ev==='dragover')dz.classList.add('drag');if(ev==='dragleave')dz.classList.remove('drag');if(ev==='drop'){dz.classList.remove('drag');handleFile(e.dataTransfer.files[0]);}})); fi.onchange=e=>handleFile(e.target.files[0]);}
function handleFile(f){if(!f) return; currentFile=f; run.disabled=false; const p=document.getElementById('preview'); if(f.type.startsWith('image/')){p.innerHTML=`<img src="${URL.createObjectURL(f)}" style="max-width:280px;border-radius:8px">`;}else{p.textContent=`Đã chọn video: ${f.name}`;}}
run && (run.onclick=async()=>{if(!currentFile) return; document.getElementById('progressWrap').classList.remove('hidden'); prog=0; const bar=document.getElementById('progressBar'); const txt=document.getElementById('progressText'); let idx=0; timer=setInterval(()=>{prog=Math.min(90,prog+4); bar.style.width=prog+'%'; renderSteps(idx,idx-1); txt.textContent='Đang chạy: '+steps[idx%steps.length]; idx=(idx+1)%steps.length;},400);
const fd=new FormData(); fd.append('file',currentFile); fd.append('task',document.getElementById('task').value); fd.append('pipeline',document.getElementById('pipeline').value); fd.append('augment',document.getElementById('augment').value);
try{const r=await fetch('/api/process',{method:'POST',body:fd}); const data=await r.json(); clearInterval(timer); bar.style.width='100%'; renderResult(data);}catch(e){clearInterval(timer); txt.textContent='Lỗi: '+e.message;}});
function renderResult(data){document.getElementById('resultSection').classList.remove('hidden'); const s=data.summary; document.getElementById('resultBadge').innerHTML=s.alert?'<span style="color:var(--red)">⚠ Cảnh báo</span>':'<span style="color:var(--green)">✓ Bình thường</span>';
const cards=[['Điểm bất thường',s.anomaly_score],['Số người',s.person_count],['Mật độ',`${s.density_level} (${s.density_score})`],['Ngưỡng CCG',s.threshold_used],['Thời gian',data.elapsed_sec+'s'],['Loại media',data.media_type]];
document.getElementById('summaryCards').innerHTML=cards.map(c=>`<div class='card panel'><small>${c[0]}</small><b>${c[1]}</b></div>`).join('');
const order=['augmented','pre1','pre2','segmented','density','flow','result']; document.getElementById('gallery').innerHTML=order.map(k=>data.output_images[k]?`<img loading='lazy' src='${data.output_images[k]}' data-full='${data.output_images[k]}'>`:'').join('');
document.getElementById('stepLog').textContent=data.steps.map(s=>`${s.ok?'✓':'✗'} ${s.name}\n${s.info}`).join('\n\n');
updateStats();
}
async function updateStats(){const r=await fetch('/api/stats');const s=await r.json(); ['processed','alerts','normal','uploads'].forEach(k=>document.getElementById('s-'+k).textContent=s['total_'+k]??s[k]);}
window.deleteHistory=async(id)=>{await fetch('/api/history/'+id,{method:'DELETE'}); location.reload();}
document.addEventListener('click',e=>{if(e.target.matches('#gallery img')){const lb=document.getElementById('lightbox');document.getElementById('lightboxImg').src=e.target.dataset.full;lb.classList.remove('hidden');} if(e.target.id==='lightbox'){e.target.classList.add('hidden');}})
