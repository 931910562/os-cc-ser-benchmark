(()=>{'use strict';
const data=window.BENCHMARK,config=window.PROJECT_CONFIG||{},names={egemaps:'eGeMAPS',resnet:'ResNet-50',wav2vec:'wav2vec 2.0',wavlm:'WavLM',hubert:'HuBERT'};
const $=id=>document.getElementById(id),text=(tag,value)=>{const n=document.createElement(tag);n.textContent=value;return n};
for(const r of data.summary){const row=text('div','');row.className='bar-row';row.append(text('span',r.method));const track=text('div','');track.className='bar-track';const fill=text('div','');fill.className='bar-fill';fill.style.width=(r.HOS/40*100)+'%';track.append(fill);row.append(track);const v=text('span',r.HOS.toFixed(2));v.className='bar-value';row.append(v);$('bars').append(row)}
$('discrepancies').textContent=data.hos_discrepancies;
for(const m of [...new Set(data.results.map(r=>r.method))]){const o=text('option',m);o.value=m;$('method').append(o)}
let page=0,sort='HOS',descending=true;const limit=15;
function render(){let rows=data.results.filter(r=>['feature','method','source','target'].every(k=>!$(k).value||r[k]===$(k).value));rows.sort((a,b)=>(descending?-1:1)*(a[sort]-b[sort]));const pages=Math.max(1,Math.ceil(rows.length/limit));page=Math.min(page,pages-1);$('results-body').replaceChildren();
for(const r of rows.slice(page*limit,(page+1)*limit)){const tr=document.createElement('tr');for(const v of [r.method,names[r.feature],r.source+' → '+r.target,r.OS_star.toFixed(2),r.UNK.toFixed(2),r.HOS.toFixed(2)])tr.append(text('td',v));$('results-body').append(tr)}
if(!rows.length){const tr=document.createElement('tr'),td=text('td','此组合没有迁移任务；请选择不同源域和目标域。');td.colSpan=6;tr.append(td);$('results-body').append(tr)}
$('result-count').textContent=rows.length+' 个配置 · 数值单位 % · 按 '+(sort==='OS_star'?'OS*':sort)+(descending?' 降序':' 升序');$('page-label').textContent=(page+1)+' / '+pages;$('prev').disabled=page===0;$('next').disabled=page>=pages-1;
}
for(const id of ['feature','method','source','target'])$(id).addEventListener('change',()=>{page=0;render()});$('reset').addEventListener('click',()=>{for(const id of ['feature','method','source','target'])$(id).value='';page=0;render()});$('prev').addEventListener('click',()=>{page--;render()});$('next').addEventListener('click',()=>{page++;render()});for(const b of document.querySelectorAll('[data-sort]'))b.addEventListener('click',()=>{if(sort===b.dataset.sort)descending=!descending;else{sort=b.dataset.sort;descending=true}page=0;render()});render();
const repository=config.repository&&/^https:\/\/github\.com\/[\w.-]+\/[\w.-]+\/?$/.test(config.repository)?config.repository.replace(/\/$/,''):'';
if(repository){for(const id of ['github-nav','source-link'])$(id).href=repository;$('publication-status').textContent='下载 v0.1 实验特征；解压后可核对文件 SHA256。'}
for(const item of window.RELEASE_ASSETS||[]){const row=text('div','');row.className='download-row';const name=item.name.replace('features-','').replace('-v0.1.zip','');const d=text('div',names[name]||'STIENet · ResNet 逐样本特征');d.append(text('small',(item.bytes/1024/1024).toFixed(1)+' MB · ZIP · 四个语料库'));row.append(d);if(repository){const a=text('a','下载 ↓');a.href=repository+'/releases/download/'+encodeURIComponent(config.releaseTag||'v0.1')+'/'+item.name;row.append(a)}else{const status=text('span','待发布');status.className='status';row.append(status)}$('downloads').append(row)}
})();
