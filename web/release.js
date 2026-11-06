// This runs after the original atlas script; no data or counts are changed.
const startYear = document.getElementById('start-year');
const endYear = document.getElementById('end-year');
Q.forEach((year, i) => {
  startYear.add(new Option(String(year), String(i)));
  endYear.add(new Option(String(i === NQ - 1 ? 1700 : year + 25), String(i)));
});
function syncPeriod() { startYear.value = String(lo); endYear.value = String(hi); }
startYear.onchange = () => { lo = Number(startYear.value); hi = Math.max(lo, hi); update(); };
endYear.onchange = () => { hi = Number(endYear.value); lo = Math.min(lo, hi); update(); };
new MutationObserver(syncPeriod).observe(document.getElementById('range'), {childList:true, subtree:true, characterData:true});
syncPeriod();
document.getElementById('reset-explore').onclick = () => {
  lo = 0; hi = NQ - 1;
  Object.keys(kindOn).forEach(k => kindOn[k] = true);
  qi.value = ''; sg.hidden = true; qi.setAttribute('aria-expanded', 'false');
  clearPick(); setEdition(null); kinds(); update(); syncPeriod();
};
document.addEventListener('keydown', event => {
  if(event.key === '/' && !event.metaKey && !event.ctrlKey && !event.altKey && !/INPUT|TEXTAREA|SELECT/.test(event.target.tagName) && !event.target.isContentEditable) {
    event.preventDefault(); qi.focus();
  }
});
const topList = document.getElementById('top');
const empty = document.createElement('p');
empty.className = 'note'; empty.textContent = 'No places match this selection. Try a wider period or reset the filters.';
topList.after(empty);
const syncEmpty = () => { empty.hidden = topList.children.length > 0 || !document.getElementById('detail').hidden; };
new MutationObserver(syncEmpty).observe(topList, {childList:true});
new MutationObserver(syncEmpty).observe(document.getElementById('detail'), {attributes:true, attributeFilter:['hidden']});
syncEmpty();
const mapContext = document.querySelector('.map-intro strong');
const filterSummary = document.querySelector('.advanced>summary');
function syncContext() {
  mapContext.textContent = mode === 'letters' ? 'Letters between places' : (aug ? 'Expanded sources' : 'Editor sources');
  const exclusions = Object.values(kindOn).filter(on => !on).length;
  filterSummary.textContent = 'Filters & display' + (exclusions ? ` · ${exclusions} place types hidden` : '') + (aug && mode !== 'letters' ? ' · expanded sources' : '');
}
new MutationObserver(syncContext).observe(document.getElementById('reading'), {attributes:true, subtree:true, attributeFilter:['aria-pressed']});
new MutationObserver(syncContext).observe(document.getElementById('modes'), {attributes:true, subtree:true, attributeFilter:['aria-pressed']});
new MutationObserver(syncContext).observe(document.getElementById('kinds'), {childList:true, attributes:true, subtree:true, attributeFilter:['aria-pressed']});
document.querySelector('.header-link').onclick = () => { document.getElementById('about-atlas').open = true; };
const fsb = document.getElementById('fsbtn');
if (fsb) {
  // Fullscreen layout follows the real Fullscreen API, the body.fs class, or F11 (detected on resize).
  let fsUser = false;  // the button asked for the layout; keeps it when the Fullscreen API is unavailable
  const f11 = () => !document.fullscreenElement &&
                   window.innerWidth >= screen.width - 2 && window.innerHeight >= screen.height - 2;
  const fsSet = () => {
    const fs = !!document.fullscreenElement || fsUser || f11();
    document.body.classList.toggle('fs', fs);
    const label = fs ? 'Exit fullscreen' : 'Fullscreen';
    fsb.setAttribute('aria-pressed', String(fs));
    fsb.textContent = label;
    fsb.title = label;
    fsb.setAttribute('aria-label', label);
  };
  fsb.onclick = () => {
    const inFs = !!document.fullscreenElement || fsUser || f11();
    fsUser = !inFs;  // one click always toggles: leaving drops the fallback layout immediately, even if the API call is still pending
    if (inFs) {
      if (document.fullscreenElement && document.exitFullscreen) { document.exitFullscreen().catch(() => {}); }
    } else if (document.documentElement.requestFullscreen) {
      document.documentElement.requestFullscreen().catch(() => {});
    }
    fsSet();
    window.dispatchEvent(new Event('resize'));  // let the map recompute its size even when no real fullscreen happens
  };
  document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement) fsUser = false;  // Esc leaves fullscreen; drop the layout too
    fsSet();
  });
  window.addEventListener('resize', fsSet);
  fsSet();
}
syncContext();

const sb = document.getElementById('sidebtn');
if (sb) {
  sb.onclick = () => {
    const m = document.querySelector('.main');
    const hide = !m.classList.contains('noside');
    m.classList.toggle('noside', hide);
    sb.setAttribute('aria-expanded', String(!hide));
    sb.textContent = hide ? 'Show panel' : 'Hide panel';
    window.dispatchEvent(new Event('resize'));  // let the map recompute its size
  };
}
